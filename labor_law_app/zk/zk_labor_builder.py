"""Labor-law-specific ZK input builder.

Converts labor law TruthFinder results into circom-compatible inputs for the
v2 truthfinder.circom Groth16 proof pipeline.  The circuit semantics are
shared with the translation app; this module provides the labour-domain
adaptation layer that maps the 7 labour objects, their closed-set options,
model top-1 choices, inter-fact relations and model dependencies into the
fixed-shape arrays the circuit expects.

Design
------
1.  ``collect_labor_zk_state()`` – gather all needed data from the Streamlit
    session state (normalised outputs, TruthFinder debug info).
2.  ``build_labor_circom_input()`` – produce a ``truthfinder_circom_input.json``
    payload that is directly consumable by ``prepare_circom_input.py`` and
    ``TruthFinder_circuit_ref.py``.
3.  ``run_labor_zk_verification()`` – execute the Python circuit reference,
    compare its output with the front-end TruthFinder results, and return a
    structured verdict.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Path setup – make sibling modules importable
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_ZK_DIR = _THIS_FILE.parent
_APP_DIR = _ZK_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_ZK_DIR) not in sys.path:
    sys.path.insert(0, str(_ZK_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports from the labour-law app
# ---------------------------------------------------------------------------
from labor_law_app.labor_truthfinder import (  # noqa: E402
    LABOR_OBJECT_IDS,
    LABOR_OBJECT_LABELS,
    LABOR_OBJECT_MODES,
    LABOR_OBJECT_OPTIONS,
    LaborTruthFinderConfig,
    build_labor_model_facts,
    build_labor_relation_matrix,
    compute_dependency_with_family,
    compute_observed_rho_from_support,
    labor_fact_relation_score,
    select_labor_top1_fact,
)
from labor_law_app.normalize_labor import (  # noqa: E402
    LABOR_OBJECTS,
    OBJECT_FACT_SETS,
)

# ---------------------------------------------------------------------------
# Re-use the circuit reference from the ZK directory
# ---------------------------------------------------------------------------
try:
    from .TruthFinder_circuit_ref import run_truthfinder_circuit_ref  # noqa: E402
except ImportError:
    from TruthFinder_circuit_ref import run_truthfinder_circuit_ref  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen circuit constants (must match truthfinder.circom)
# ---------------------------------------------------------------------------
Q16 = 65536
Q16_SCALE = 1 << 16
M_FIXED = 4
K_MAX_FIXED = 10
N_MAX_FIXED = 8
ITER_N_FIXED = 15

FIXED_PARAMS_Q16: Dict[str, int] = {
    "t0": 49152,
    "beta": 22938,
    "gamma": 19661,
    "alpha_imp": 13107,
    "alpha_conflict": 6554,
    "min_tau_scale": 26214,
    "init_last_s": 32768,
    "trust_prior_default": 49152,
    "trust_prior_strength": 131072,
}

EMPTY_FACT = "(空)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stable_json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(data: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(data)).hexdigest()


def _float_to_q16(value: float) -> int:
    """Convert a float in [0, 1] to Q16 integer in [0, 65536]."""
    q = int(round(value * Q16_SCALE))
    return max(0, min(Q16, q))


def _signed_float_to_q16(value: float) -> int:
    """Convert a signed float to Q16 integer in [-65536, 65536]."""
    q = int(round(value * Q16_SCALE))
    return max(-Q16, min(Q16, q))


# ---------------------------------------------------------------------------
# 1.  Collect ZK state from session / TruthFinder results
# ---------------------------------------------------------------------------
def collect_labor_zk_state(
    *,
    case_id: str,
    case_text: str,
    model_ids: Sequence[str],
    normalized_all: Mapping[str, Any],
    truthfinder_payload: Mapping[str, Any],
    debug_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Gather all data needed to build ZK circuit inputs from the labour-law
    TruthFinder results that are already in the Streamlit session state."""

    models = list(model_ids)
    if len(models) != M_FIXED:
        raise ValueError(f"ZK circuit requires exactly {M_FIXED} models, got {len(models)}")

    # --- Build model facts (closed-set options only) ---
    model_facts = build_labor_model_facts(
        normalized_all,
        models,
        source="from_model_fields",
        exclude_fallbacks=True,
        strict=True,
    )

    # --- Build candidate (fact) lists per object (must come BEFORE top1) ---
    facts_by_o: Dict[int, List[str]] = {}
    for o, object_id in enumerate(LABOR_OBJECT_IDS):
        options = list(LABOR_OBJECT_OPTIONS.get(object_id, ()))
        selected: Set[str] = set()
        for model in models:
            for fact in model_facts.get(model, {}).get(object_id, []):
                if fact in OBJECT_FACT_SETS.get(object_id, set()):
                    selected.add(fact)
        if not selected:
            selected = set(options[:1])
        facts_by_o[o] = [opt for opt in options if opt in selected]

    # --- Build top-1 choices using facts_by_o SUBSET index ---
    top1_choice: Dict[str, Dict[int, int]] = {m: {} for m in models}
    for o, object_id in enumerate(LABOR_OBJECT_IDS):
        fo = facts_by_o[o]
        for w, model in enumerate(models):
            facts = model_facts.get(model, {}).get(object_id, [])
            top1 = select_labor_top1_fact(object_id, facts)
            if top1 is not None and top1 in fo:
                top1_choice[model][o] = fo.index(top1)  # SUBSET index
            else:
                top1_choice[model][o] = 0  # fallback to first fact in subset

    # --- Build relation matrices ---
    cfg_labor = LaborTruthFinderConfig()
    relation_patches: List[Dict[str, Any]] = []
    imp_patches: List[Dict[str, Any]] = []
    conf_patches: List[Dict[str, Any]] = []

    for o, object_id in enumerate(LABOR_OBJECT_IDS):
        facts = facts_by_o[o]
        n_o = len(facts)
        if n_o <= 1:
            continue
        rel = build_labor_relation_matrix(object_id, facts, cfg_labor)
        for (g, f), score in rel.items():
            if g == f or score == 0.0:
                continue
            g_idx = facts.index(g)
            f_idx = facts.index(f)
            rel_q16 = _signed_float_to_q16(score)
            if rel_q16 == 0:
                continue
            relation_patches.append({"o": o, "g": g_idx, "f": f_idx, "value": str(rel_q16)})
            if rel_q16 > 0:
                imp_patches.append({"o": o, "g": g_idx, "f": f_idx, "value": str(rel_q16)})
            else:
                conf_patches.append({"o": o, "g": g_idx, "f": f_idx, "value": str(-rel_q16)})

    # --- Build dependency averages ---
    # Build a minimal support structure for dependency computation
    support: Dict[Tuple[Tuple[str, str], str], Dict[str, float]] = {}
    top1_text: Dict[str, Dict[Tuple[str, str], str]] = {m: {} for m in models}
    for o, object_id in enumerate(LABOR_OBJECT_IDS):
        options = list(LABOR_OBJECT_OPTIONS.get(object_id, ()))
        for w, model in enumerate(models):
            idx = top1_choice[model][o]
            fact = options[idx] if 0 <= idx < len(options) else options[0]
            key = ((case_id, object_id), fact)
            support.setdefault(key, {})[model] = 1.0
            top1_text[model][(case_id, object_id)] = fact

    observed_rho = compute_observed_rho_from_support(models, support)
    dependency = compute_dependency_with_family(models, observed_rho, cfg_labor)
    dep_avg: List[float] = []
    for model in models:
        deps = [dependency.get((model, other), 0.0) for other in models if other != model]
        dep_avg.append(sum(deps) / len(deps) if deps else 0.0)

    # --- Effective trust from debug_info or trust scores ---
    trust_scores: Dict[str, float] = {}
    if truthfinder_payload:
        for entry in truthfinder_payload.get("model_trust_rank", []) or []:
            trust_scores[str(entry.get("model", ""))] = float(entry.get("trust", 0.0))

    return {
        "case_id": case_id,
        "case_text": case_text,
        "models": models,
        "facts_by_o": {str(o): facts for o, facts in facts_by_o.items()},
        "top1_choice": {m: {str(o): idx for o, idx in choices.items()} for m, choices in top1_choice.items()},
        "relation_patches": relation_patches,
        "imp_patches": imp_patches,
        "conf_patches": conf_patches,
        "dep_avg": {m: dep_avg[i] for i, m in enumerate(models)},
        "trust_scores": trust_scores,
    }


# ---------------------------------------------------------------------------
# 2.  Build circom input from labor ZK state
# ---------------------------------------------------------------------------
def build_labor_circom_input(zk_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Produce a ``truthfinder_circom_input.json``-compatible payload from the
    labour-law ZK state dictionary returned by ``collect_labor_zk_state()``."""

    models: List[str] = list(zk_state["models"])
    facts_by_o: Dict[str, List[str]] = {str(k): list(v) for k, v in zk_state["facts_by_o"].items()}
    top1_choice: Dict[str, Dict[str, int]] = {
        m: {str(o): int(idx) for o, idx in choices.items()}
        for m, choices in zk_state["top1_choice"].items()
    }
    dep_avg_map: Dict[str, float] = dict(zk_state.get("dep_avg", {}))
    imp_patches: List[Dict[str, Any]] = list(zk_state.get("imp_patches", []))
    conf_patches: List[Dict[str, Any]] = list(zk_state.get("conf_patches", []))
    k = len(facts_by_o)

    # --- Build dense arrays (K_MAX × N_MAX × N_MAX) ---
    imp_dense: List[List[List[str]]] = [
        [["0" for _ in range(N_MAX_FIXED)] for _ in range(N_MAX_FIXED)]
        for _ in range(K_MAX_FIXED)
    ]
    conf_dense: List[List[List[str]]] = [
        [["0" for _ in range(N_MAX_FIXED)] for _ in range(N_MAX_FIXED)]
        for _ in range(K_MAX_FIXED)
    ]

    for patch in imp_patches:
        o = int(patch["o"])
        g = int(patch["g"])
        f = int(patch["f"])
        if 0 <= o < K_MAX_FIXED and 0 <= g < N_MAX_FIXED and 0 <= f < N_MAX_FIXED:
            imp_dense[o][g][f] = str(patch["value"])

    for patch in conf_patches:
        o = int(patch["o"])
        g = int(patch["g"])
        f = int(patch["f"])
        if 0 <= o < K_MAX_FIXED and 0 <= g < N_MAX_FIXED and 0 <= f < N_MAX_FIXED:
            conf_dense[o][g][f] = str(patch["value"])

    # --- Build flattened arrays ---
    fact_count_by_object: List[int] = []
    is_effective_by_object: List[int] = []
    top1_choice_flat: List[int] = []

    for o in range(K_MAX_FIXED):
        key = str(o)
        if o < k:
            facts = facts_by_o.get(key, [])
            fc = len(facts)
            fact_count_by_object.append(fc)
            is_effective_by_object.append(1)
            for model in models:
                idx = top1_choice.get(model, {}).get(key, 0)
                idx = max(0, min(idx, fc - 1)) if fc > 0 else -1
                top1_choice_flat.append(idx if fc > 0 else -1)
        else:
            fact_count_by_object.append(0)
            is_effective_by_object.append(0)
            for _ in models:
                top1_choice_flat.append(-1)

    # --- Build dep_avg array ---
    dep_avg_arr: List[str] = []
    for model in models:
        val = _float_to_q16(dep_avg_map.get(model, 0.0))
        dep_avg_arr.append(str(val))

    # --- Build imp_flat / conf_flat ---
    imp_flat: List[str] = []
    conf_flat: List[str] = []
    for o in range(K_MAX_FIXED):
        for g in range(N_MAX_FIXED):
            for f_item in range(N_MAX_FIXED):
                imp_flat.append(imp_dense[o][g][f_item])
                conf_flat.append(conf_dense[o][g][f_item])

    # --- Assemble circom input ---
    params_q16_str = {k: str(v) for k, v in FIXED_PARAMS_Q16.items()}

    circom_arrays: Dict[str, Any] = {
        "K": k,
        "model_ids": models,
        "dep_avg": dep_avg_arr,
        "imp_flat": imp_flat,
        "conf_flat": conf_flat,
    }

    object_meta = {
        "fact_count_by_object": fact_count_by_object,
        "is_effective_by_object": is_effective_by_object,
        "top1_choice_flat": top1_choice_flat,
    }

    provenance = {
        "generator": "labor_law_app.zk.zk_labor_builder",
        "facts_hash": _sha256_hex({"facts_by_o": facts_by_o}),
        "top1_hash": _sha256_hex({"top1_choice": top1_choice}),
        "imp_hash": _sha256_hex({"imp_patches": imp_patches}),
        "conf_hash": _sha256_hex({"conf_patches": conf_patches}),
        "dep_hash": _sha256_hex({"dep_avg": dep_avg_map}),
    }

    return {
        "shape": {
            "M": M_FIXED,
            "K_MAX": K_MAX_FIXED,
            "N_MAX": N_MAX_FIXED,
            "ITER_N": ITER_N_FIXED,
        },
        "fixed_point": {
            "format": "Q16",
            "scale_pow2": 16,
            "scale": Q16,
        },
        "runtime": {
            "K": k,
            "model_ids": models,
            "facts_by_o": {str(o): facts for o, facts in facts_by_o.items()},
            "session_id": zk_state.get("case_id", ""),
            "sentence_id": zk_state.get("case_id", ""),
            "input_text": zk_state.get("case_text", ""),
            "support_mode": "top1_in_circuit",
        },
        "params_q16": params_q16_str,
        "params_meta": {
            "topn_candidates": 3,
            "max_iter": ITER_N_FIXED,
            "zk_iter_mode": "fixed_15_rounds",
            "use_trust_prior": True,
            "trust_prior_default": 0.75,
            "trust_prior_strength": 2.0,
            "support_mode": "top1_in_circuit",
        },
        "object_meta": object_meta,
        "circom_arrays": circom_arrays,
        "provenance": provenance,
        "constraints": {},
    }


# ---------------------------------------------------------------------------
# 3.  Run circuit reference and compare with front-end results
# ---------------------------------------------------------------------------
def run_labor_zk_verification(
    circom_input: Mapping[str, Any],
    frontend_truthfinder: Mapping[str, Any],
) -> Dict[str, Any]:
    """Execute the Python circuit reference and compare its output with the
    front-end TruthFinder results.

    Returns
    -------
    dict with keys:
        success : bool
        reference_output : dict  – raw circuit-reference output
        comparison : dict       – per-object and per-model comparison
        verdict : str           – "一致" | "部分一致" | "不一致"
        details : list[str]     – human-readable comparison notes
    """

    # Deep-copy to avoid mutating the caller's dict
    circom_payload = copy.deepcopy(dict(circom_input))

    # Run the circuit reference
    try:
        ref_output = run_truthfinder_circuit_ref(circom_payload)
    except Exception as exc:
        return {
            "success": False,
            "error": f"Circuit reference execution failed: {exc}",
            "reference_output": None,
            "comparison": {},
            "verdict": "错误",
            "details": [str(exc)],
        }

    # --- Extract front-end results ---
    front_trust_rank: List[Dict[str, Any]] = frontend_truthfinder.get("model_trust_rank", []) or []
    front_object_results: List[Dict[str, Any]] = frontend_truthfinder.get("object_results", []) or []

    # Build model name → index mapping
    model_ids = circom_input.get("runtime", {}).get("model_ids", [])
    if not model_ids:
        model_ids = [f"model_{i}" for i in range(M_FIXED)]
    model_to_idx = {m: i for i, m in enumerate(model_ids)}

    # --- Compare model trust rankings ---
    ref_best_idx = ref_output.get("best_model_idx", 0)
    ref_best_model = model_ids[ref_best_idx] if 0 <= ref_best_idx < len(model_ids) else "?"
    ref_t_final = ref_output.get("t_final", [0.0] * M_FIXED)

    front_trust_map: Dict[str, float] = {}
    for entry in front_trust_rank:
        front_trust_map[str(entry.get("model", ""))] = float(entry.get("trust", 0.0))

    model_compare_rows: List[Dict[str, Any]] = []
    for model in model_ids:
        idx = model_to_idx.get(model, -1)
        ref_trust = ref_t_final[idx] / Q16 if 0 <= idx < len(ref_t_final) else 0.0
        front_trust = front_trust_map.get(model, 0.0)
        diff = abs(ref_trust - front_trust)
        model_compare_rows.append({
            "model": model,
            "ref_trust_q16": ref_t_final[idx] if 0 <= idx < len(ref_t_final) else 0,
            "ref_trust": round(ref_trust, 6),
            "front_trust": round(front_trust, 6),
            "delta": round(diff, 6),
            "consistent": diff < 0.05,
        })

    # --- Compare per-object winning facts ---
    ref_winning = ref_output.get("winning_fact_idx_by_object", [])
    object_meta = circom_input.get("object_meta", {})
    fact_counts = object_meta.get("fact_count_by_object", [])
    k = circom_input.get("runtime", {}).get("K", 0)

    # Build fact lists per object from the circom input's facts_by_o (subset)
    facts_by_o_verif = circom_input.get("runtime", {}).get("facts_by_o", {}) or {}
    facts_per_obj: Dict[int, List[str]] = {}
    for o in range(min(k, len(LABOR_OBJECT_IDS))):
        key = str(o)
        if key in facts_by_o_verif and facts_by_o_verif[key]:
            facts_per_obj[o] = list(facts_by_o_verif[key])
        else:
            # Fallback: full option list
            facts_per_obj[o] = list(LABOR_OBJECT_OPTIONS.get(LABOR_OBJECT_IDS[o], ()))

    object_compare_rows: List[Dict[str, Any]] = []
    for o in range(min(k, len(LABOR_OBJECT_IDS))):
        object_id = LABOR_OBJECT_IDS[o]
        object_label = LABOR_OBJECT_LABELS.get(object_id, object_id)

        # Circuit reference winner
        ref_win_idx = ref_winning[o] if o < len(ref_winning) else 0
        ref_fact = ""
        if o in facts_per_obj and 0 <= ref_win_idx < len(facts_per_obj[o]):
            ref_fact = facts_per_obj[o][ref_win_idx]

        # Front-end winner
        front_row = next((r for r in front_object_results if r.get("object_id") == object_id), None)
        front_selected = (front_row or {}).get("selected_facts", []) or []
        front_winner = front_selected[0] if front_selected else ""

        consistent = (ref_fact == front_winner)
        object_compare_rows.append({
            "object_id": object_id,
            "object_label": object_label,
            "ref_winner": ref_fact,
            "front_winner": front_winner,
            "consistent": consistent,
        })

    # --- Overall verdict ---
    obj_consistent = sum(1 for r in object_compare_rows if r["consistent"])
    obj_total = len(object_compare_rows)
    model_consistent = sum(1 for r in model_compare_rows if r["consistent"])
    model_total = len(model_compare_rows)

    if obj_consistent == obj_total and model_consistent == model_total:
        verdict = "✅ 完全一致 — 电路参考输出与前端 TruthFinder 结果一致"
    elif obj_consistent >= obj_total * 0.7:
        verdict = "⚠️ 部分一致 — 存在可接受的浮点/近似差异"
    else:
        verdict = "❌ 不一致 — 电路参考输出与前端结果存在显著差异，需检查输入"

    details: List[str] = []
    details.append(f"模型可信度一致: {model_consistent}/{model_total}")
    for r in model_compare_rows:
        icon = "✅" if r["consistent"] else "⚠️"
        details.append(
            f"  {icon} {r['model']}: 电路={r['ref_trust']:.4f}  前端={r['front_trust']:.4f}  Δ={r['delta']:.4f}"
        )
    details.append(f"对象胜出事实一致: {obj_consistent}/{obj_total}")
    for r in object_compare_rows:
        icon = "✅" if r["consistent"] else "❌"
        details.append(
            f"  {icon} {r['object_label']}: 电路={r['ref_winner']!r}  前端={r['front_winner']!r}"
        )

    return {
        "success": True,
        "reference_output": ref_output,
        "model_comparison": model_compare_rows,
        "object_comparison": object_compare_rows,
        "verdict": verdict,
        "details": details,
    }


# ---------------------------------------------------------------------------
# 4.  Full ZK pipeline (build → verify → report)
# ---------------------------------------------------------------------------
def run_full_labor_zk_pipeline(
    *,
    case_id: str,
    case_text: str,
    model_ids: Sequence[str],
    normalized_all: Mapping[str, Any],
    truthfinder_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """One-shot: collect state → build circom input → run circuit reference
    → compare with front-end results.  Returns a comprehensive report dict."""

    zk_state = collect_labor_zk_state(
        case_id=case_id,
        case_text=case_text,
        model_ids=model_ids,
        normalized_all=normalized_all,
        truthfinder_payload=truthfinder_payload,
    )

    circom_input = build_labor_circom_input(zk_state)

    verification = run_labor_zk_verification(
        circom_input=circom_input,
        frontend_truthfinder=truthfinder_payload,
    )

    return {
        "zk_state": zk_state,
        "circom_input": circom_input,
        "verification": verification,
    }


__all__ = [
    "Q16",
    "M_FIXED",
    "K_MAX_FIXED",
    "N_MAX_FIXED",
    "ITER_N_FIXED",
    "FIXED_PARAMS_Q16",
    "collect_labor_zk_state",
    "build_labor_circom_input",
    "run_labor_zk_verification",
    "run_full_labor_zk_pipeline",
]
