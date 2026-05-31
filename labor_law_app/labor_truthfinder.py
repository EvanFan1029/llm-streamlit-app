from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import re
from typing import Any

from labor_law_app.normalize_labor import (
    ARTICLE_BY_LABEL,
    LABOR_OBJECTS,
    build_labor_fact_table,
)


EMPTY_FACT = "(空)"
LABOR_OBJECT_IDS = tuple(schema.object_id for schema in LABOR_OBJECTS)
LABOR_OBJECT_LABELS = {schema.object_id: schema.label for schema in LABOR_OBJECTS}
LABOR_OBJECT_MODES = {schema.object_id: schema.mode for schema in LABOR_OBJECTS}
LABOR_OBJECT_OPTIONS = {schema.object_id: tuple(schema.options) for schema in LABOR_OBJECTS}
LABOR_FACT_SETS = {schema.object_id: set(schema.options) for schema in LABOR_OBJECTS}
SINGLE_SELECT_OBJECTS = {
    schema.object_id for schema in LABOR_OBJECTS if schema.mode == "single"
}

_ADJUDICATION_PRIORITY = (
    "支持劳动者主要请求倾向",
    "支持用人单位抗辩倾向",
    "部分支持/需分项判断",
    "证据不足/需补充事实",
)
_KEY_FACT_PRIORITY = tuple(
    fact for fact in LABOR_OBJECT_OPTIONS["key_fact"] if fact != "证据不足/事实待补充"
) + ("证据不足/事实待补充",)
_TOP1_PRIORITY = {
    "adjudication_tendency": _ADJUDICATION_PRIORITY,
    "key_fact": _KEY_FACT_PRIORITY,
}

DEFAULT_MODEL_FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "qwen": ("qwen",),
    "gemma": ("gemma",),
    "mistral": ("mistral", "mixtral"),
    "llama": ("llama",),
    "deepseek": ("deepseek",),
    "gpt": ("gpt", "openai"),
    "claude": ("claude",),
}


@dataclass
class LaborTruthFinderConfig:
    t0: float = 0.75
    beta: float = 0.35
    gamma: float = 0.30
    alpha_imp: float = 0.20
    alpha_conflict: float = 0.14
    max_iter: int = 25
    early_stop: bool = True
    delta: float = 1e-4
    abs_delta: float = 1e-4
    min_iter: int = 2
    init_last_s: float = 0.5
    min_tau_scale: float = 0.35

    use_family_dependency: bool = True
    family_dep_same: float = 0.45
    family_dep_unknown: float = 0.10
    family_dep_different: float = 0.0
    model_family: dict[str, str] = field(default_factory=dict)

    use_trust_prior: bool = True
    trust_prior_default: float = 0.75
    trust_prior_by_model: dict[str, float] = field(default_factory=dict)
    trust_prior_strength: float = 2.0

    support_mode: str = "multi"
    empty_fact: str = EMPTY_FACT
    debug_relations: bool = True


def _sigmoid(x: float) -> float:
    if x >= 60:
        return 1.0
    if x <= -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _tau(t: float) -> float:
    t = min(max(float(t), 1e-6), 1.0 - 1e-6)
    return -math.log(1.0 - t)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _filter_valid_facts(object_id: str, facts: Any) -> list[str]:
    allowed = LABOR_FACT_SETS.get(object_id, set())
    if facts is None:
        raw_facts = []
    elif isinstance(facts, str):
        raw_facts = [facts]
    elif isinstance(facts, (list, tuple, set)):
        raw_facts = list(facts)
    else:
        raw_facts = [str(facts)]
    valid = [str(fact) for fact in raw_facts if str(fact) in allowed]
    return _dedupe_keep_order(valid)


def _default_model_fact_map(models: list[str]) -> dict[str, dict[str, list[str]]]:
    return {
        model: {object_id: [] for object_id in LABOR_OBJECT_IDS}
        for model in models
    }


def build_labor_model_facts(
    normalized_all: Any,
    models: list[str],
    source: str = "from_model_fields",
    exclude_fallbacks: bool = True,
    *,
    strict: bool = True,
) -> dict[str, dict[str, list[str]]]:
    model_facts = _default_model_fact_map(list(models or []))
    try:
        table = build_labor_fact_table(
            normalized_all or {},
            source=source,
            exclude_fallbacks=exclude_fallbacks,
        )
    except Exception:
        if strict:
            raise
        return model_facts

    for row in table:
        model = row.get("model")
        object_id = row.get("object_id")
        if model not in model_facts or object_id not in LABOR_FACT_SETS:
            continue
        model_facts[model][object_id] = _filter_valid_facts(object_id, row.get("facts", []))

    return model_facts


def select_labor_top1_fact(object_id: str, facts: Any) -> str | None:
    valid_facts = _filter_valid_facts(object_id, facts)
    if not valid_facts:
        return None
    if object_id in SINGLE_SELECT_OBJECTS:
        return valid_facts[0]
    priority = _TOP1_PRIORITY.get(object_id)
    if priority is None:
        priority = LABOR_OBJECT_OPTIONS.get(object_id, ())
    order = {fact: idx for idx, fact in enumerate(priority)}
    fallback_order = {
        fact: idx for idx, fact in enumerate(LABOR_OBJECT_OPTIONS.get(object_id, ()))
    }
    return min(valid_facts, key=lambda fact: (order.get(fact, 10_000), fallback_order.get(fact, 10_000)))


def _symmetric_negative_pair(
    g: str,
    f: str,
    scores: dict[tuple[str, str], float],
) -> float:
    if (g, f) in scores:
        return -scores[(g, f)]
    if (f, g) in scores:
        return -scores[(f, g)]
    return 0.0


def _article_relation(g: str, f: str) -> float:
    article_g = ARTICLE_BY_LABEL.get(g)
    article_f = ARTICLE_BY_LABEL.get(f)
    if not article_g or not article_f:
        return 0.0
    shared = set(article_g.issue_keywords) & set(article_f.issue_keywords)
    if not shared:
        return 0.0
    return min(0.35 + 0.08 * len(shared), 0.60)


def labor_fact_relation_score(
    object_id: str,
    g: str,
    f: str,
    cfg: LaborTruthFinderConfig,
) -> float:
    if g == f:
        return 0.0

    if object_id == "relationship_type":
        return _symmetric_negative_pair(
            g,
            f,
            {
                ("劳动关系倾向", "劳务/承揽关系倾向"): 0.85,
                ("劳动关系倾向", "事实不清"): 0.25,
                ("劳务/承揽关系倾向", "事实不清"): 0.20,
            },
        )

    if object_id == "adjudication_tendency":
        return _symmetric_negative_pair(
            g,
            f,
            {
                ("支持劳动者主要请求倾向", "支持用人单位抗辩倾向"): 0.90,
                ("支持劳动者主要请求倾向", "证据不足/需补充事实"): 0.30,
                ("支持用人单位抗辩倾向", "证据不足/需补充事实"): 0.30,
                ("部分支持/需分项判断", "证据不足/需补充事实"): 0.15,
            },
        )

    if object_id == "key_fact":
        if "证据不足/事实待补充" in {g, f} and g != f:
            return -0.25
        return 0.0

    if object_id == "issue_keyword":
        related_groups = (
            {"书面劳动合同", "双倍工资", "劳动关系"},
            {"违法解除", "经济补偿", "赔偿金", "举证责任"},
            {"工资支付", "加班费", "举证责任"},
            {"竞业限制", "服务期违约金"},
        )
        for group in related_groups:
            if g in group and f in group:
                return 0.25
        return 0.0

    if object_id == "article_reference":
        return _article_relation(g, f)

    return 0.0


def build_labor_relation_matrix(
    object_id: str,
    facts: list[str],
    cfg: LaborTruthFinderConfig,
) -> dict[tuple[str, str], float]:
    rel: dict[tuple[str, str], float] = {}
    unique_facts = _dedupe_keep_order([fact for fact in facts if fact])
    for g in unique_facts:
        for f in unique_facts:
            if g == f:
                continue
            score = labor_fact_relation_score(object_id, g, f, cfg)
            if score != 0.0:
                rel[(g, f)] = score
    return rel


def _matches_family_token(token: str, family: str, aliases: tuple[str, ...]) -> bool:
    if family in {"gpt", "claude"}:
        return any(token == alias or token.startswith(alias) for alias in aliases)
    return any(token == alias or token.startswith(alias) for alias in aliases)


def infer_model_family(model_name: str) -> str:
    normalized = (model_name or "").strip().lower()
    if not normalized:
        return "unknown"
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    for family, aliases in DEFAULT_MODEL_FAMILY_ALIASES.items():
        for token in tokens:
            if _matches_family_token(token, family, aliases):
                return family
    return "unknown"


def _family_for_model(model: str, cfg: LaborTruthFinderConfig) -> str:
    family = cfg.model_family.get(model)
    if family:
        return str(family).lower()
    return infer_model_family(model)


def _jaccard_pairs(
    set_a: set[tuple[tuple[str, str], str]],
    set_b: set[tuple[tuple[str, str], str]],
) -> float:
    union = set_a | set_b
    return (len(set_a & set_b) / len(union)) if union else 0.0


def compute_observed_rho_from_support(
    models: list[str],
    support: dict[tuple[tuple[str, str], str], dict[str, float]],
) -> dict[tuple[str, str], float]:
    choice_sets: dict[str, set[tuple[tuple[str, str], str]]] = {model: set() for model in models}
    for (obj, fact), by_model in (support or {}).items():
        for model, weight in (by_model or {}).items():
            if model in choice_sets and float(weight) > 0.0:
                choice_sets[model].add((obj, fact))
    rho: dict[tuple[str, str], float] = {}
    for model_a in models:
        for model_b in models:
            if model_a == model_b:
                continue
            rho[(model_a, model_b)] = _jaccard_pairs(
                choice_sets.get(model_a, set()),
                choice_sets.get(model_b, set()),
            )
    return rho


def compute_dependency_with_family(
    models: list[str],
    observed_rho: dict[tuple[str, str], float],
    cfg: LaborTruthFinderConfig,
) -> dict[tuple[str, str], float]:
    dependency: dict[tuple[str, str], float] = {}
    for model_a in models:
        for model_b in models:
            if model_a == model_b:
                continue
            observed = float(observed_rho.get((model_a, model_b), 0.0))
            if not cfg.use_family_dependency:
                dependency[(model_a, model_b)] = observed
                continue
            family_a = _family_for_model(model_a, cfg)
            family_b = _family_for_model(model_b, cfg)
            if family_a == family_b and family_a != "unknown":
                prior = float(cfg.family_dep_same)
            elif "unknown" in {family_a, family_b}:
                prior = float(cfg.family_dep_unknown)
            else:
                prior = float(cfg.family_dep_different)
            dependency[(model_a, model_b)] = max(observed, prior)
    return dependency


def _get_trust_prior(model: str, cfg: LaborTruthFinderConfig) -> float:
    prior = cfg.trust_prior_by_model.get(model, cfg.trust_prior_default)
    return min(max(float(prior), 0.01), 0.99)


def _build_support_structures(
    models: list[str],
    objects: list[tuple[str, str]],
    model_facts: dict[str, dict[str, list[str]]],
    cfg: LaborTruthFinderConfig,
) -> tuple[
    dict[tuple[tuple[str, str], str], dict[str, float]],
    dict[str, dict[tuple[str, str], str | None]],
    dict[str, dict[tuple[str, str], int]],
]:
    support: dict[tuple[tuple[str, str], str], dict[str, float]] = {}
    top1_choice: dict[str, dict[tuple[str, str], str | None]] = {model: {} for model in models}
    support_mask: dict[str, dict[tuple[str, str], int]] = {model: {} for model in models}

    for model in models:
        for obj in objects:
            object_id = obj[1]
            facts = _filter_valid_facts(object_id, model_facts.get(model, {}).get(object_id, []))
            if not facts:
                top1_choice[model][obj] = None
                support_mask[model][obj] = 0
                continue
            support_mask[model][obj] = 1
            chosen_top1 = select_labor_top1_fact(object_id, facts)
            top1_choice[model][obj] = chosen_top1

            if cfg.support_mode == "zk_top1":
                if chosen_top1 is None:
                    support_mask[model][obj] = 0
                    continue
                support.setdefault((obj, chosen_top1), {})
                support[(obj, chosen_top1)][model] = 1.0
                continue

            if object_id in SINGLE_SELECT_OBJECTS:
                chosen = facts[0]
                support.setdefault((obj, chosen), {})
                support[(obj, chosen)][model] = 1.0
                continue

            weight = 1.0 / len(facts)
            for fact in facts:
                support.setdefault((obj, fact), {})
                support[(obj, fact)][model] = support[(obj, fact)].get(model, 0.0) + weight

    return support, top1_choice, support_mask


def _build_candidate_map(
    models: list[str],
    objects: list[tuple[str, str]],
    model_facts: dict[str, dict[str, list[str]]],
    cfg: LaborTruthFinderConfig,
) -> dict[tuple[str, str], list[str]]:
    cand_map: dict[tuple[str, str], list[str]] = {}
    for obj in objects:
        object_id = obj[1]
        candidates: list[str] = []
        for model in models:
            facts = _filter_valid_facts(object_id, model_facts.get(model, {}).get(object_id, []))
            if cfg.support_mode == "zk_top1":
                top1_fact = select_labor_top1_fact(object_id, facts)
                if top1_fact is not None:
                    candidates.append(top1_fact)
                continue
            if object_id in SINGLE_SELECT_OBJECTS:
                if facts:
                    candidates.append(facts[0])
                continue
            candidates.extend(facts)
        candidates = _dedupe_keep_order(candidates)
        if not candidates:
            candidates = [cfg.empty_fact]
        cand_map[obj] = candidates
    return cand_map


def _json_key(value: Any) -> str:
    if isinstance(value, tuple):
        return " :: ".join(_json_key(item) for item in value)
    return str(value)


def make_labor_debug_jsonable(debug_info: dict[str, Any]) -> dict[str, Any]:
    support_rows: list[dict[str, Any]] = []
    for (obj, fact), by_model in sorted(
        (debug_info.get("support", {}) or {}).items(),
        key=lambda item: (item[0][0][0], item[0][0][1], item[0][1]),
    ):
        support_rows.append(
            {
                "case_id": obj[0],
                "object_id": obj[1],
                "object_label": LABOR_OBJECT_LABELS.get(obj[1], obj[1]),
                "fact": fact,
                "support_by_model": {model: float(weight) for model, weight in (by_model or {}).items()},
            }
        )
    return {
        "support_rows": support_rows,
        "dep_avg": {model: float(value) for model, value in (debug_info.get("dep_avg", {}) or {}).items()},
        "model_coverage": {model: float(value) for model, value in (debug_info.get("model_coverage", {}) or {}).items()},
        "effective_trust": {model: float(value) for model, value in (debug_info.get("effective_trust", {}) or {}).items()},
        "iter_count": int(debug_info.get("iter_count", 0) or 0),
        "support_mode": debug_info.get("support_mode"),
        "source": debug_info.get("source"),
        "exclude_fallbacks": bool(debug_info.get("exclude_fallbacks")),
        "relation_mats": {
            _json_key(obj): {_json_key(pair): float(score) for pair, score in rel.items()}
            for obj, rel in (debug_info.get("relation_mats", {}) or {}).items()
        },
    }


def labor_truthfinder_run(
    *,
    models: list[str],
    case_id: str,
    normalized_all: Any,
    cfg: LaborTruthFinderConfig | None = None,
    source: str = "from_model_fields",
    exclude_fallbacks: bool = True,
    support_mode: str | None = None,
    return_debug: bool = False,
):
    run_cfg = replace(cfg or LaborTruthFinderConfig())
    if support_mode is not None:
        run_cfg.support_mode = support_mode

    objects = [(case_id, object_id) for object_id in LABOR_OBJECT_IDS]
    model_facts = build_labor_model_facts(
        normalized_all,
        models,
        source=source,
        exclude_fallbacks=exclude_fallbacks,
    )
    cand_map = _build_candidate_map(models, objects, model_facts, run_cfg)
    support, top1_choice, support_mask = _build_support_structures(
        models,
        objects,
        model_facts,
        run_cfg,
    )

    relation_mats = {
        obj: build_labor_relation_matrix(obj[1], cand_map.get(obj, []), run_cfg)
        for obj in objects
    }
    observed_rho = compute_observed_rho_from_support(models, support)
    dependency = compute_dependency_with_family(models, observed_rho, run_cfg)
    dep_avg = {
        model: (
            sum(dependency.get((model, other), 0.0) for other in models if other != model)
            / max(1, len(models) - 1)
        )
        for model in models
    }
    model_coverage = {
        model: (
            sum(int((support_mask.get(model, {}) or {}).get(obj, 0)) for obj in objects)
            / max(1, len(objects))
        )
        for model in models
    }

    t_score = {model: _get_trust_prior(model, run_cfg) for model in models}
    s_score: dict[tuple[str, str], dict[str, float]] = {
        obj: {fact: run_cfg.init_last_s for fact in cand_map.get(obj, [])}
        for obj in objects
    }
    t_history: list[dict[str, float]] = [dict(t_score)]
    change_history: list[float] = []
    abs_change_history: list[float] = []
    iter_count = 0

    entries: dict[str, list[tuple[tuple[str, str], str, float]]] = {model: [] for model in models}
    for (obj, fact), by_model in support.items():
        for model, weight in by_model.items():
            entries.setdefault(model, []).append((obj, fact, float(weight)))

    for iteration in range(int(run_cfg.max_iter)):
        iter_count = iteration + 1
        old_t = dict(t_score)

        effective_tau = {}
        for model in models:
            dep = float(dep_avg.get(model, 0.0))
            scale = max(float(run_cfg.min_tau_scale), 1.0 - float(run_cfg.gamma) * dep)
            effective_tau[model] = _tau(old_t.get(model, run_cfg.t0)) * scale

        for obj in objects:
            facts = cand_map.get(obj, [])
            rel_map = relation_mats.get(obj, {}) or {}
            for fact in facts:
                support_by_model = support.get((obj, fact), {}) or {}
                base = sum(
                    float(weight) * effective_tau.get(model, _tau(run_cfg.t0))
                    for model, weight in support_by_model.items()
                )
                rel_effect = 0.0
                for other_fact in facts:
                    if other_fact == fact:
                        continue
                    relation = float(rel_map.get((other_fact, fact), 0.0))
                    other_score = float(s_score.get(obj, {}).get(other_fact, run_cfg.init_last_s))
                    if relation > 0.0:
                        rel_effect += run_cfg.alpha_imp * relation * other_score
                    elif relation < 0.0:
                        rel_effect += run_cfg.alpha_conflict * relation * other_score
                s_score[obj][fact] = _sigmoid(run_cfg.beta * (base + rel_effect))

        new_t: dict[str, float] = {}
        for model in models:
            model_entries = entries.get(model, [])
            if not model_entries:
                new_t[model] = old_t[model]
                continue
            numerator = 0.0
            denominator = 0.0
            for obj, fact, weight in model_entries:
                numerator += weight * float(s_score.get(obj, {}).get(fact, 0.0))
                denominator += weight
            if run_cfg.use_trust_prior:
                prior = _get_trust_prior(model, run_cfg)
                mu = max(0.0, float(run_cfg.trust_prior_strength))
                new_value = (mu * prior + numerator) / (mu + denominator)
            else:
                new_value = numerator / denominator if denominator > 0.0 else old_t[model]
            new_t[model] = min(max(float(new_value), 1e-6), 1.0 - 1e-6)

        old_vec = [old_t[model] for model in models]
        new_vec = [new_t[model] for model in models]
        max_abs_change = max(abs(new_t[model] - old_t[model]) for model in models) if models else 0.0
        change = max_abs_change if len(models) <= 1 else max(0.0, 1.0 - _cosine(old_vec, new_vec))
        change_history.append(change)
        abs_change_history.append(max_abs_change)
        t_score = new_t
        t_history.append(dict(t_score))

        if (
            run_cfg.early_stop
            and iter_count >= run_cfg.min_iter
            and change < run_cfg.delta
            and max_abs_change < run_cfg.abs_delta
        ):
            break

    effective_trust = {
        model: float(t_score.get(model, 0.0)) * float(model_coverage.get(model, 0.0))
        for model in models
    }

    if not return_debug:
        return t_score, s_score, cand_map

    debug_info = {
        "support": support,
        "top1_choice": top1_choice,
        "support_mask": support_mask,
        "relation_mats": relation_mats if run_cfg.debug_relations else {},
        "dep_avg": dep_avg,
        "model_coverage": model_coverage,
        "effective_trust": effective_trust,
        "rho": observed_rho,
        "dependency": dependency,
        "t_history": t_history,
        "change_history": change_history,
        "abs_change_history": abs_change_history,
        "iter_count": iter_count,
        "objects": objects,
        "model_facts": model_facts,
        "support_mode": run_cfg.support_mode,
        "source": source,
        "exclude_fallbacks": exclude_fallbacks,
    }
    debug_info["jsonable"] = make_labor_debug_jsonable(debug_info)
    return t_score, s_score, cand_map, debug_info


def explain_truth_per_labor_object(
    case_id: str,
    s_score: dict[tuple[str, str], dict[str, float]],
    cand_map: dict[tuple[str, str], list[str]],
    support: dict[tuple[tuple[str, str], str], dict[str, float]] | None = None,
    top_k: int = 3,
    multi_threshold: float = 0.55,
    single_margin: float = 0.03,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for object_id in LABOR_OBJECT_IDS:
        obj = (case_id, object_id)
        facts_conf = list((s_score.get(obj, {}) or {}).items())
        if not facts_conf:
            facts_conf = [(fact, 0.0) for fact in cand_map.get(obj, [EMPTY_FACT])]
        facts_conf.sort(key=lambda item: item[1], reverse=True)
        is_only_empty = len(facts_conf) == 1 and facts_conf[0][0] == EMPTY_FACT

        if is_only_empty:
            selected: list[tuple[str, float]] = []
        elif LABOR_OBJECT_MODES[object_id] == "single":
            selected = facts_conf[:1]
            if len(facts_conf) > 1 and float(facts_conf[0][1]) - float(facts_conf[1][1]) <= single_margin:
                selected = facts_conf[:2]
        else:
            selected = [item for item in facts_conf if float(item[1]) >= multi_threshold]
            if not selected:
                selected = facts_conf[: min(top_k, len(facts_conf))]
            if object_id == "article_reference":
                selected = selected[:8]

        selected_facts = {fact for fact, _ in selected}
        candidates = []
        for rank, (fact, confidence) in enumerate(facts_conf, start=1):
            support_by_model = dict((support or {}).get((obj, fact), {}) or {})
            support_weight = float(sum(support_by_model.values()))
            candidates.append(
                {
                    "rank": rank,
                    "fact": fact,
                    "confidence": float(confidence),
                    "is_selected": fact in selected_facts,
                    "is_empty": fact == EMPTY_FACT,
                    "support_weight": support_weight,
                    "support_by_model": {
                        model: float(weight) for model, weight in support_by_model.items()
                    },
                }
            )

        out.append(
            {
                "object_id": object_id,
                "object_label": LABOR_OBJECT_LABELS[object_id],
                "mode": LABOR_OBJECT_MODES[object_id],
                "is_empty": is_only_empty,
                "has_valid_result": not is_only_empty,
                "selected_facts": [fact for fact, _ in selected],
                "selected_conf": [float(conf) for _, conf in selected],
                "candidates": candidates,
            }
        )
    return out


def rank_models_by_trust(
    t_score: dict[str, float],
    *,
    effective_trust: dict[str, float] | None = None,
    use_effective: bool = False,
) -> list[tuple[str, float]]:
    score_map = effective_trust if use_effective and effective_trust is not None else t_score
    return sorted((score_map or {}).items(), key=lambda x: x[1], reverse=True)


__all__ = [
    "EMPTY_FACT",
    "LABOR_OBJECT_IDS",
    "LaborTruthFinderConfig",
    "build_labor_model_facts",
    "build_labor_relation_matrix",
    "compute_dependency_with_family",
    "compute_observed_rho_from_support",
    "explain_truth_per_labor_object",
    "labor_fact_relation_score",
    "labor_truthfinder_run",
    "make_labor_debug_jsonable",
    "rank_models_by_trust",
    "select_labor_top1_fact",
]

