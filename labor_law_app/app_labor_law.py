from __future__ import annotations

import html
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ── 路径配置 ──
_APP_FILE = Path(__file__).resolve()
_APP_DIR = _APP_FILE.parent
_PROJECT_ROOT = _APP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from labor_law_app.normalize_labor import (
    build_labor_fact_table,
    get_labor_objects,
    normalize_all_models_labor_outputs,
    extract_case_fact_patch,
)
from labor_law_app.labor_truthfinder import (
    explain_truth_per_labor_object,
    labor_truthfinder_run,
    rank_models_by_trust,
)

# ── BERT 模块 ──
_BERT_AVAILABLE = False
try:
    from labor_law_app.bert_processor import BERTProcessor
    from labor_law_app.bert_input_processor import BERTInputProcessor, CaseSemanticProfile
    from labor_law_app.bert_output_processor import BERTOutputProcessor
    from labor_law_app.bert_prompts import build_labor_prompt_v2, build_profile_hint
    from labor_law_app.bert_report_generator import BERTReportGenerator
    _BERT_AVAILABLE = True
except ImportError:
    pass

# ── ZK 模块 ──
_ZK_AVAILABLE = False
try:
    from labor_law_app.zk.zk_labor_builder import (
        collect_labor_zk_state,
        build_labor_circom_input,
        run_labor_zk_verification,
        run_full_labor_zk_pipeline,
    )
    _ZK_AVAILABLE = True
except ImportError:
    pass

st.set_page_config(
    page_title="劳动法 TruthFinder 可信聚合系统",
    page_icon="⚖️",
    layout="wide",
)

MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "mistral:7b-instruct-v0.3-q5_0",
    "gemma2:9b-instruct-q4_K_M",
    "deepseek-r1:7b",
]

OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL_LABELS = {
    "qwen2.5:7b-instruct-q4_K_M": "Qwen2.5",
    "mistral:7b-instruct-v0.3-q5_0": "Mistral",
    "gemma2:9b-instruct-q4_K_M": "Gemma2",
    "deepseek-r1:7b": "DeepSeek-R1",
}

LABOR_OBJECTS = get_labor_objects()
OBJECT_LABELS = {item["object_id"]: item["label"] for item in LABOR_OBJECTS}
OBJECT_MODES = {item["object_id"]: item["mode"] for item in LABOR_OBJECTS}

SESSION_DEFAULTS = {
    "labor_case_text": "",
    "labor_results": None,
    "labor_times": None,
    "labor_normalized_all": None,
    "labor_truthfinder_payload": None,
    "labor_final_report": "",
    "labor_bert_profile": None,
    "labor_model_running": "",
    "labor_error": "",
    # ZK
    "zk_pipeline_result": None,
    "zk_verification_ran": False,
    "zk_stage_status": "未启动",
    "zk_error": "",
    "zk_model_ids": [],
}


# ══════════════════════════════════════════════════════════════════
#  CSS + 基础渲染
# ══════════════════════════════════════════════════════════════════

def inject_css() -> None:
    st.markdown("""
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.08), transparent 26%),
            linear-gradient(180deg, #F8FCFF 0%, #FFFFFF 18%, #FFFFFF 100%);
        color: #0F172A;
        font-family: "Avenir Next", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    .block-container { max-width: 1180px; padding-top: 1.1rem; padding-bottom: 3rem; }
    .hero-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F4FBFF 100%);
        border: 1px solid #D9ECF7; border-radius: 24px;
        padding: 1.3rem 1.6rem; margin-top: 0.75rem; margin-bottom: 1rem;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.06);
    }
    .hero-card h1 { margin: 0; color: #0F172A; font-size: 2.05rem; font-weight: 750; }
    .hero-card p { margin: 0.55rem 0 0 0; color: #334155; line-height: 1.5; font-size: 0.96rem; }
    .section-card {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 18px;
        padding: 1rem 1.15rem; margin: 0.6rem 0 1rem 0;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.03);
    }
    .soft-card {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
        padding: 0.95rem 1rem; margin-bottom: 0.8rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.03);
    }
    .disclaimer-card {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.12), rgba(255, 255, 255, 0.98));
        border: 1px solid rgba(245, 158, 11, 0.32); border-radius: 18px;
        padding: 1rem 1.15rem; margin-bottom: 1rem;
    }
    .disclaimer-card h4 { margin: 0 0 0.45rem 0; color: #92400E; }
    .disclaimer-card p { margin: 0; color: #7C2D12; line-height: 1.55; }
    .step-title { color: #0284C7; font-size: 0.86rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
    .small-caption { color: #64748B; font-size: 0.82rem; }
    .muted-text { color: #475569; line-height: 1.55; }
    .status-badge {
        display: inline-block; margin: 0.2rem 0.45rem 0.2rem 0;
        padding: 0.34rem 0.75rem; border-radius: 999px; font-size: 0.84rem;
        font-weight: 650; border: 1px solid #E2E8F0;
    }
    .status-badge.is-info { background: rgba(14, 165, 233, 0.08); color: #0369A1; border-color: rgba(14, 165, 233, 0.18); }
    .status-badge.is-success { background: rgba(34, 197, 94, 0.12); color: #15803D; border-color: rgba(34, 197, 94, 0.35); }
    .status-badge.is-warning { background: rgba(245, 158, 11, 0.12); color: #B45309; border-color: rgba(245, 158, 11, 0.35); }
    .status-badge.is-error { background: rgba(239, 68, 68, 0.12); color: #B91C1C; border-color: rgba(239, 68, 68, 0.35); }
    .status-badge.is-pending { background: rgba(148, 163, 184, 0.10); color: #64748B; border-color: rgba(148, 163, 184, 0.26); }
    .light-table-wrap {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
        overflow: hidden; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.02);
    }
    .light-table { width: 100%; border-collapse: collapse; background: #FFFFFF; color: #0F172A; }
    .light-table thead tr { background: #EFF8FF; }
    .light-table th, .light-table td {
        padding: 0.78rem 0.9rem; border-bottom: 1px solid #E2E8F0;
        text-align: left; vertical-align: top; font-size: 0.94rem;
        color: #0F172A; word-break: break-word; white-space: pre-wrap;
    }
    .light-table tbody tr:hover { background: #F8FAFC; }
    .light-table tbody tr:last-child td { border-bottom: none; }
    .model-card-title { font-weight: 700; color: #0F172A; margin-bottom: 0.25rem; }
    .code-preview {
        background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 0.85rem 0.95rem; font-family: "IBM Plex Mono", "Consolas", monospace;
        font-size: 0.84rem; color: #0F172A; white-space: pre-wrap; word-break: break-word;
    }
    </style>
    """, unsafe_allow_html=True)


def model_ui_name(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def render_header() -> None:
    st.markdown("""
    <div class="hero-card">
        <h1>⚖️ 劳动法场景多模型 TruthFinder 可信聚合系统</h1>
        <p>输入劳动争议案件描述 → 四模型结构化分析 → BERT 语义对齐 → TruthFinder 可信聚合 → 律师综合报告</p>
    </div>
    """, unsafe_allow_html=True)


def render_step_header(step_no: int, title: str, caption: str = "") -> None:
    caption_html = f'<div class="small-caption" style="margin-top: 0.2rem;">{html.escape(caption)}</div>' if caption else ""
    st.markdown(f"""
    <div class="section-card">
        <div class="step-title">Step {step_no}</div>
        <h3 style="margin: 0.25rem 0 0.1rem 0; color: #0F172A;">{html.escape(title)}</h3>
        {caption_html}
    </div>
    """, unsafe_allow_html=True)


def render_light_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    header_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    st.markdown(f"""
    <div class="light-table-wrap">
        <table class="light-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


def render_labor_disclaimer() -> None:
    st.markdown("""
    <div class="disclaimer-card">
        <h4>⚖️ 法律免责声明</h4>
        <p>
            本系统为律师办案辅助工具，通过多模型 TruthFinder 可信聚合算法提供参考分析。<br/>
            不构成正式法律意见、裁判预测或对案件结果的保证。法律判断请以执业律师结合全部案件材料后的专业意见为准。<br/>
            引用法条请以国家法律法规数据库公布的最新有效文本为准。
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  状态徽章
# ══════════════════════════════════════════════════════════════════

def _badge_tone(status: str) -> str:
    if status in {"已完成", "已归一化", "已聚合", "已生成", "可用"}:
        return "is-success"
    if status in {"调用异常", "失败", "未就绪"}:
        return "is-error"
    if status in {"运行中", "生成中"}:
        return "is-warning"
    if status in {"部分失败", "解析失败"}:
        return "is-warning"
    if status in {"待输入", "待运行", "未生成"}:
        return "is-pending"
    return "is-info"


def _badge_html(label: str, status: str) -> str:
    return f'<span class="status-badge {_badge_tone(status)}">{html.escape(label)}：{html.escape(status)}</span>'


def render_flow_status() -> None:
    text_status = "已完成" if (st.session_state.get("labor_case_text") or "").strip() else "待输入"
    results = st.session_state.get("labor_results")
    normalized_all = st.session_state.get("labor_normalized_all")
    tf_payload = st.session_state.get("labor_truthfinder_payload")
    bert_profile = st.session_state.get("labor_bert_profile")
    zk_result = st.session_state.get("zk_pipeline_result")

    if results:
        has_error = any(
            (isinstance(v, str) and v.startswith("[ERROR]"))
            for v in (results or {}).values()
        )
        model_status = "部分失败" if has_error else "已完成"
    else:
        model_status = "待运行"

    bert_status = "已生成" if bert_profile else "待运行"
    norm_status = "已归一化" if normalized_all else "待运行"
    tf_status = "已聚合" if tf_payload else "待运行"
    zk_status = "已生成" if zk_result else "未生成"

    st.markdown(f"""
    <div class="section-card">
        <div class="small-caption">流程状态</div>
        <div style="margin-top: 0.45rem;">
            {_badge_html("案件描述", text_status)}
            {_badge_html("四模型调用", model_status)}
            {_badge_html("BERT画像", bert_status)}
            {_badge_html("归一化", norm_status)}
            {_badge_html("TruthFinder", tf_status)}
            {_badge_html("ZK证明", zk_status)}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  Session
# ══════════════════════════════════════════════════════════════════

def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_labor_state() -> None:
    for key in [
        "labor_results", "labor_times", "labor_normalized_all",
        "labor_truthfinder_payload", "labor_final_report",
        "labor_bert_profile", "labor_model_running", "labor_error",
        "zk_pipeline_result", "zk_verification_ran",
        "zk_stage_status", "zk_error", "zk_model_ids",
    ]:
        st.session_state[key] = SESSION_DEFAULTS[key]


# ══════════════════════════════════════════════════════════════════
#  JSON 解析
# ══════════════════════════════════════════════════════════════════

def try_parse_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        try:
            return json.loads(text[left:right + 1])
        except Exception:
            return None
    return None


def _json_dumps_pretty(data: Any) -> str:
    return json.dumps(_json_safe(data), ensure_ascii=False, indent=2)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {_stringify_key(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _stringify_key(value: Any) -> str:
    if isinstance(value, tuple):
        return " :: ".join(_stringify_key(item) for item in value)
    if isinstance(value, list):
        return " / ".join(_stringify_key(item) for item in value)
    return str(value)


def _render_json_block(data: Any) -> None:
    st.markdown(f'<div class="code-preview">{html.escape(_json_dumps_pretty(data))}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  Ollama 调用
# ══════════════════════════════════════════════════════════════════

def call_ollama_labor(model: str, prompt: str, timeout: int = 300) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一位谨慎、可靠的劳动法律师助理。你不能出具正式法律意见书，不能代替执业律师，不能对案件结果做出保证。",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "top_p": 0.9, "num_predict": 1536},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return (data.get("message", {}) or {}).get("content", "").strip()


def parse_labor_model_output(raw_output: str) -> dict[str, Any]:
    parsed = try_parse_json(raw_output)
    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "raw_output": raw_output,
            "parse_error": "无法解析为合法 JSON",
            "user_explanation": "",
            "structured_analysis": {},
            "error": "模型输出不是合法 JSON",
        }
    user_explanation = str(parsed.get("user_explanation", "") or "").strip()
    structured = parsed.get("structured_analysis", {})
    if not isinstance(structured, dict):
        return {
            "ok": False,
            "raw_output": raw_output,
            "parse_error": "structured_analysis 不是对象",
            "user_explanation": user_explanation,
            "structured_analysis": {},
            "error": "structured_analysis 不是合法对象",
        }
    return {
        "ok": True,
        "raw_output": raw_output,
        "user_explanation": user_explanation,
        "structured_analysis": structured,
    }


def _render_single_model_bert_match(model_name: str, parsed_payload: dict[str, Any]) -> None:
    """Render compact BERT matching bars for one model right after it finishes."""
    if not _BERT_AVAILABLE:
        return
    structured = (parsed_payload or {}).get("structured_analysis", {}) or {}
    if not structured:
        return

    bert = BERTProcessor.get_instance()
    if not bert.is_loaded:
        bert._ensure_loaded()
    output_proc = BERTOutputProcessor(bert)

    st.markdown(f"#### 🔎 BERT 实时语义匹配：{model_ui_name(model_name)}")

    for item in LABOR_OBJECTS:
        oid = item["object_id"]
        label = item["label"]
        raw_value = structured.get(oid)
        if raw_value is None:
            continue

        match_result = output_proc.match_model_field(oid, raw_value, model_name)
        matches = getattr(match_result, "matches", []) or []
        if not matches:
            continue

        raw_str = _format_structured_cell(raw_value)
        st.caption(f"{label}  ← 模型输出：`{html.escape(raw_str[:100])}`")

        max_sim = max(m.similarity for m in matches) if matches else 1.0
        bars_html = ""
        for m in matches[:5]:
            pct = m.similarity / max_sim if max_sim > 0 else 0
            is_selected = any(s.option == m.option for s in (getattr(match_result, "selected_matches", []) or []))
            bar_color = "#0284C7" if is_selected else ("#0EA5E9" if m.is_above_threshold else "#CBD5E1")
            bars_html += f"""
            <div style="display:flex;align-items:center;margin:1px 0;font-size:0.78rem;">
                <span style="width:200px;text-align:right;padding-right:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(m.option)}</span>
                <div style="flex:1;background:#F1F5F9;border-radius:3px;height:14px;margin:0 6px;">
                    <div style="background:{bar_color};width:{pct*100:.0f}%;height:100%;border-radius:3px;"></div>
                </div>
                <span style="width:48px;text-align:right;font-size:0.75rem;">{m.similarity:.3f}</span>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)

    st.markdown("---")


def run_model_pipeline(case_text: str, selected_models: List[str]) -> None:
    reset_labor_state()
    st.session_state["labor_case_text"] = case_text

    # ── Step 0: BERT 案件方向分析 ──
    profile_hint = ""
    if _BERT_AVAILABLE:
        with st.spinner("🧠 BERT 正在分析案件关键讨论方向..."):
            bert = BERTProcessor.get_instance()
            if not bert.is_loaded:
                bert._ensure_loaded()
            input_proc = BERTInputProcessor(bert)
            profile = input_proc.analyze_case(case_text)
            st.session_state["labor_bert_profile"] = profile.to_dict()
            profile_hint = build_profile_hint(profile)

            # Show BERT's extracted directions inline
            st.markdown("#### 🧠 BERT 案件关键方向分析")
            cols = st.columns(5)
            axes = [
                ("劳动关系信号", profile.employment_relation_score),
                ("证据充分性", profile.evidence_completeness),
                ("雇主违法程度", profile.employer_conduct_severity),
                ("法定违规可能", profile.statutory_violation_score),
                ("诉求强度", profile.claim_strength_signal),
            ]
            for i, (label, val) in enumerate(axes):
                with cols[i]:
                    st.metric(label, f"{val:.2f}")
                    st.progress(min(max(val, 0.0), 1.0))

            if profile_hint:
                st.markdown(f"""
                <div class="soft-card" style="margin-top:0.5rem;">
                    <strong>🔑 BERT 提取的关键讨论方向：</strong><br>
                    <span style="color:#334155;">{html.escape(profile_hint)}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")

    if _BERT_AVAILABLE:
        prompt = build_labor_prompt_v2(case_text, profile_hint)
    else:
        from labor_law_app.bert_prompts import build_labor_prompt
        prompt = build_labor_prompt(case_text)

    progress = st.progress(0.0)
    status = st.empty()
    results: dict[str, Any] = {}
    times: dict[str, float] = {}
    error_count = 0

    for idx, model_name in enumerate(selected_models, start=1):
        st.session_state["labor_model_running"] = model_name
        status.info(f"正在调用 {model_ui_name(model_name)} ({idx}/{len(selected_models)})")
        start = time.time()
        try:
            raw_output = call_ollama_labor(model_name, prompt)
            parsed_payload = parse_labor_model_output(raw_output)
            if not parsed_payload.get("ok"):
                error_count += 1
            results[model_name] = parsed_payload
        except Exception as ex:
            error_count += 1
            results[model_name] = {
                "ok": False,
                "raw_output": "",
                "parse_error": "",
                "user_explanation": "",
                "structured_analysis": {},
                "error": f"{type(ex).__name__}: {ex}",
            }
        times[model_name] = time.time() - start
        progress.progress(idx / len(selected_models))

        # Show BERT matching immediately after each model completes
        if results[model_name].get("ok") and _BERT_AVAILABLE:
            _render_single_model_bert_match(model_name, results[model_name])

    st.session_state["labor_model_running"] = ""
    st.session_state["labor_results"] = results
    st.session_state["labor_times"] = times
    st.session_state["labor_error"] = (
        f"共有 {error_count} 个模型未成功返回可解析结果。" if error_count else ""
    )
    if error_count:
        status.warning(st.session_state["labor_error"])
    else:
        status.success(f"全部 {len(selected_models)} 个模型调用完成。")


# ══════════════════════════════════════════════════════════════════
#  Step 1 — 案件输入
# ══════════════════════════════════════════════════════════════════

def render_step1_input() -> None:
    render_step_header(1, "案件信息输入", "输入劳动争议案件的自然语言描述")
    case_text = st.text_area(
        "案件描述",
        value=st.session_state["labor_case_text"],
        height=180,
        placeholder="例：劳动者2023年3月入职一家科技公司，一直未签书面劳动合同，2024年4月被口头辞退，主张双倍工资和违法解除赔偿金。月薪8000元，有银行工资流水但没有劳动合同。",
        key="case_text_input",
    )
    st.session_state["labor_case_text"] = case_text

    col1, col2 = st.columns(2)
    with col1:
        selected_models = st.multiselect(
            "选择要调用的模型",
            options=MODELS,
            default=MODELS[:2],
            format_func=model_ui_name,
        )
    with col2:
        use_rule_baseline = st.checkbox(
            "纯规则基线模式（不调用 LLM）",
            value=False,
            help="不调用任何大模型，仅使用规则引擎抽取事实和法条，秒级出结果。",
        )

    if st.button("⚖️ 开始四模型分析", use_container_width=True, type="primary"):
        if not case_text.strip():
            st.warning("请先输入案件描述。")
        elif use_rule_baseline:
            reset_labor_state()
            st.session_state["labor_case_text"] = case_text.strip()
            st.session_state["labor_results"] = {}
            st.session_state["labor_times"] = {}
            st.success("已切换至规则基线模式。")
        elif not selected_models:
            st.warning("请至少选择一个模型。")
        else:
            run_model_pipeline(case_text.strip(), selected_models)


# ══════════════════════════════════════════════════════════════════
#  Step 3 — 模型分析结果（user_explanation + structured_analysis）
# ══════════════════════════════════════════════════════════════════

def _format_structured_cell(value: Any) -> str:
    if isinstance(value, list):
        return " · ".join(str(item) for item in value) if value else "未返回"
    if value is None:
        return "未返回"
    text = str(value).strip()
    return text or "未返回"


def render_model_explanations(results: dict[str, Any], times: dict[str, float]) -> None:
    rows = []
    for model_name in MODELS:
        if model_name not in results:
            continue
        payload = results.get(model_name, {}) or {}
        if payload.get("ok"):
            model_status = "已完成"
            explanation = (payload.get("user_explanation") or "").strip() or "模型未返回 user_explanation"
        elif payload.get("parse_error"):
            model_status = "解析失败"
            explanation = payload.get("error") or "JSON 解析失败"
        else:
            model_status = "调用异常"
            explanation = payload.get("error") or "模型调用失败"
        rows.append({
            "模型": model_ui_name(model_name),
            "状态": model_status,
            "耗时（秒）": f"{times.get(model_name, 0.0):.2f}",
            "法律初步分析": explanation,
        })
    render_light_table(rows, ["模型", "状态", "耗时（秒）", "法律初步分析"])

    with st.expander("查看单模型原始输出与结构化分析", expanded=False):
        for model_name in MODELS:
            if model_name not in results:
                continue
            payload = results.get(model_name, {}) or {}
            model_status = "已完成" if payload.get("ok") else ("解析失败" if payload.get("parse_error") else "调用异常")
            st.markdown(f"""
            <div class="soft-card">
                <div class="model-card-title">{html.escape(model_ui_name(model_name))}</div>
                <div class="small-caption">{html.escape(model_name)}</div>
                <div style="margin-top: 0.45rem;">{_badge_html("状态", model_status)}</div>
            </div>
            """, unsafe_allow_html=True)
            if payload.get("error"):
                st.warning(payload["error"])
            if payload.get("user_explanation"):
                st.markdown("**📝 自然语言分析**")
                st.write(payload["user_explanation"])
            if payload.get("structured_analysis"):
                st.markdown("**📊 结构化判断 (structured_analysis)**")
                _render_json_block(payload["structured_analysis"])
            if payload.get("raw_output"):
                with st.expander(f"查看 {model_ui_name(model_name)} 原始输出", expanded=False):
                    st.code(payload["raw_output"][:3000], language="json")


def render_structured_comparison(results: dict[str, Any]) -> None:
    present_models = [m for m in MODELS if m in results]
    if not present_models:
        st.info("暂无模型结果")
        return
    model_cols = [model_ui_name(m) for m in present_models]

    rows: list[dict[str, Any]] = []
    for item in LABOR_OBJECTS:
        oid = item["object_id"]
        values: dict[str, str] = {}
        for model_name in present_models:
            structured = (results.get(model_name, {}) or {}).get("structured_analysis", {}) or {}
            values[model_name] = _format_structured_cell(structured.get(oid))

        # Detect disagreement: are there at least 2 distinct non-empty values?
        non_empty = [v for v in values.values() if v and v != "未返回"]
        has_disagreement = len(set(non_empty)) >= 2

        row = {"维度": item["label"]}
        for model_name in present_models:
            val = values[model_name]
            if has_disagreement and val and val != "未返回":
                row[model_ui_name(model_name)] = (
                    f'<span style="background:#FEE2E2;color:#991B1B;font-weight:700;'
                    f'font-style:italic;padding:2px 4px;border-radius:3px;">'
                    f'{html.escape(val)}</span>'
                )
            else:
                row[model_ui_name(model_name)] = html.escape(val)
        rows.append(row)

    # Render with HTML-safe cells
    header_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in (["维度"] + model_cols))
    body_rows: list[str] = []
    for row in rows:
        cells = "".join(f"<td>{row.get(col, '')}</td>" for col in (["维度"] + model_cols))
        body_rows.append(f"<tr>{cells}</tr>")
    st.markdown(f"""
    <div class="light-table-wrap">
        <table class="light-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
    </div>
    <div class="small-caption" style="margin-top: 0.4rem;">
        <span style="background:#FEE2E2;color:#991B1B;font-weight:700;font-style:italic;padding:1px 4px;border-radius:3px;">红底加粗斜体</span> = 各模型判断不一致，存在分歧，需重点关注
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  BERT 匹配过程可视化
# ══════════════════════════════════════════════════════════════════

def render_bert_matching_visualization(results: dict[str, Any]) -> None:
    if not _BERT_AVAILABLE:
        st.info("BERT 模块未加载，无法展示匹配过程")
        return
    if not results:
        return

    bert = BERTProcessor.get_instance()
    if not bert.is_loaded:
        bert._ensure_loaded()
    output_proc = BERTOutputProcessor(bert)

    for model_name in MODELS:
        if model_name not in results:
            continue
        payload = results.get(model_name, {}) or {}
        model_label = model_ui_name(model_name)
        structured = payload.get("structured_analysis", {}) or {}

        with st.expander(f"🔍 {model_label} → BERT 语义匹配详情", expanded=False):
            for item in LABOR_OBJECTS:
                oid = item["object_id"]
                label = item["label"]
                raw_value = structured.get(oid)
                if raw_value is None:
                    continue

                match_result = output_proc.match_model_field(oid, raw_value, model_name)
                if not match_result.matches:
                    continue

                # Show raw value + top matches with bars
                raw_str = _format_structured_cell(raw_value)
                st.markdown(f"**{label}** · 模型输出：`{html.escape(raw_str[:120])}`")

                bars_html = ""
                max_sim = max(m.similarity for m in match_result.matches) or 1.0
                for m in match_result.matches[:6]:
                    pct = m.similarity / max_sim if max_sim > 0 else 0
                    bar_color = "#0EA5E9" if m.is_above_threshold else "#94A3B8"
                    border = ""
                    if m in match_result.selected_matches:
                        border = "border: 2px solid #0EA5E9;"
                        bar_color = "#0284C7"
                    bars_html += f"""
                    <div style="display:flex;align-items:center;margin:2px 0;font-size:0.82rem;{border}">
                        <span style="width:220px;text-align:right;padding-right:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(m.option)}</span>
                        <div style="flex:1;background:#F1F5F9;border-radius:4px;height:18px;margin:0 8px;">
                            <div style="background:{bar_color};width:{pct*100:.0f}%;height:100%;border-radius:4px;"></div>
                        </div>
                        <span style="width:55px;text-align:right;">{m.similarity:.3f}</span>
                    </div>"""
                st.markdown(bars_html, unsafe_allow_html=True)
                st.caption("")  # spacing


# ══════════════════════════════════════════════════════════════════
#  Step 4 — BERT 案件语义画像
# ══════════════════════════════════════════════════════════════════

def render_bert_profile() -> None:
    case_text = st.session_state.get("labor_case_text", "")
    if not case_text or not _BERT_AVAILABLE:
        return

    try:
        bert_proc = BERTProcessor.get_instance()
        input_proc = BERTInputProcessor(bert_proc)
        profile = input_proc.analyze_case(case_text)
        st.session_state["labor_bert_profile"] = profile.to_dict()

        cols = st.columns(5)
        axes = [
            ("劳动关系信号", "employment_relation_score"),
            ("证据充分性", "evidence_completeness"),
            ("雇主违法程度", "employer_conduct_severity"),
            ("法定违规可能", "statutory_violation_score"),
            ("诉求强度", "claim_strength_signal"),
        ]
        for i, (label, key) in enumerate(axes):
            val = getattr(profile, key, 0.5)
            with cols[i]:
                st.metric(label, f"{val:.2f}")
                st.progress(min(max(val, 0.0), 1.0))

        # Show profile hint
        hint = build_profile_hint(profile)
        if hint:
            st.markdown(f"""
            <div class="soft-card">
                <div class="small-caption" style="margin-bottom:0.3rem;">📋 案件背景提示（已注入 LLM prompt）</div>
                <div class="muted-text">{html.escape(hint)}</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"BERT 画像生成失败: {e}")


# ══════════════════════════════════════════════════════════════════
#  Step 6 — 归一化
# ══════════════════════════════════════════════════════════════════

def _fact_table_rows(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in table:
        rows.append({
            "模型": model_ui_name(str(row.get("model", ""))),
            "维度": OBJECT_LABELS.get(str(row.get("object_id", "")), str(row.get("object_id", ""))),
            "facts": " · ".join(str(item) for item in row.get("facts", []) or []) or "(空)",
        })
    return rows


def render_normalization() -> None:
    results: dict = st.session_state.get("labor_results") or {}
    case_text: str = st.session_state.get("labor_case_text", "") or ""

    if not results or not case_text:
        # Rule baseline
        if not results and case_text:
            # Only rule baseline available
            pass
        return

    if st.button("🔬 运行归一化", use_container_width=True, key="btn_run_normalize", type="primary"):
        try:
            model_outputs = {}
            for model_name in MODELS:
                if model_name not in results:
                    continue
                payload = results.get(model_name, {}) or {}
                model_outputs[model_name] = {
                    "user_explanation": payload.get("user_explanation", ""),
                    "structured_analysis": payload.get("structured_analysis", {}),
                }

            normalized_all = normalize_all_models_labor_outputs(
                model_outputs,
                user_text=case_text,
            )
            st.session_state["labor_normalized_all"] = normalized_all
            st.session_state["labor_error"] = ""
        except Exception as ex:
            st.session_state["labor_error"] = f"归一化失败：{type(ex).__name__}: {ex}"

    if st.session_state.get("labor_error") and "归一化失败" in st.session_state["labor_error"]:
        st.error(st.session_state["labor_error"])

    normalized_all = st.session_state.get("labor_normalized_all")
    if not normalized_all:
        return

    st.markdown("""
    <div class="section-card">
        <div class="muted-text">
            <strong>三层区分：</strong> user_explanation 是模型给用户的自然语言分析；
            structured_analysis 是模型原始结构化回答；
            normalized 是前端展示用归一化结果；
            TruthFinder 默认输入来自 from_model_fields + exclude_fallbacks=True。
        </div>
    </div>
    """, unsafe_allow_html=True)

    normalized_table = build_labor_fact_table(normalized_all, source="normalized")
    with st.expander("查看前端归一化结果（source=normalized）", expanded=True):
        render_light_table(_fact_table_rows(normalized_table), ["模型", "维度", "facts"])

    tf_input_table = build_labor_fact_table(
        normalized_all, source="from_model_fields", exclude_fallbacks=True,
    )
    with st.expander("查看 TruthFinder 输入预览（from_model_fields, exclude_fallbacks=True）", expanded=False):
        render_light_table(_fact_table_rows(tf_input_table), ["模型", "维度", "facts"])

    with st.expander("查看归一化 warnings 与安全补丁", expanded=False):
        for model_name in MODELS:
            if model_name not in normalized_all:
                continue
            payload = normalized_all.get(model_name, {}) or {}
            st.markdown(f"""
            <div class="soft-card">
                <div class="model-card-title">{html.escape(model_ui_name(model_name))}</div>
                <div class="small-caption">{html.escape(model_name)}</div>
            </div>
            """, unsafe_allow_html=True)
            warnings = payload.get("warnings", []) or []
            patches = ((payload.get("patches", {}) or {}).get("from_user_text", [])) or []
            if warnings:
                st.write("warnings:")
                _render_json_block(warnings)
            else:
                st.caption("无 warnings")
            if patches:
                st.write("from_user_text patches:")
                _render_json_block(patches)
            else:
                st.caption("无 patches")


# ══════════════════════════════════════════════════════════════════
#  Step 7 — TruthFinder
# ══════════════════════════════════════════════════════════════════

def render_truthfinder() -> None:
    normalized_all = st.session_state.get("labor_normalized_all")
    results = st.session_state.get("labor_results") or {}

    if not normalized_all:
        st.info("请先完成归一化。")
        return

    # Determine which models actually participated
    active_models = [m for m in MODELS if m in normalized_all]

    if st.button("🔍 运行 TruthFinder 可信聚合", use_container_width=True, key="btn_run_tf", type="primary"):
        try:
            t_score, s_score, cand_map, debug_info = labor_truthfinder_run(
                models=active_models,
                case_id="case_0",
                normalized_all=normalized_all,
                source="from_model_fields",
                exclude_fallbacks=True,
                support_mode="multi",
                return_debug=True,
            )
            truth_rows = explain_truth_per_labor_object(
                case_id="case_0",
                s_score=s_score,
                cand_map=cand_map,
                support=debug_info.get("support"),
            )
            st.session_state["labor_truthfinder_payload"] = {
                "t_score": t_score,
                "s_score": s_score,
                "cand_map": cand_map,
                "debug_info": _json_safe(debug_info),
                "truth_rows": truth_rows,
            }
            st.session_state["labor_error"] = ""
        except Exception as ex:
            st.session_state["labor_error"] = f"TruthFinder 运行失败：{type(ex).__name__}: {ex}"

    if st.session_state.get("labor_error") and "TruthFinder 运行失败" in st.session_state["labor_error"]:
        st.error(st.session_state["labor_error"])

    tf_payload = st.session_state.get("labor_truthfinder_payload")
    if not tf_payload:
        return

    t_score = tf_payload.get("t_score", {}) or {}
    truth_rows = tf_payload.get("truth_rows", []) or []
    debug_info = tf_payload.get("debug_info", {}) or {}

    # Trust ranking
    rank_rows = [
        {"排名": idx + 1, "模型": model_ui_name(m), "可信度": f"{float(s):.4f}"}
        for idx, (m, s) in enumerate(rank_models_by_trust(t_score))
    ]
    st.markdown("#### 模型可信度排名")
    render_light_table(rank_rows, ["排名", "模型", "可信度"])

    # Object summary
    summary_rows = []
    for row in truth_rows:
        selected_facts = row.get("selected_facts", []) or []
        selected_conf = row.get("selected_conf", []) or []
        summary_rows.append({
            "维度": OBJECT_LABELS.get(row.get("object_id", ""), row.get("object_id", "")),
            "聚合可信结果": " · ".join(selected_facts) if selected_facts else "(空)",
            "置信度": " · ".join(f"{float(c):.3f}" for c in selected_conf) if selected_conf else "0.000",
            "候选数": len(row.get("candidates", []) or []),
        })
    st.markdown("#### 七个维度的聚合可信结果")
    render_light_table(summary_rows, ["维度", "聚合可信结果", "置信度", "候选数"])

    # Detail per object
    with st.expander("查看候选 fact 置信度明细", expanded=False):
        for row in truth_rows:
            st.markdown(f"**{OBJECT_LABELS.get(row.get('object_id', ''), row.get('object_id', ''))}**")
            cand_rows = []
            for c in (row.get("candidates", []) or []):
                support_by = c.get("support_by_model", {}) or {}
                support_str = " · ".join(
                    f"{model_ui_name(m)}:{float(w):.3f}"
                    for m, w in support_by.items() if float(w) > 0
                ) or "(空)"
                cand_rows.append({
                    "排名": c.get("rank"),
                    "事实": c.get("fact"),
                    "置信度": f"{float(c.get('confidence', 0)):.4f}",
                    "选中": "✅" if c.get("is_selected") else "",
                    "支持模型": support_str,
                })
            render_light_table(cand_rows, ["排名", "事实", "置信度", "选中", "支持模型"])

    # Debug
    with st.expander("查看 TruthFinder debug 信息", expanded=False):
        _render_json_block(debug_info)

    # BERT report (divergence + confidence)
    if _BERT_AVAILABLE:
        try:
            report_gen = BERTReportGenerator()
            divergences = report_gen.compute_divergence(truth_rows, t_score)
            confidences = report_gen.compute_confidence(truth_rows, t_score)

            div_rows = []
            for oid, d in divergences.items():
                div_rows.append({
                    "维度": OBJECT_LABELS.get(oid, oid),
                    "分歧度": d.divergence_level,
                    "模型一致性": f"{d.inter_model_agreement_rate:.2%}",
                    "解读": d.interpretation,
                })
            st.markdown("#### BERT 分歧度分析")
            render_light_table(div_rows, ["维度", "分歧度", "模型一致性", "解读"])

            conf_rows = []
            for oid, c in confidences.items():
                conf_rows.append({
                    "维度": OBJECT_LABELS.get(oid, oid),
                    "置信度": c.confidence_level,
                    "综合得分": f"{c.overall_confidence:.3f}",
                    "注意事项": "；".join(c.caveats) if c.caveats else "无",
                })
            st.markdown("#### BERT 置信度分析")
            render_light_table(conf_rows, ["维度", "置信度", "综合得分", "注意事项"])
        except Exception as e:
            st.warning(f"BERT 报告生成失败: {e}")


# ══════════════════════════════════════════════════════════════════
#  Step 8 — 律师综合报告（模板生成，不额外调 LLM）
# ══════════════════════════════════════════════════════════════════

def build_final_labor_report() -> str:
    tf_payload = st.session_state.get("labor_truthfinder_payload") or {}
    truth_rows = tf_payload.get("truth_rows", []) or []
    t_score = tf_payload.get("t_score", {}) or {}

    row_map = {row.get("object_id"): row for row in truth_rows}

    lines = []
    lines.append("## 劳动争议案件综合分析报告")
    lines.append("")

    # 1. 法律关系认定
    rel_row = row_map.get("relationship_type")
    rel_fact = "未确定"
    if rel_row:
        sel = rel_row.get("selected_facts", []) or []
        rel_fact = sel[0] if sel else "未确定"
    lines.append(f"**法律关系认定**：{rel_fact}")

    # 2. 裁判倾向
    adj_row = row_map.get("adjudication_tendency")
    adj_fact = "未确定"
    if adj_row:
        sel = adj_row.get("selected_facts", []) or []
        adj_fact = sel[0] if sel else "未确定"
    lines.append(f"**裁判/处理倾向初筛**：{adj_fact}")

    # 3. 核心争议
    disp_row = row_map.get("dispute_focus")
    if disp_row:
        sel = disp_row.get("selected_facts", []) or []
        if sel:
            lines.append(f"**核心争议类型**：{'、'.join(sel)}")

    # 4. 关键事实
    fact_row = row_map.get("key_fact")
    if fact_row:
        sel = fact_row.get("selected_facts", []) or []
        real = [f for f in sel if "不足" not in f and "待补充" not in f]
        if real:
            lines.append(f"**已识别关键事实**：{'、'.join(real)}")
        gaps = [f for f in sel if "不足" in f or "待补充" in f]
        if gaps:
            lines.append(f"**证据不足项**：{'、'.join(gaps)}")

    # 5. 重点法条
    art_row = row_map.get("article_reference")
    if art_row:
        sel = art_row.get("selected_facts", []) or []
        if sel:
            lines.append(f"**重点法条方向**：{'、'.join(sel[:8])}")

    # 6. 背景
    bg_row = row_map.get("background")
    if bg_row:
        sel = bg_row.get("selected_facts", []) or []
        non_default = [b for b in sel if "无特殊" not in b]
        if non_default:
            lines.append(f"**重要背景信息**：{'、'.join(non_default)}")

    lines.append("")

    # 7. 模型可信度
    n_models = len(t_score)
    if n_models > 1:
        lines.append(f"**模型可信度排行**（共 {n_models} 个模型）：")
        for m, s in sorted(t_score.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {model_ui_name(m)}：可信度 {float(s):.4f}")
    elif n_models == 1:
        m = list(t_score.keys())[0]
        lines.append(f"**注意**：当前仅使用 1 个模型（{model_ui_name(m)}），无法发挥 TruthFinder 多源交叉验证优势。建议至少使用 2 个不同模型族的模型。")

    lines.append("")

    # 8. 证据缺口 & 下一步
    all_gaps = []
    for row in truth_rows:
        for c in (row.get("candidates", []) or []):
            if c.get("is_selected") and ("不足" in str(c.get("fact", "")) or "待补充" in str(c.get("fact", ""))):
                all_gaps.append(f"{row.get('object_label', '')}: {c.get('fact', '')}")

    if all_gaps:
        lines.append("**证据缺口及补充建议**：")
        for g in all_gaps[:5]:
            lines.append(f"  - ⚠️ {g}")
        lines.append("  - 📋 建议补充：书面劳动合同、工资银行流水、考勤记录、解除通知原件、社保缴纳记录、培训服务期协议、竞业限制协议及补偿记录")
        lines.append("  - ⏰ 如涉及仲裁或诉讼，请注意劳动争议调解仲裁法第 27 条关于仲裁时效的规定（一般为一年）")

    lines.append("")
    lines.append("---")
    lines.append("**免责声明**：本报告由多模型 TruthFinder 可信聚合系统自动生成，不构成正式法律意见。法律判断请以执业律师结合全部案件材料后的专业意见为准。引用法条请以国家法律法规数据库公布的最新有效文本为准。")

    return "\n".join(lines)


def render_final_report() -> None:
    if not st.session_state.get("labor_truthfinder_payload"):
        st.info("请先完成 TruthFinder 聚合。")
        return

    # Try BERT comprehensive report first
    if _BERT_AVAILABLE:
        try:
            tf_payload = st.session_state.get("labor_truthfinder_payload") or {}
            report_gen = BERTReportGenerator()
            report = report_gen.generate_comprehensive_report(
                case_id="case_0",
                case_text=st.session_state.get("labor_case_text", ""),
                truth_rows=tf_payload.get("truth_rows", []),
                t_score=tf_payload.get("t_score", {}),
            )
            st.session_state["labor_final_report"] = report
            st.json(report.to_dict())
            return
        except Exception as e:
            st.warning(f"BERT 综合报告生成失败，回退至模板报告: {e}")

    # Fallback to template
    report_text = build_final_labor_report()
    st.session_state["labor_final_report"] = report_text
    st.text_area("律师综合报告", value=report_text, height=400, key="final_report_display")


# ══════════════════════════════════════════════════════════════════
#  Step 9 — ZK
# ══════════════════════════════════════════════════════════════════

def _reset_zk_state() -> None:
    st.session_state["zk_pipeline_result"] = None
    st.session_state["zk_verification_ran"] = False
    st.session_state["zk_stage_status"] = "未启动"
    st.session_state["zk_error"] = ""


def _zk_badge(status: str) -> str:
    cls_map = {
        "未启动": "is-pending", "运行中": "is-info", "已完成": "is-success",
        "验证通过": "is-success", "验证失败": "is-error", "已生成": "is-success",
        "失败": "is-error",
    }
    css = cls_map.get(status, "is-pending")
    return f'<span class="status-badge {css}">{status}</span>'


def render_zk() -> None:
    if not _ZK_AVAILABLE:
        st.info("零知识证明模块尚未就绪。")
        return

    truthfinder = st.session_state.get("labor_truthfinder_payload")
    if not truthfinder:
        st.info("🔒 请先完成 TruthFinder 聚合，才能进行零知识证明验证。")
        return

    trust_rank = truthfinder.get("truth_rows", []) or []
    n_models = len(st.session_state.get("labor_results", {}) or {})
    if n_models < 2:
        st.warning("⚠️ 需要至少 2 个模型才能进行有意义的 ZK 验证。")
        return

    status = st.session_state.get("zk_stage_status", "未启动")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:0.8rem;text-align:center;">
            <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.25rem;">电路状态</div>
            <div style="font-weight:700;font-size:0.95rem;">Circom v2</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:0.8rem;text-align:center;">
            <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.25rem;">证明对象</div>
            <div style="font-weight:700;font-size:0.95rem;">M={n_models} K=7 N≤8</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:0.8rem;text-align:center;">
            <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.25rem;">迭代</div>
            <div style="font-weight:700;font-size:0.95rem;">15 轮 Q16</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:0.8rem;text-align:center;">
            <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.25rem;">证明状态</div>
            {_zk_badge(status)}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.get("zk_error"):
        st.error(st.session_state["zk_error"])

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🔒 生成电路证明", use_container_width=True, type="primary",
                      disabled=(status == "运行中"), key="btn_zk_generate"):
            _reset_zk_state()
            st.session_state["zk_stage_status"] = "运行中"
            try:
                normalized_all = st.session_state["labor_normalized_all"] or {}
                case_text = st.session_state["labor_case_text"] or ""
                used_models = [m for m in MODELS if m in (st.session_state.get("labor_results") or {})]
                if len(used_models) < 2:
                    used_models = list(MODELS[:2])

                with st.status("正在生成零知识证明…", expanded=True) as zk_status:
                    st.write("1/3 正在收集 ZK 状态…")
                    pipeline_result = run_full_labor_zk_pipeline(
                        case_id="labor_case_0",
                        case_text=case_text,
                        model_ids=used_models,
                        normalized_all=normalized_all,
                        truthfinder_payload=truthfinder,
                    )
                    st.write("2/3 电路参考执行完成…")
                    st.write("3/3 正在与前端结果对比…")
                    st.session_state["zk_pipeline_result"] = pipeline_result
                    st.session_state["zk_verification_ran"] = True
                    st.session_state["zk_stage_status"] = "已完成"
                    st.session_state["zk_model_ids"] = used_models
                    zk_status.update(label="电路证明生成完成", state="complete")
            except Exception as exc:
                import traceback
                st.session_state["zk_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                st.session_state["zk_stage_status"] = "失败"
            st.rerun()

    with c2:
        zk_result = st.session_state.get("zk_pipeline_result")
        if st.button("✅ 验证一致性", use_container_width=True,
                      disabled=not (zk_result and st.session_state.get("zk_verification_ran")),
                      key="btn_zk_verify"):
            if zk_result:
                verification = zk_result.get("verification", {})
                circom_input = zk_result.get("circom_input", {})
                if verification and circom_input:
                    try:
                        re_verify = run_labor_zk_verification(circom_input, truthfinder)
                        zk_result["verification"] = re_verify
                        st.session_state["zk_pipeline_result"] = zk_result
                    except Exception as exc:
                        st.session_state["zk_error"] = f"重新验证失败: {exc}"
                st.rerun()

    with c3:
        if st.button("🗑 清除 ZK 结果", use_container_width=True, key="btn_zk_clear"):
            _reset_zk_state()
            st.rerun()

    zk_result = st.session_state.get("zk_pipeline_result")
    if not zk_result:
        return

    verification = zk_result.get("verification", {})
    if not verification.get("success"):
        st.error(f"ZK 验证失败: {verification.get('error', '未知错误')}")
        return

    st.markdown("---")
    st.markdown("### 📊 验证结果")
    verdict = verification.get("verdict", "")
    if "完全一致" in str(verdict):
        st.success(verdict)
    elif "部分一致" in str(verdict):
        st.warning(verdict)
    else:
        st.error(verdict)

    model_compare = verification.get("model_comparison", [])
    if model_compare:
        st.markdown("#### 模型可信度对比")
        mc_rows = [{
            "模型": r.get("model", "?"),
            "电路信任度": f"{r.get('ref_trust', 0):.4f}",
            "前端信任度": f"{r.get('front_trust', 0):.4f}",
            "差异": f"{r.get('delta', 0):.6f}",
            "一致性": "✅" if r.get("consistent") else "⚠️",
        } for r in model_compare]
        render_light_table(mc_rows, ["模型", "电路信任度", "前端信任度", "差异", "一致性"])

    obj_compare = verification.get("object_comparison", [])
    if obj_compare:
        st.markdown("#### 各维度胜出事实对比")
        oc_rows = [{
            "维度": r.get("object_label", "?"),
            "电路胜出": r.get("ref_winner", "—"),
            "前端胜出": r.get("front_winner", "—"),
            "一致性": "✅" if r.get("consistent") else "❌",
        } for r in obj_compare]
        render_light_table(oc_rows, ["维度", "电路胜出", "前端胜出", "一致性"])

    with st.expander("🔧 技术细节", expanded=False):
        ref_out = verification.get("reference_output", {})
        st.json({
            "best_model_idx": ref_out.get("best_model_idx"),
            "best_model_score_q16": ref_out.get("best_model_score_q16"),
            "winning_fact_idx_by_object": ref_out.get("winning_fact_idx_by_object"),
            "t_final": ref_out.get("t_final"),
        })


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    inject_css()
    init_session_state()

    render_header()
    render_labor_disclaimer()
    render_flow_status()

    # Step 1: Input
    render_step1_input()

    if st.session_state.get("labor_error") and "归一化失败" not in str(st.session_state["labor_error"]) \
            and "TruthFinder" not in str(st.session_state["labor_error"]):
        st.error(st.session_state["labor_error"])

    results = st.session_state.get("labor_results")
    times = st.session_state.get("labor_times")

    if results is not None:
        st.divider()

        if results:
            # Step 3: Model outputs
            render_step_header(3, "四模型分析结果", "优先展示 user_explanation 自然语言分析，可展开查看结构化 JSON")
            render_model_explanations(results, times or {})

            # Step 4: BERT profile
            if _BERT_AVAILABLE:
                st.divider()
                render_step_header(4, "BERT 案件语义画像", "基于 BERT 锚定法提取的 5 维语义特征")
                render_bert_profile()

            # Step 5: Structured comparison
            st.divider()
            render_step_header(5, "结构化七维度对比", "七个维度 × 各模型的原始结构化判断")
            with st.expander("查看七维度结构化判断对比", expanded=True):
                render_structured_comparison(results)

            # BERT matching visualization
            if _BERT_AVAILABLE:
                st.divider()
                render_step_header("5a", "BERT 语义匹配过程", "展示 BERT 如何将每个模型的自然语言输出映射到标准闭集选项")
                with st.expander("🔍 展开查看 BERT 匹配详情", expanded=False):
                    render_bert_matching_visualization(results)

            # Step 6: Normalization
            st.divider()
            render_step_header(6, "归一化", "normalized 用于前端展示；TruthFinder 默认输入来自 from_model_fields + exclude_fallbacks=True")
            render_normalization()

        elif results is not None and not results:
            # Rule baseline mode
            st.divider()
            render_step_header(2, "规则基线模式", "未调用 LLM，使用规则引擎直接从案件文本抽取事实和法条")
            case_text = st.session_state.get("labor_case_text", "")
            if case_text:
                try:
                    patch = extract_case_fact_patch(case_text)
                    st.markdown("#### 规则引擎抽取结果")
                    _render_json_block(patch)

                    # Auto-normalize rule baseline
                    from labor_law_app.api import analyze_labor_case
                    api_result = analyze_labor_case({"case_text": case_text})
                    st.session_state["labor_normalized_all"] = api_result.get("normalized_by_source", {})
                    st.session_state["labor_truthfinder_payload"] = api_result.get("truthfinder", {})

                    # Show rule baseline results
                    st.divider()
                    render_step_header(2, "规则基线分析结果", "基于关键词和正则匹配的直接法条映射")
                    issue_kw = api_result.get("issue_keywords", []) or []
                    articles = api_result.get("article_candidates", []) or []
                    evidence_gaps = api_result.get("evidence_gaps", []) or []
                    lawyer_brief = api_result.get("lawyer_brief", {}) or {}

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**争议关键词**")
                        if issue_kw:
                            kw_rows = [{"关键词": kw.get("keyword", kw) if isinstance(kw, dict) else str(kw),
                                        "置信度": f"{float(kw.get('confidence', 1.0)):.2f}" if isinstance(kw, dict) else "1.00"}
                                       for kw in issue_kw]
                            render_light_table(kw_rows, ["关键词", "置信度"])
                    with col_b:
                        st.markdown("**候选法条**")
                        if articles:
                            art_rows = [{"法条": a.get("article_label", a.get("label", str(a))),
                                         "来源": str(a.get("source", "规则")),
                                         "摘要": str(a.get("summary", ""))[:80]}
                                        for a in articles[:8]]
                            render_light_table(art_rows, ["法条", "来源", "摘要"])

                    if evidence_gaps:
                        st.markdown("**证据缺口**")
                        for g in evidence_gaps:
                            st.markdown(f"- ⚠️ {g}")

                    if lawyer_brief:
                        st.markdown("**律师工作摘要**")
                        st.json(lawyer_brief)
                except Exception as e:
                    st.error(f"规则基线分析失败: {e}")

    # Step 7: TruthFinder
    if st.session_state.get("labor_normalized_all"):
        st.divider()
        render_step_header(7, "TruthFinder 可信聚合", "多模型交叉验证，迭代置信度传播")
        render_truthfinder()

    # Step 8: Final report
    if st.session_state.get("labor_truthfinder_payload"):
        st.divider()
        render_step_header(8, "律师综合报告", "基于 TruthFinder 聚合结果 + BERT 分歧度/置信度分析，模板生成，不额外调用 LLM")
        render_final_report()

    # Step 9: ZK
    if st.session_state.get("labor_truthfinder_payload"):
        st.divider()
        render_step_header(9, "零知识证明验证", "Groth16 电路参考 — 验证 TruthFinder 计算完整性")
        render_zk()

    # Footer
    st.divider()
    render_labor_disclaimer()


if __name__ == "__main__":
    main()
