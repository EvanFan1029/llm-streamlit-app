from __future__ import annotations

import html
import json
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
)
from labor_law_app.labor_truthfinder import (
    explain_truth_per_labor_object,
    labor_truthfinder_run,
    rank_models_by_trust,
)
from labor_law_app.api import analyze_labor_case

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
    "labor_model_running": "",
    "labor_error": "",
    "labor_bert_analysis": None,
}


def init_session():
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def inject_css():
    st.markdown("""
    <style>
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
    .step-title { color: #0284C7; font-size: 0.86rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
    .small-caption { color: #64748B; font-size: 0.82rem; }
    .muted-text { color: #475569; line-height: 1.55; }
    .status-badge {
        display: inline-block; margin: 0.2rem 0.45rem 0.2rem 0;
        padding: 0.34rem 0.75rem; border-radius: 999px; font-size: 0.84rem;
        font-weight: 650; border: 1px solid #E2E8F0;
    }
    .status-badge.is-success { background: rgba(34, 197, 94, 0.12); color: #15803D; border-color: rgba(34, 197, 94, 0.35); }
    .status-badge.is-warning { background: rgba(245, 158, 11, 0.12); color: #B45309; border-color: rgba(245, 158, 11, 0.35); }
    .status-badge.is-error { background: rgba(239, 68, 68, 0.12); color: #B91C1C; border-color: rgba(239, 68, 68, 0.35); }
    .status-badge.is-info { background: rgba(14, 165, 233, 0.08); color: #0369A1; border-color: rgba(14, 165, 233, 0.18); }
    .status-badge.is-pending { background: rgba(148, 163, 184, 0.10); color: #64748B; border-color: rgba(148, 163, 184, 0.26); }
    .light-table-wrap {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
        overflow: hidden; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.02);
    }
    .light-table { width: 100%; border-collapse: collapse; background: #FFFFFF; }
    .light-table thead tr { background: #EFF8FF; }
    .light-table th, .light-table td {
        padding: 0.78rem 0.9rem; border-bottom: 1px solid #E2E8F0;
        text-align: left; vertical-align: top; font-size: 0.94rem;
    }
    .light-table tbody tr:hover { background: #F8FAFC; }
    .light-table tbody tr:last-child td { border-bottom: none; }
    .model-card-title { font-weight: 700; margin-bottom: 0.25rem; }
    .code-preview {
        background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 0.85rem 0.95rem; font-family: "IBM Plex Mono", "Consolas", monospace;
        font-size: 0.84rem; white-space: pre-wrap; word-break: break-word;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="hero-card">
        <h1>⚖️ 劳动法场景多模型 TruthFinder 可信聚合系统</h1>
        <p>输入劳动争议案件描述 → 四模型结构化分析 → BERT 语义匹配 → TruthFinder 可信聚合 → 律师综合报告</p>
    </div>
    """, unsafe_allow_html=True)


def render_step_header(step_no: int, title: str, caption: str = ""):
    caption_html = f'<div class="small-caption" style="margin-top: 0.2rem;">{html.escape(caption)}</div>' if caption else ""
    st.markdown(f"""
    <div class="section-card">
        <div class="step-title">Step {step_no}</div>
        <h3 style="margin: 0.25rem 0 0.1rem 0; color: #0F172A;">{html.escape(title)}</h3>
        {caption_html}
    </div>
    """, unsafe_allow_html=True)


def render_light_table(rows: list[dict[str, Any]], columns: list[str]):
    header_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    st.markdown(f"""
    <div class="light-table-wrap">
        <table class="light-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{"".join(body_rows)}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


def model_ui_name(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def call_ollama(model: str, prompt: str, timeout: int = 180) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 1024},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except Exception as e:
        return f"[ERROR] {model}: {e}"


def run_all_models(prompt: str, selected_models: List[str]) -> tuple[Dict[str, str], Dict[str, float]]:
    results: Dict[str, str] = {}
    times: Dict[str, float] = {}

    for i, model in enumerate(selected_models):
        t0 = time.time()
        label = model_ui_name(model)
        st.info(f"🔄 正在调用 {label} ({i+1}/{len(selected_models)})...")
        results[model] = call_ollama(model, prompt)
        times[model] = time.time() - t0
        if results[model].startswith("[ERROR]"):
            st.warning(f"❌ {label}: {results[model]}")
        else:
            st.success(f"✅ {label} ({times[model]:.0f}s)")

        # Unload model immediately after use
        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=3,
            )
        except Exception:
            pass

    return results, times


def parse_model_results(raw_results: Dict[str, str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for model, raw in raw_results.items():
        if raw.startswith("[ERROR]"):
            parsed[model] = {"case_summary": raw, "structured_analysis": {}}
            continue
        c = raw.strip()
        if c.startswith("```"):
            lines = c.split("\n")
            c = "\n".join(lines[1:-1]) if len(lines) > 2 else c
        try:
            parsed[model] = json.loads(c)
        except Exception:
            parsed[model] = {"case_summary": c[:500], "structured_analysis": {}}
    return parsed


def run_analysis(case_text: str, use_bert: bool, manual_outputs: Optional[str] = None,
                  selected_models: Optional[List[str]] = None):
    st.session_state.labor_error = ""
    st.session_state.labor_results = None
    st.session_state.labor_bert_analysis = None

    try:
        if manual_outputs:
            try:
                model_outputs = json.loads(manual_outputs)
            except Exception:
                st.session_state.labor_error = "手动输入的模型输出格式不正确，请检查 JSON 格式"
                return
            results = {}
            times = {}
            for model, raw in model_outputs.items():
                results[model] = json.dumps(raw, ensure_ascii=False) if not isinstance(raw, str) else raw
                times[model] = 0.0
        elif selected_models:
            from labor_law_app.bert_prompts import build_labor_prompt as blp
            prompt = blp(case_text)
            results, times = run_all_models(prompt, selected_models)
        else:
            results = {}
            times = {}

        st.session_state.labor_results = results
        st.session_state.labor_times = times
        parsed = parse_model_results(results)

        api_result = analyze_labor_case({
            "case_text": case_text,
            "model_outputs": parsed,
            "use_bert": use_bert,
        })

        st.session_state.labor_normalized_all = api_result.get("normalized_by_source", {})
        st.session_state.labor_truthfinder_payload = api_result.get("truthfinder", {})
        st.session_state.labor_bert_analysis = api_result.get("_bert_analysis", {})
    except Exception as e:
        st.session_state.labor_error = f"分析过程出错: {e}"
        import traceback
        st.session_state.labor_error += "\n" + traceback.format_exc()


def render_step1_input():
    render_step_header(1, "案件信息输入", "输入劳动争议案件描述")
    case_text = st.text_area(
        "案件描述",
        value=st.session_state.labor_case_text,
        height=150,
        placeholder="例：劳动者2023年3月入职一家科技公司，一直未签书面劳动合同，2024年4月被口头辞退，主张双倍工资和违法解除赔偿金。月薪8000元，有银行工资流水但没有劳动合同。",
        key="case_text_input",
    )
    st.session_state.labor_case_text = case_text

    use_bert = st.checkbox("启用 BERT 语义分析", value=False,
                           help="首次加载模型约 30 秒。生成案件语义画像 + 分歧度/置信度报告。")

    col1, col2 = st.columns(2)
    with col1:
        use_ollama = st.checkbox("调用本地 Ollama 模型", value=False,
                                 help="CPU 推理较慢，每模型约 2-5 分钟。不勾选则使用规则基线，秒出结果。")
    selected_models: List[str] = []
    if use_ollama:
        with col2:
            model_options = st.multiselect(
                "选择要调用的模型（建议 1 个）",
                options=MODELS,
                default=MODELS[:1],
                format_func=model_ui_name,
            )
            selected_models = list(model_options)

    manual_raw = None
    manual_mode = st.checkbox("手动输入模型输出（JSON）", value=False,
                              help="粘贴已准备好的模型 JSON 输出，完全跳过 LLM 调用。")
    if manual_mode:
        manual_raw = st.text_area(
            "模型输出 JSON",
            height=200,
            placeholder='{"qwen2.5:7b": {"case_summary": "...", "structured_analysis": {...}}}',
        )

    if st.button("❙❙ 开始分析", type="primary", use_container_width=True):
        if not case_text.strip():
            st.error("请输入案件描述")
            return
        if manual_mode and manual_raw:
            run_analysis(case_text, use_bert, manual_raw)
        elif use_ollama and selected_models:
            run_analysis(case_text, use_bert, None, selected_models)
        else:
            run_analysis(case_text, use_bert, None)


def render_step2_bert():
    render_step_header(2, "BERT 案件语义画像", "BERT 从案件文本中提取的语义特征")
    bert_data = st.session_state.labor_bert_analysis or {}
    if not bert_data.get("bert_available"):
        st.info("BERT 未启用。勾选「启用 BERT 语义分析」可使用 BERT 增强语义匹配和报告生成。")
        return

    profile = bert_data.get("semantic_case_profile") or {}
    if not profile:
        st.info("BERT 已加载，正在分析...")
        return

    cols = st.columns(5)
    axes = [
        ("劳动关系信号", "employment_relation_score"),
        ("证据充分性", "evidence_completeness"),
        ("雇主违法程度", "employer_conduct_severity"),
        ("法定违规可能", "statutory_violation_score"),
        ("诉求强度", "claim_strength_signal"),
    ]
    for i, (label, key) in enumerate(axes):
        val = profile.get(key, 0.5)
        with cols[i]:
            st.metric(label, f"{val:.2f}")
            st.progress(min(max(val, 0.0), 1.0))


def render_step3_outputs():
    render_step_header(3, "四模型原始输出", "四个本地大模型的结构化分析结果")
    results = st.session_state.labor_results
    times = st.session_state.labor_times
    if results is None:
        st.info("暂无模型输出（分析过程可能出错，请查看错误信息）")
        if st.session_state.labor_error:
            st.error(st.session_state.labor_error)
        return
    if not results:
        st.info("当前使用规则基线模式（未调用 LLM），跳过模型输出展示。勾选「调用本地 Ollama 模型」可启用 LLM 推理。")
        return

    cols = st.columns(4)
    for i, model in enumerate(MODELS):
        raw = results.get(model, "")
        elapsed = (times or {}).get(model, 0)
        with cols[i]:
            st.markdown(f'<div class="model-card-title">{html.escape(model_ui_name(model))}</div>', unsafe_allow_html=True)
            st.caption(f"`{model}` · {elapsed:.1f}s" if elapsed else f"`{model}`")
            if raw.startswith("[ERROR]"):
                st.error(raw[:200])
            elif raw:
                with st.expander("查看原始输出", expanded=False):
                    st.code(raw[:3000], language="json")
            else:
                st.caption("未调用")


def render_step4_comparison():
    render_step_header(4, "结构化分析对比", "7 个维度 × 4 个模型的结构化判断对比")
    normalized = st.session_state.labor_normalized_all
    if not normalized:
        st.info("暂无规范化数据")
        return

    rows = []
    for obj in LABOR_OBJECTS:
        oid = obj["object_id"]
        row = {"维度": obj["label"]}
        for model in MODELS:
            facts = (normalized.get(model, {}) or {}).get("normalized", {}).get(oid, [])
            row[model_ui_name(model)] = "、".join(facts) if facts else "—"
        rows.append(row)

    cols = ["维度"] + [model_ui_name(m) for m in MODELS]
    render_light_table(rows, cols)


def render_step5_truthfinder():
    render_step_header(5, "TruthFinder 可信聚合结果", "模型可信度排名 + 每维度事实排行")
    payload = st.session_state.labor_truthfinder_payload
    if not payload:
        st.info("暂无聚合结果")
        return

    trust_rank = payload.get("model_trust_rank", [])
    if len(trust_rank) <= 1:
        st.warning("⚠️ 当前仅使用 1 个模型。TruthFinder 的真值发现算法需要 2 个以上模型才能发挥多源聚合优势。分歧度和置信度在单模型下无参考意义。建议勾选至少 2 个模型后重新分析。")
    if trust_rank:
        st.markdown("#### 模型可信度排名")
        trust_rows = [
            {"排名": i + 1, "模型": model_ui_name(r["model"]), "信任度": f"{r['trust']:.4f}"}
            for i, r in enumerate(trust_rank)
        ]
        render_light_table(trust_rows, ["排名", "模型", "信任度"])

    st.markdown("#### 各维度事实排行")
    bert_data = st.session_state.labor_bert_analysis or {}
    report = (bert_data.get("comprehensive_report") or {}) if bert_data.get("bert_available") else {}

    for obj in LABOR_OBJECTS:
        oid = obj["object_id"]
        obj_results = payload.get("object_results", [])
        obj_row = next((r for r in obj_results if r.get("object_id") == oid), None)
        if not obj_row:
            continue

        candidates = obj_row.get("candidates", [])[:5]
        obj_reports = report.get("object_reports", []) if report else []
        obj_rpt = next((r for r in obj_reports if r.get("object_id") == oid), {}) if obj_reports else {}
        div_info = (obj_rpt.get("divergence") or {}) if obj_rpt else {}
        conf_info = (obj_rpt.get("confidence") or {}) if obj_rpt else {}

        mode_tag = "单选" if OBJECT_MODES.get(oid) == "single" else "多选"
        div_level = div_info.get("level", "—")
        conf_level = conf_info.get("level", "—")
        div_cls = "is-success" if div_level == "低分歧" else ("is-warning" if div_level == "中分歧" else "is-error")

        with st.expander(
            f"{obj['label']} [{mode_tag}] —— "
            f"分歧度: {div_level} | 置信度: {conf_level}",
            expanded=(oid == "adjudication_tendency"),
        ):
            if not candidates:
                st.caption("无候选事实")
                continue
            fact_rows = []
            for c in candidates:
                selected = "✅" if c.get("is_selected") else ""
                support_by = c.get("support_by_model", {}) or {}
                model_names = [model_ui_name(m) for m, w in support_by.items() if float(w) > 0]
                support_str = ", ".join(model_names) if model_names else "—"
                fact_rows.append({
                    "排名": c.get("rank", "—"),
                    "选中": selected,
                    "事实": c.get("fact", "—"),
                    "置信度": f"{c.get('confidence', 0):.4f}",
                    "支持模型": support_str,
                })
            render_light_table(fact_rows, ["排名", "选中", "事实", "置信度", "支持模型"])


def render_step6_report():
    render_step_header(6, "综合律师分析报告", "自然语言综合报告")
    payload = st.session_state.labor_truthfinder_payload
    bert_data = st.session_state.labor_bert_analysis or {}

    if not payload:
        st.info("暂无 TruthFinder 聚合结果")
        return

    # ── Build natural language report from TruthFinder results ──
    trust_rank = payload.get("model_trust_rank", [])
    object_results = payload.get("object_results", [])
    n_models = len(trust_rank)

    lines = []
    lines.append("## 案件综合分析报告")
    lines.append("")

    # 1. Relationship assessment
    rel_row = next((r for r in object_results if r.get("object_id") == "relationship_type"), None)
    adj_row = next((r for r in object_results if r.get("object_id") == "adjudication_tendency"), None)
    if rel_row:
        rel_fact = (rel_row.get("candidates", []) or [{}])[0].get("fact", "未确定")
        lines.append(f"**法律关系认定**：{rel_fact}")
    if adj_row:
        adj_fact = (adj_row.get("candidates", []) or [{}])[0].get("fact", "未确定")
        lines.append(f"**裁判倾向预判**：{adj_fact}")

    # 2. Dispute focus
    disp_row = next((r for r in object_results if r.get("object_id") == "dispute_focus"), None)
    if disp_row:
        selected = [c.get("fact") for c in (disp_row.get("candidates", []) or []) if c.get("is_selected")]
        if selected:
            lines.append(f"**核心争议类型**：{'、'.join(selected)}")

    # 3. Key facts
    fact_row = next((r for r in object_results if r.get("object_id") == "key_fact"), None)
    if fact_row:
        selected = [c.get("fact") for c in (fact_row.get("candidates", []) or []) if c.get("is_selected")]
        real_facts = [f for f in selected if "证据不足" not in f]
        if real_facts:
            lines.append(f"**已识别关键事实**：{'、'.join(real_facts)}")

    # 4. Legal articles
    art_row = next((r for r in object_results if r.get("object_id") == "article_reference"), None)
    if art_row:
        selected = [c.get("fact") for c in (art_row.get("candidates", []) or []) if c.get("is_selected")]
        if selected:
            lines.append(f"**重点法条方向**：{'、'.join(selected[:8])}")

    # 5. Background
    bg_row = next((r for r in object_results if r.get("object_id") == "background"), None)
    if bg_row:
        bg_facts = [c.get("fact") for c in (bg_row.get("candidates", []) or []) if c.get("is_selected")]
        non_default = [b for b in bg_facts if "无特殊" not in b]
        if non_default:
            lines.append(f"**特殊背景**：{'、'.join(non_default)}")

    lines.append("")

    # 6. Model trust
    if n_models > 1:
        lines.append(f"**模型可信度排行**（共 {n_models} 个模型）:")
        for r in trust_rank:
            lines.append(f"- {model_ui_name(r['model'])}：信任度 {r['trust']:.4f}")
    elif n_models == 1:
        lines.append(f"**注意**：当前只使用了 1 个模型（{model_ui_name(trust_rank[0]['model'])}），无法进行多模型对比。建议至少使用 2 个模型才能发挥 TruthFinder 的多源聚合优势。分歧度和置信度在单模型下无参考意义。")

    lines.append("")

    # 7. Evidence gaps + actions
    gaps = []
    for row in object_results:
        for c in (row.get("candidates", []) or []):
            if c.get("is_selected") and "证据不足" in str(c.get("fact", "")):
                gaps.append(f"{row.get('object_label', '')}: {c.get('fact', '')}")

    if gaps:
        lines.append("**证据缺口及补充建议**：")
        for g in gaps:
            lines.append(f"- ⚠️ {g}")
        lines.append("- 建议补充：书面劳动合同、工资银行流水、考勤记录、解除通知原件、社保缴纳记录")
        lines.append("- 如需仲裁或诉讼，请注意劳动争议调解仲裁法第 27 条关于仲裁时效的规定（一般为一年）")

    lines.append("")
    lines.append("---")
    lines.append("**免责声明**：本报告由多模型 TruthFinder 可信聚合系统自动生成，不构成正式法律意见。法律判断请以执业律师结合全部案件材料后的专业意见为准。")

    report_text = "\n".join(lines)
    st.markdown(report_text)

    # If BERT report is available, show it as additional detail
    report = bert_data.get("comprehensive_report") if bert_data.get("bert_available") else None
    if report:
        with st.expander("BERT 增强分析（技术细节）", expanded=False):
            st.json(report.get("object_reports", [])[:3])


def render_step7_zk():
    render_step_header(7, "ZK 证明预览（技术预览）", "Groth16 零知识证明电路状态")
    with st.expander("ZK 电路信息", expanded=False):
        st.markdown("""
        | 项目 | 状态 |
        |------|------|
        | Circom 电路 | 已就绪 (`truthfinder.circom`) |
        | 证明对象 | TruthFinder 15 轮 Q16 定点迭代 |
        | 输入格式 | M=4 models × K≤10 objects × N≤8 facts |
        | BERT 影响 | 无——BERT 处理在电路边界外 |
        | 确定性 | ✅ (BERT `model.eval()` + `no_grad()`) |
        """)
        st.caption("注：ZK 证明层当前仅支持翻译场景。劳动法场景的 ZK 适配为未来工作计划。")


def main():
    init_session()
    inject_css()
    render_header()

    render_step1_input()

    if st.session_state.labor_error:
        st.error(st.session_state.labor_error)

    if st.session_state.labor_results:
        st.divider()
        render_step2_bert()

        st.divider()
        render_step3_outputs()

        st.divider()
        render_step4_comparison()

        st.divider()
        render_step5_truthfinder()

        st.divider()
        render_step6_report()

        st.divider()
        render_step7_zk()

    st.divider()
    st.markdown("""
    <div class="disclaimer-card">
        <h4>⚖️ 法律免责声明</h4>
        <p>本系统为律师办案辅助工具，通过多模型 TruthFinder 可信聚合算法提供参考分析。</p>
        <p>不构成正式法律意见、裁判预测或对案件结果的保证。法律判断请以执业律师结合全部案件材料后的专业意见为准。引用法条请以官方公布的最新有效文本为准。</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
