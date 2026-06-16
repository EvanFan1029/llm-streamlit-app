from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from labor_law_app.normalize_labor import BACKGROUND_OPTIONS, LABOR_OBJECTS, get_labor_objects


_LABOR_OBJECTS_META = get_labor_objects()

# ── v1 (legacy) prompt kept for backward compatibility ──

UNIFIED_SYSTEM_PROMPT = """你是一位精通中国劳动法和相关司法实践的律师助理。

请基于用户提供的案件描述，输出以下模块化分析结果。输出必须是合法的 JSON 格式。

## 输出格式要求

{
  "case_summary": "用100-200字总结案件核心事实",
  "structured_analysis": {
"""


def build_unified_prompt_template() -> str:
    lines = [UNIFIED_SYSTEM_PROMPT]
    for obj in _LABOR_OBJECTS_META:
        oid = obj["object_id"]
        label = obj["label"]
        mode = obj["mode"]
        opts = obj["options"]
        mode_hint = "（单选，只选最匹配的一个）" if mode == "single" else "（多选，可同时选择多个匹配项）"
        lines.append(f'    "{oid}": ["选项1", "选项2", ...],  // {label} {mode_hint}')
        lines.append(f"    // 可选值: {', '.join(opts[:8])}{'...' if len(opts) > 8 else ''}")
    lines.append("  }")
    lines.append("}")

    lines.append("""
## 分析要求
1. 严格基于案件描述中的事实进行判断，不要推测未提及的信息
2. 对于不确定的判断，选择最接近的选项
3. 多选字段应当包含所有相关选项，不要遗漏
4. 如果没有足够信息支持某个字段的判断，使用 fallback 默认值
5. case_summary 控制在80字以内""")

    return "\n".join(lines)


UNIFIED_PROMPT = build_unified_prompt_template()


def build_labor_prompt(case_text: str, profile_hint: str = "") -> str:
    parts: List[str] = []
    parts.append(UNIFIED_PROMPT)
    parts.append("")
    parts.append("## 案件描述")
    parts.append(case_text)
    if profile_hint:
        parts.append("")
        parts.append("## 案件背景提示")
        parts.append(profile_hint)
    parts.append("")
    parts.append("请按照上述格式输出 JSON 分析结果。")
    return "\n".join(parts)


# ────────────────────────────────────────────────────────────
#  v2 — 对齐 medical_app 的严格结构化 prompt 模板
# ────────────────────────────────────────────────────────────

LABOR_LAW_SYSTEM_PROMPT = """你是一位谨慎、可靠的劳动法律师助理。你不能出具正式法律意见书，不能代替执业律师，不能对案件结果做出保证。

请只输出合法 JSON，不要输出 Markdown，不要输出代码块，不要输出解释性前缀。"""


def build_labor_object_blocks() -> str:
    """Build per-object option blocks for the prompt, similar to build_medical_prompt."""
    blocks = []
    for schema in LABOR_OBJECTS:
        options = "\n".join(f"  - {opt}" for opt in schema.options)
        plural_hint = "可以多个" if schema.mode == "multi" else "只能选一个"
        fallback = schema.fallback[0] if schema.fallback else "无"
        blocks.append(
            f"{schema.object_id}（{schema.label}，{plural_hint}，默认值: {fallback}）:\n{options}"
        )
    return "\n\n".join(blocks)


def build_labor_prompt_v2(case_text: str, profile_hint: str = "") -> str:
    """Build a strict structured prompt for labor law LLM analysis.

    Guides each LLM to output:
      1. user_explanation — natural-language advice for the consulter
      2. structured_analysis — 7-object structured judgment using canonical options
    """
    object_blocks = build_labor_object_blocks()

    prompt = f"""你是一个谨慎、可靠的劳动法律师助理。你不能出具正式法律意见书，不能代替执业律师，不能对案件结果做出保证。

请只输出合法 JSON，不要输出 Markdown，不要输出代码块，不要输出解释性前缀。

请根据用户提供的劳动争议案件描述，输出：
1. user_explanation：给咨询者看的自然语言法律初步分析；
2. structured_analysis：七个维度下的结构化判断。

严格要求：
1. 不能出具正式法律意见书。
2. 不能说"一定赢""肯定胜诉"。
3. 条件性提醒不能覆盖当前事实的紧急程度。
4. 如果证据不足，应明确指出缺失哪些材料。
5. structured_analysis 不要输出长句，尽量直接从候选集合中选。
6. 严格基于用户原始案情文本，不得补写未出现的事实。
7. 不得根据性别推断孕期、产期、哺乳期。仅出现"女性、女员工、女士、她"只能确认性别，不得推断三期女职工。
8. 不得根据"有孩子、已婚、母亲、宝妈、接送孩子"推断哺乳期。
9. 特殊身份背景必须有原文明确触发词（如怀孕、孕期、产假、哺乳期、工伤、高管、劳务派遣等），若无则 background 必须是"无特殊背景信息"。
10. 证据不足时输出"证据不足/需补充材料"，不得自行脑补证据。
6. user_explanation 可以自然语言表达，但必须谨慎、清楚，不过度承诺。
7. single 维度只能输出一个候选值（字符串）。
8. multi 维度只能输出候选集合中的若干值（数组）。
9. 必须覆盖全部七个维度，不能遗漏。
10. 注意劳动争议调解仲裁法第 27 条关于仲裁时效的规定（一般为一年）。

七个维度与候选集合如下：

{object_blocks}

输出 JSON 格式必须严格如下：
{{{{
  "user_explanation": "给咨询者看的自然语言法律初步分析",
  "structured_analysis": {{{{
    "relationship_type": "只能从候选集合中选择一个",
    "dispute_focus": ["只能从候选集合中选择，可以多个"],
    "key_fact": ["只能从候选集合中选择，可以多个"],
    "issue_keyword": ["只能从候选集合中选择，可以多个"],
    "article_reference": ["只能从候选集合中选择，可以多个"],
    "adjudication_tendency": "只能从候选集合中选择一个",
    "background": ["只能从候选集合中选择，可以多个"]
  }}}}
}}}}

{ '## 案件背景提示' + chr(10) + profile_hint if profile_hint else '' }

用户案件描述：
\"\"\"{case_text.strip()}\"\"\""""

    return prompt.strip()


def build_profile_hint(profile: "CaseSemanticProfile") -> str:
    hints: List[str] = []
    es = profile.employment_relation_score
    if es > 0.6:
        hints.append("本案劳动关系特征较为明显，重点审查是否存在未签书面劳动合同、违法解除等实体性争议。")
    elif es < 0.4:
        hints.append("本案劳动关系的认定存在不确定性，需首先关注劳务关系与劳动关系的区分，审查用工管理、报酬支付等核心要素。")

    ev = profile.evidence_completeness
    if ev > 0.5:
        hints.append("案件描述中包含较多事实信息，可基于现有信息进行初步法律分析。")
    else:
        hints.append("案件描述较为简略，部分事实信息有待补充，部分判断需标注不确定性。")

    sv = profile.statutory_violation_score
    if sv > 0.6:
        hints.append("案件中存在较为明确的法律违规信号，可重点关注相关法条的适用条件。")

    cs = profile.claim_strength_signal
    if cs > 0.6:
        hints.append("劳动者诉求具有一定的法律依据支持。")
    elif cs < 0.35:
        hints.append("劳动者诉求的法律依据尚需进一步核实。")

    return " ".join(hints) if hints else ""
