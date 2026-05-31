from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from labor_law_app.normalize_labor import BACKGROUND_OPTIONS, get_labor_objects


_LABOR_OBJECTS_META = get_labor_objects()

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
