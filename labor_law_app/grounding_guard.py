from __future__ import annotations

import re
from typing import Any, List, Tuple


# ── Sensitive fact grounding rules ──
# Key concept: LLM/BERT can propose candidate facts, but certain sensitive facts
# MUST have explicit trigger words in the original case text. Without them,
# the fact is removed from normalized output before entering TruthFinder/ZK.

# Each rule: (object_id, fact_pattern, required_triggers, reason, explicitly_forbidden_indicators)
# - required_triggers: at least one must appear in case_text to keep the fact
# - explicitly_forbidden_indicators: if ONLY these appear (without any trigger),
#   the fact is removed even more aggressively

GROUNDING_RULES: list[dict[str, Any]] = [
    {
        "object_id": "background",
        "fact": "劳动者为孕期/产期/哺乳期女职工",
        "required_triggers": [
            "怀孕", "孕期", "产期", "产假", "哺乳期",
            "孕妇", "产妇", "三期", "产后",
            "生孩子", "生育", "待产", "临产", "预产期",
        ],
        "forbidden_indicators": [
            "女性", "女员工", "女士", "她", "女职工", "女劳动者",
            "已婚", "孩子", "母亲", "宝妈", "接送孩子",
            "小孩", "子女", "带孩子", "家中有孩子",
        ],
        "reason": "原文未出现怀孕、孕期、产假、哺乳期等明确依据，仅凭女性身份或母亲身份不得推断三期女职工",
    },
    {
        "object_id": "background",
        "fact": "劳动者为工伤/职业病职工",
        "required_triggers": [
            "工伤", "职业病", "工伤认定", "劳动能力鉴定",
            "伤残等级", "因工受伤", "工作中受伤", "工伤事故",
            "工伤保险", "工伤待遇",
        ],
        "forbidden_indicators": [],
        "reason": "原文未出现工伤、职业病、工伤认定、因工受伤等明确依据",
    },
    {
        "object_id": "background",
        "fact": "劳动者为劳务派遣用工",
        "required_triggers": [
            "劳务派遣", "派遣员工", "派遣用工", "用工单位",
            "派遣单位", "第三方派遣", "被派遣",
        ],
        "forbidden_indicators": [],
        "reason": "原文未出现劳务派遣、派遣员工、派遣用工等明确依据",
    },
    {
        "object_id": "background",
        "fact": "劳动者为高级管理人员",
        "required_triggers": [
            "高级管理人员", "高管", "总经理", "副总经理",
            "总监", "董事", "监事", "高级管理岗",
            "CTO", "CEO", "CFO", "COO",
        ],
        "forbidden_indicators": [
            "管理", "负责", "主管", "组长", "经理",
        ],
        "reason": "原文未出现高级管理人员、高管、总经理、总监、董事等明确依据，普通管理岗位不等同于高级管理人员",
    },
]


def _has_trigger(text: str, triggers: List[str]) -> bool:
    t = (text or "").lower()
    return any(tr.lower() in t for tr in triggers)


def _has_only_forbidden(text: str, triggers: List[str], forbidden: List[str]) -> bool:
    t = (text or "").lower()
    has_trigger = any(tr.lower() in t for tr in triggers)
    if has_trigger:
        return False
    has_forbidden = any(fb.lower() in t for fb in forbidden)
    return has_forbidden


def filter_unsupported_facts(
    normalized: dict[str, list[str]],
    case_text: str,
    *,
    source_name: str | None = None,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    filtered: dict[str, list[str]] = {}
    warnings: list[dict[str, Any]] = []

    for object_id, facts in (normalized or {}).items():
        kept: list[str] = []
        for fact in (facts or []):
            removed = False
            for rule in GROUNDING_RULES:
                if rule["object_id"] != object_id:
                    continue
                if rule["fact"] != fact:
                    continue
                has_trigger = _has_trigger(case_text, rule["required_triggers"])
                if has_trigger:
                    continue

                # 修改后逻辑：无触发词一律删除
                # 无论原文是否存在禁用语，只要没有明确的触发依据，敏感事实都被拦截
                warnings.append({
                    "source": source_name or "unknown",
                    "object_id": object_id,
                    "fact": fact,
                    "reason": rule["reason"],
                    "action": "removed",
                })
                removed = True
                break

            if not removed:
                kept.append(fact)

        if not kept and object_id == "background":
            kept = ["无特殊背景信息"]
        filtered[object_id] = kept

    return filtered, warnings
