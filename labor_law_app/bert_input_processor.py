from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from labor_law_app.bert_processor import BERTProcessor
from labor_law_app.bert_prompts import build_labor_prompt, build_profile_hint


@dataclass
class AnchorAxis:
    name: str
    positive_anchors: Tuple[str, ...]
    negative_anchors: Tuple[str, ...]


SEMANTIC_AXES: List[AnchorAxis] = [
    AnchorAxis(
        name="employment_relation_score",
        positive_anchors=(
            "劳动者接受用人单位管理",
            "按月领取固定工资",
            "公司统一考勤打卡",
            "用人单位缴纳社会保险",
            "劳动者从事用人单位安排的有报酬劳动",
        ),
        negative_anchors=(
            "独立承揽人以自己设备完成工作并交付成果",
            "按项目进度结算报酬",
            "个人间劳务合同关系",
            "临时性用工无固定管理",
        ),
    ),
    AnchorAxis(
        name="evidence_completeness",
        positive_anchors=(
            "已签订书面劳动合同",
            "有工资银行流水记录",
            "有考勤打卡记录",
            "有书面解除通知",
            "有劳动合同原件或照片",
        ),
        negative_anchors=(
            "无法提供劳动合同",
            "口头约定无书面记录",
            "现金发放无记录",
            "无法提供考勤记录",
            "没有明确的解除通知",
        ),
    ),
    AnchorAxis(
        name="employer_conduct_severity",
        positive_anchors=(
            "违法解除劳动合同",
            "拖欠数月工资",
            "未依法缴纳社会保险",
            "单方面变更劳动条件",
            "威胁或强迫劳动者辞职",
        ),
        negative_anchors=(
            "双方协商解除劳动合同",
            "工资核算有微小差异",
            "程序性瑕疵但实质合规",
            "正常到期不续签",
            "按合同约定处理",
        ),
    ),
    AnchorAxis(
        name="statutory_violation_score",
        positive_anchors=(
            "未签书面劳动合同超过一个月",
            "未支付加班费",
            "无正当理由单方解除",
            "未支付经济补偿金",
            "低于最低工资标准",
        ),
        negative_anchors=(
            "已签书面劳动合同",
            "依法支付了经济补偿",
            "协商一致解除",
            "用人单位有正当理由解除",
            "程序合法合规",
        ),
    ),
    AnchorAxis(
        name="claim_strength_signal",
        positive_anchors=(
            "有书面证据支持劳动者主张",
            "用人单位有明显违法行为",
            "同类案件裁判倾向支持劳动者",
            "劳动者主张有明确法条依据",
            "多份证据相互印证",
        ),
        negative_anchors=(
            "证据不足难以支持主张",
            "劳动者自身存在过失",
            "超过法定时效",
            "主张金额计算缺乏依据",
            "无直接证据仅凭口述",
        ),
    ),
]


@dataclass
class CaseSemanticProfile:
    employment_relation_score: float = 0.5
    evidence_completeness: float = 0.5
    employer_conduct_severity: float = 0.5
    statutory_violation_score: float = 0.5
    claim_strength_signal: float = 0.5
    dominant_dispute_category: str = ""
    top_articles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "employment_relation_score": round(self.employment_relation_score, 3),
            "evidence_completeness": round(self.evidence_completeness, 3),
            "employer_conduct_severity": round(self.employer_conduct_severity, 3),
            "statutory_violation_score": round(self.statutory_violation_score, 3),
            "claim_strength_signal": round(self.claim_strength_signal, 3),
            "dominant_dispute_category": self.dominant_dispute_category,
            "top_articles": self.top_articles,
        }


class BERTInputProcessor:
    def __init__(self, bert_processor: BERTProcessor):
        self.bert = bert_processor
        self._precomputed: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._precompute_axes()

    def _precompute_axes(self) -> None:
        for axis in SEMANTIC_AXES:
            pos_embs = self.bert.embed_texts(list(axis.positive_anchors))
            neg_embs = self.bert.embed_texts(list(axis.negative_anchors))
            pos_mean = pos_embs.mean(axis=0)
            neg_mean = neg_embs.mean(axis=0)
            self._precomputed[axis.name] = (pos_mean, neg_mean)

    def analyze_case(self, case_text: str) -> CaseSemanticProfile:
        if not case_text or not case_text.strip():
            return CaseSemanticProfile()

        case_emb = self.bert.embed_single(case_text.strip())
        profile = CaseSemanticProfile()

        for axis in SEMANTIC_AXES:
            pos_mean, neg_mean = self._precomputed[axis.name]
            pos_sim = self.bert.cosine_similarity(case_emb, pos_mean)
            neg_sim = self.bert.cosine_similarity(case_emb, neg_mean)
            # Use difference instead of ratio — produces actually discriminating scores
            diff = pos_sim - neg_sim
            # Rescale from [-0.5, 0.5] to [0, 1]
            score = max(0.0, min(1.0, (diff + 0.3) / 0.6))
            setattr(profile, axis.name, float(score))

        return profile

    def build_unified_prompt(self, case_text: str) -> str:
        profile = self.analyze_case(case_text)
        hint = build_profile_hint(profile)
        return build_labor_prompt(case_text, hint)
