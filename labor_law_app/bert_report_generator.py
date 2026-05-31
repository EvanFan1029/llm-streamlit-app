from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

from labor_law_app.labor_truthfinder import (
    LABOR_OBJECT_IDS,
    LABOR_OBJECT_LABELS,
    LABOR_OBJECT_MODES,
    LABOR_OBJECT_OPTIONS,
)


@dataclass
class DivergenceMetrics:
    object_id: str
    object_label: str
    fact_entropy: float = 0.0
    normalized_entropy: float = 0.0
    inter_model_agreement_rate: float = 0.0
    divergence_level: str = "低分歧"
    interpretation: str = ""


@dataclass
class ConfidenceMetrics:
    object_id: str
    object_label: str
    trust_weighted_confidence: float = 0.0
    support_density: float = 0.0
    confidence_stability: float = 1.0
    overall_confidence: float = 0.0
    confidence_level: str = "中等置信度"
    caveats: List[str] = field(default_factory=list)


@dataclass
class ObjectReport:
    object_id: str
    object_label: str
    mode: str
    ranked_facts: List[Dict[str, Any]] = field(default_factory=list)
    divergence: Optional[DivergenceMetrics] = None
    confidence: Optional[ConfidenceMetrics] = None


@dataclass
class ComprehensiveReport:
    case_id: str
    case_summary: str = ""
    core_legal_issues: List[str] = field(default_factory=list)
    relationship_assessment: str = ""
    overall_tendency: str = ""
    object_reports: List[ObjectReport] = field(default_factory=list)
    article_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    evidence_gaps: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    model_trust_ranking: List[Dict[str, Any]] = field(default_factory=list)
    overall_divergence: str = ""
    overall_confidence: str = ""
    disclaimer: str = (
        "本报告由多模型 TruthFinder 可信聚合系统自动生成，"
        "不构成正式法律意见。法律判断请以执业律师结合全部案件材料后的专业意见为准。"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_summary": self.case_summary,
            "core_legal_issues": self.core_legal_issues,
            "relationship_assessment": self.relationship_assessment,
            "overall_tendency": self.overall_tendency,
            "object_reports": [
                {
                    "object_id": r.object_id,
                    "object_label": r.object_label,
                    "mode": r.mode,
                    "ranked_facts": r.ranked_facts,
                    "divergence": {
                        "level": r.divergence.divergence_level if r.divergence else "N/A",
                        "entropy": round(r.divergence.fact_entropy, 4) if r.divergence else 0,
                        "normalized_entropy": round(r.divergence.normalized_entropy, 4) if r.divergence else 0,
                        "agreement_rate": round(r.divergence.inter_model_agreement_rate, 4) if r.divergence else 0,
                    } if r.divergence else None,
                    "confidence": {
                        "level": r.confidence.confidence_level if r.confidence else "N/A",
                        "overall": round(r.confidence.overall_confidence, 4) if r.confidence else 0,
                        "trust_weighted": round(r.confidence.trust_weighted_confidence, 4) if r.confidence else 0,
                        "support_density": round(r.confidence.support_density, 4) if r.confidence else 0,
                    } if r.confidence else None,
                }
                for r in self.object_reports
            ],
            "article_recommendations": self.article_recommendations,
            "evidence_gaps": self.evidence_gaps,
            "recommended_actions": self.recommended_actions,
            "model_trust_ranking": self.model_trust_ranking,
            "overall_divergence": self.overall_divergence,
            "overall_confidence": self.overall_confidence,
            "disclaimer": self.disclaimer,
        }


class BERTReportGenerator:
    def compute_divergence(
        self,
        truth_rows: List[Dict[str, Any]],
        t_score: Dict[str, float],
        model_facts: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ) -> Dict[str, DivergenceMetrics]:
        models = list(t_score.keys())
        n_models = max(len(models), 1)
        metrics: Dict[str, DivergenceMetrics] = {}

        for row in (truth_rows or []):
            oid = row.get("object_id", "")
            label = row.get("object_label", oid)
            candidates = row.get("candidates", []) or []
            selected = [c for c in candidates if c.get("is_selected")]
            num_options = max(len(LABOR_OBJECT_OPTIONS.get(oid, [])), 2)

            if not selected:
                m = DivergenceMetrics(object_id=oid, object_label=label)
                m.divergence_level = "低分歧" if n_models <= 1 else "中分歧"
                metrics[oid] = m
                continue

            # Count unique top-1 facts across models (direct disagreement measure)
            model_top1: Dict[str, Optional[str]] = {}
            for c in candidates:
                support = c.get("support_by_model", {}) or {}
                for m in models:
                    w = float(support.get(m, 0))
                    if w > 0:
                        current = model_top1.get(m)
                        if current is None or w > float(next((cc.get("support_by_model", {}).get(m, 0) for cc in candidates if cc.get("fact") == current), 0)):
                            model_top1[m] = c.get("fact")

            unique_top1_facts = set(f for f in model_top1.values() if f is not None)
            n_unique = len(unique_top1_facts)

            # Simple divergence: how many different top-1 answers?
            if n_models <= 1:
                agreement = 1.0
                level = "低分歧"
            elif n_unique == 1:
                agreement = 1.0
                level = "低分歧"
            elif n_unique == n_models:
                agreement = 0.0
                level = "高分歧"
            else:
                agreement = 1.0 - (n_unique - 1) / (n_models - 1) if n_models > 1 else 1.0
                level = "中分歧" if agreement > 0.3 else "高分歧"

            # Entropy for display only
            confs = [float(c.get("confidence", 0)) for c in selected]
            entropy = self._compute_entropy(confs)
            norm_entropy = entropy / math.log(num_options) if num_options > 1 else 0.0

            interpretations = {
                "低分歧": f"全部 {n_models} 个模型对该维度的判断一致",
                "中分歧": f"{n_unique}/{n_models} 个模型给出不同判断，存在部分分歧",
                "高分歧": f"全部 {n_models} 个模型给出不同判断，建议人工复核",
            }

            m = DivergenceMetrics(
                object_id=oid,
                object_label=label,
                fact_entropy=entropy,
                normalized_entropy=norm_entropy,
                inter_model_agreement_rate=agreement,
                divergence_level=level,
                interpretation=interpretations.get(level, ""),
            )
            metrics[oid] = m

        return metrics

    def compute_confidence(
        self,
        truth_rows: List[Dict[str, Any]],
        t_score: Dict[str, float],
        effective_trust: Optional[Dict[str, float]] = None,
        model_coverage: Optional[Dict[str, float]] = None,
        change_history: Optional[List[float]] = None,
    ) -> Dict[str, ConfidenceMetrics]:
        if effective_trust is None:
            effective_trust = t_score
        n_models = max(len(t_score), 1)
        metrics: Dict[str, ConfidenceMetrics] = {}

        last_change = 0.0
        max_change = 1.0
        if change_history and len(change_history) > 0:
            last_change = change_history[-1]
            max_change = max(change_history) if max(change_history) > 0 else 1.0

        for row in (truth_rows or []):
            oid = row.get("object_id", "")
            label = row.get("object_label", oid)
            selected = [c for c in (row.get("candidates", []) or []) if c.get("is_selected")]

            if not selected:
                continue

            sel_confs = [float(c.get("confidence", 0)) for c in selected]
            mean_sel = sum(sel_confs) / len(sel_confs) if sel_confs else 0.0

            eff_vals = [float(effective_trust.get(m, t_score.get(m, 0.0))) for m in t_score]
            mean_eff = sum(eff_vals) / len(eff_vals) if eff_vals else 0.5

            trust_weighted = mean_sel * mean_eff / 0.75

            supporting = 0
            for c in selected:
                support = c.get("support_by_model", {}) or {}
                supporting += sum(1 for m in t_score if float(support.get(m, 0)) > 0)
            support_density = min(supporting / (n_models * max(len(selected), 1)), 1.0)

            coverage = 1.0
            if model_coverage:
                cv_vals = [float(model_coverage.get(m, 1.0)) for m in t_score]
                coverage = min(cv_vals) if cv_vals else 1.0

            stability = 1.0 - min(last_change / max_change, 1.0) if max_change > 0 else 1.0
            overall = (
                trust_weighted * 0.4
                + support_density * 0.3
                + coverage * 0.15
                + stability * 0.15
            )
            level = self._classify_confidence(overall)

            caveats: List[str] = []
            if support_density < 0.5:
                caveats.append("支持该结论的模型较少")
            if stability < 0.5:
                caveats.append("聚合结果收敛不稳定，建议增加模型或检查数据")
            if trust_weighted < 0.4:
                caveats.append("模型整体信任度偏低")

            metrics[oid] = ConfidenceMetrics(
                object_id=oid,
                object_label=label,
                trust_weighted_confidence=trust_weighted,
                support_density=support_density,
                confidence_stability=stability,
                overall_confidence=overall,
                confidence_level=level,
                caveats=caveats,
            )

        return metrics

    def generate_comprehensive_report(
        self,
        case_id: str,
        case_text: str = "",
        truth_rows: Optional[List[Dict[str, Any]]] = None,
        t_score: Optional[Dict[str, float]] = None,
        effective_trust: Optional[Dict[str, float]] = None,
        model_coverage: Optional[Dict[str, float]] = None,
        change_history: Optional[List[float]] = None,
        article_candidates: Optional[List[Dict[str, Any]]] = None,
        evidence_gaps: Optional[List[str]] = None,
    ) -> ComprehensiveReport:
        truth_rows = truth_rows or []
        t_score = t_score or {}
        report = ComprehensiveReport(case_id=case_id)

        divergences = self.compute_divergence(truth_rows, t_score)
        confidences = self.compute_confidence(
            truth_rows, t_score, effective_trust, model_coverage, change_history
        )

        for row in truth_rows:
            oid = row.get("object_id", "")
            obj_report = ObjectReport(
                object_id=oid,
                object_label=row.get("object_label", oid),
                mode=LABOR_OBJECT_MODES.get(oid, "multi"),
                ranked_facts=[
                    {
                        "rank": c.get("rank"),
                        "fact": c.get("fact"),
                        "confidence": round(float(c.get("confidence", 0)), 4),
                        "is_selected": c.get("is_selected", False),
                        "support_by_model": c.get("support_by_model", {}),
                        "support_weight": round(float(c.get("support_weight", 0)), 4),
                    }
                    for c in (row.get("candidates", []) or [])
                ],
                divergence=divergences.get(oid),
                confidence=confidences.get(oid),
            )
            if oid == "relationship_type" and obj_report.ranked_facts:
                report.relationship_assessment = obj_report.ranked_facts[0].get("fact", "")
            if oid == "adjudication_tendency" and obj_report.ranked_facts:
                report.overall_tendency = obj_report.ranked_facts[0].get("fact", "")

            report.object_reports.append(obj_report)

        report.core_legal_issues = self._extract_core_issues(truth_rows)
        report.article_recommendations = article_candidates or []
        report.evidence_gaps = evidence_gaps or []
        report.recommended_actions = self._build_recommended_actions(
            report.evidence_gaps, report.core_legal_issues
        )
        report.model_trust_ranking = [
            {"model": m, "trust_score": round(float(s), 4)}
            for m, s in sorted(t_score.items(), key=lambda x: x[1], reverse=True)
        ]

        div_levels = [d.divergence_level for d in divergences.values()]
        high = sum(1 for d in div_levels if d == "高分歧")
        mid = sum(1 for d in div_levels if d == "中分歧")
        if high >= len(div_levels) * 0.3:
            report.overall_divergence = "显著分歧"
        elif high + mid > 0:
            report.overall_divergence = "部分分歧"
        else:
            report.overall_divergence = "高度一致"

        conf_levels = [c.confidence_level for c in confidences.values()]
        if any(cl == "低置信度" for cl in conf_levels):
            report.overall_confidence = "低置信度"
        elif any(cl == "中等置信度" for cl in conf_levels):
            report.overall_confidence = "中等置信度"
        else:
            report.overall_confidence = "高置信度"

        return report

    @staticmethod
    def _compute_entropy(confidences: List[float]) -> float:
        cleaned = [max(c, 1e-8) for c in confidences]
        total = sum(cleaned)
        if total <= 0:
            return 0.0
        probs = [c / total for c in cleaned]
        return -sum(p * math.log(p) for p in probs)

    @staticmethod
    def _classify_confidence(overall: float) -> str:
        if overall >= 0.55:
            return "高置信度"
        if overall >= 0.35:
            return "中等置信度"
        return "低置信度"

    @staticmethod
    def _extract_core_issues(truth_rows: List[Dict[str, Any]]) -> List[str]:
        issues: List[str] = []
        for row in truth_rows:
            if row.get("object_id") == "dispute_focus":
                for c in (row.get("candidates", []) or []):
                    if c.get("is_selected"):
                        issues.append(f"争议类型: {c.get('fact', '')}")
            if row.get("object_id") == "key_fact":
                for c in (row.get("candidates", []) or []):
                    if c.get("is_selected") and "证据不足" not in str(c.get("fact", "")):
                        issues.append(f"关键事实: {c.get('fact', '')}")
        return issues[:10]

    @staticmethod
    def _build_recommended_actions(
        evidence_gaps: List[str], core_issues: List[str]
    ) -> List[str]:
        actions: List[str] = []
        if evidence_gaps:
            for gap in evidence_gaps[:5]:
                actions.append(f"补充证据: {gap}")
        if any("仲裁时效" in issue for issue in core_issues):
            actions.append("核查仲裁时效是否届满（劳动争议调解仲裁法第27条）")
        if any("双倍工资" in issue for issue in core_issues):
            actions.append("核实入职时间与书面劳动合同签订时间，计算双倍工资差额")
        if any("违法解除" in issue for issue in core_issues):
            actions.append("审查用人单位解除程序是否合法，收集解除通知及相关证据")
        if any("未签" in issue or "书面劳动合同" in issue for issue in core_issues):
            actions.append("重点收集用工事实证明材料（工资记录、考勤、工作证、社保记录等）")
        if not actions:
            actions.append("建议收集与本案相关的全部书面证据")
        return actions
