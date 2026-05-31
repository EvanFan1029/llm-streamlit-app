from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from labor_law_app.bert_processor import BERTProcessor
from labor_law_app.normalize_labor import LABOR_OBJECTS, OBJECT_OPTIONS


@dataclass
class SemanticMatch:
    option: str
    similarity: float
    confidence: float
    is_above_threshold: bool


@dataclass
class ObjectMatchResult:
    object_id: str
    object_label: str
    mode: str
    model_name: str
    raw_value: Any
    matches: List[SemanticMatch] = field(default_factory=list)
    best_match: Optional[SemanticMatch] = None
    selected_matches: List[SemanticMatch] = field(default_factory=list)


OBJECT_LABELS = {s.object_id: s.label for s in LABOR_OBJECTS}
OBJECT_MODES = {s.object_id: s.mode for s in LABOR_OBJECTS}
OBJECT_OPTION_TUPLES = {s.object_id: s.options for s in LABOR_OBJECTS}

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "relationship_type": 0.55,
    "dispute_focus": 0.45,
    "key_fact": 0.45,
    "issue_keyword": 0.40,
    "article_reference": 0.50,
    "adjudication_tendency": 0.50,
    "background": 0.40,
}

DEFAULT_SINGLE_MARGIN = 0.08
DEFAULT_MULTI_THRESHOLD = 0.45


class BERTOutputProcessor:
    def __init__(
        self,
        bert_processor: BERTProcessor,
        thresholds: Optional[Dict[str, float]] = None,
        single_margin: float = DEFAULT_SINGLE_MARGIN,
        multi_threshold: float = DEFAULT_MULTI_THRESHOLD,
        rule_weight: float = 0.6,
        bert_weight: float = 0.4,
    ):
        self.bert = bert_processor
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.single_margin = single_margin
        self.multi_threshold = multi_threshold
        self.rule_weight = rule_weight
        self.bert_weight = bert_weight

        self._option_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self._precompute_all_options()

    def _precompute_all_options(self) -> None:
        for schema in LABOR_OBJECTS:
            opts = tuple(schema.options)
            if opts:
                self._option_embeddings[schema.object_id] = self.bert.embed_options(
                    opts, group=f"labor_{schema.object_id}"
                )

    def match_model_field(
        self,
        object_id: str,
        raw_value: Any,
        model_name: str = "",
    ) -> ObjectMatchResult:
        if object_id not in OBJECT_MODES:
            raise ValueError(f"Unknown object_id: {object_id}")

        mode = OBJECT_MODES[object_id]
        label = OBJECT_LABELS[object_id]
        options = OBJECT_OPTION_TUPLES[object_id]
        threshold = self.thresholds.get(object_id, 0.45)

        result = ObjectMatchResult(
            object_id=object_id,
            object_label=label,
            mode=mode,
            model_name=model_name,
            raw_value=raw_value,
        )

        snippets = self._flatten_raw(raw_value)
        if not snippets:
            return result

        snippet_embs = self.bert.embed_texts(snippets)
        option_embs = self._option_embeddings.get(object_id, {})

        if not option_embs:
            return result

        option_list = list(options)
        opt_emb_array = np.stack([option_embs[opt] for opt in option_list])

        all_scores: Dict[str, float] = {}
        for emb in snippet_embs:
            sims = self.bert.batch_similarity(emb, opt_emb_array)
            for opt, sim in zip(option_list, sims):
                current = all_scores.get(opt, 0.0)
                if mode == "single":
                    all_scores[opt] = max(current, float(sim))
                else:
                    all_scores[opt] = current + float(sim)

        matches = []
        for opt in option_list:
            raw_sim = all_scores.get(opt, 0.0)
            if mode == "multi":
                raw_sim = min(raw_sim / max(len(snippets), 1), 1.0)
            confidence = self._calibrate(raw_sim, mode, threshold)
            matches.append(
                SemanticMatch(
                    option=opt,
                    similarity=float(raw_sim),
                    confidence=float(confidence),
                    is_above_threshold=raw_sim >= threshold,
                )
            )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        result.matches = matches

        if mode == "single":
            if matches and matches[0].similarity >= threshold:
                result.best_match = matches[0]
                result.selected_matches = [matches[0]]
                if (
                    len(matches) > 1
                    and matches[0].similarity - matches[1].similarity <= self.single_margin
                    and matches[1].similarity >= threshold
                ):
                    result.selected_matches.append(matches[1])
        else:
            result.selected_matches = [
                m for m in matches if m.similarity >= self.multi_threshold
            ]
            if not result.selected_matches and matches:
                result.selected_matches = matches[: min(3, len(matches))]
            if result.selected_matches:
                result.best_match = result.selected_matches[0]

        return result

    def match_all_objects(
        self,
        model_output: Dict[str, Any],
        model_name: str = "",
    ) -> Dict[str, ObjectMatchResult]:
        structured = model_output.get("structured_analysis", {}) or {}
        if not isinstance(structured, dict):
            structured = {}
        results: Dict[str, ObjectMatchResult] = {}
        for schema in LABOR_OBJECTS:
            raw = structured.get(schema.object_id)
            results[schema.object_id] = self.match_model_field(
                schema.object_id, raw, model_name
            )
        return results

    def build_bert_enhanced_patch(
        self,
        object_id: str,
        rule_facts: List[str],
        bert_match: ObjectMatchResult,
    ) -> List[str]:
        rule_set = {str(f).strip() for f in (rule_facts or []) if str(f).strip()}
        bert_facts: List[Tuple[str, float]] = [
            (m.option, m.similarity) for m in bert_match.selected_matches
        ]

        if not rule_set and not bert_facts:
            return []

        result: List[str] = []
        seen: set = set()

        for fact in rule_facts:
            f = str(fact).strip()
            if f and f not in seen:
                result.append(f)
                seen.add(f)

        for opt, sim in bert_facts:
            if opt not in seen and sim >= self.thresholds.get(object_id, 0.40):
                result.append(opt)
                seen.add(opt)

        return result

    @staticmethod
    def _flatten_raw(raw_value: Any) -> List[str]:
        if raw_value is None:
            return []
        if isinstance(raw_value, str):
            s = raw_value.strip()
            return [s] if s else []
        if isinstance(raw_value, (list, tuple, set)):
            return [
                str(item).strip()
                for item in raw_value
                if str(item).strip()
            ]
        return [str(raw_value).strip()] if str(raw_value).strip() else []

    @staticmethod
    def _calibrate(raw_sim: float, mode: str, threshold: float) -> float:
        if mode == "single":
            return min(max((raw_sim - 0.3) / 0.7, 0.0), 1.0)
        return min(max(raw_sim / 0.8, 0.0), 1.0)
