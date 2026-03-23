from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.search.query_analyzer import QueryAnalysis
from src.core.search.web_result_evaluator import WebEvaluation


@dataclass
class WebRouteDecision:
    fusion_strategy: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "fusion_strategy": self.fusion_strategy,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "fallback": bool(self.fallback),
        }


class WebRouter:
    def __init__(
        self,
        *,
        direct_thresholds: dict[str, float] | None = None,
    ) -> None:
        defaults = {
            "result_count_max": 8.0,
            "top3_mean_min": 0.72,
            "score_gap_min": 0.08,
            "noise_ratio_max": 0.25,
        }
        thresholds = dict(defaults)
        if isinstance(direct_thresholds, dict):
            thresholds.update({key: float(value) for key, value in direct_thresholds.items()})
        self.direct_thresholds = thresholds

    def route(
        self,
        *,
        query: str,
        analysis: QueryAnalysis,
        evaluation: WebEvaluation,
    ) -> WebRouteDecision:
        reasons: list[str] = []
        metrics = evaluation.to_dict()
        metrics.update(analysis.to_dict())

        if evaluation.result_count <= 0:
            return WebRouteDecision(
                fusion_strategy="none",
                reasons=["web_no_results"],
                metrics=metrics,
                fallback=True,
            )

        needs_traceable = self._needs_traceable_evidence(query)
        direct_ok = self._is_direct_fusion(evaluation)
        rag_signals = self._rag_signals(evaluation=evaluation, needs_traceable=needs_traceable)

        if direct_ok and not rag_signals:
            reasons.append("direct_fusion_thresholds_met")
            return WebRouteDecision(
                fusion_strategy="direct_fusion",
                reasons=reasons,
                metrics=metrics,
            )

        if rag_signals:
            reasons.extend(rag_signals)
            return WebRouteDecision(
                fusion_strategy="rag_fusion",
                reasons=reasons,
                metrics=metrics,
            )

        if analysis.need_web_search:
            reasons.append("default_direct_for_web_needed")
            return WebRouteDecision(
                fusion_strategy="direct_fusion",
                reasons=reasons,
                metrics=metrics,
            )

        return WebRouteDecision(
            fusion_strategy="none",
            reasons=["web_not_required"],
            metrics=metrics,
            fallback=True,
        )

    def _is_direct_fusion(self, evaluation: WebEvaluation) -> bool:
        thresholds = self.direct_thresholds
        return (
            evaluation.result_count <= int(thresholds["result_count_max"])
            and evaluation.top3_mean >= float(thresholds["top3_mean_min"])
            and evaluation.score_gap >= float(thresholds["score_gap_min"])
            and evaluation.noise_ratio <= float(thresholds["noise_ratio_max"])
        )

    def _rag_signals(self, *, evaluation: WebEvaluation, needs_traceable: bool) -> list[str]:
        reasons: list[str] = []
        if evaluation.result_count > int(self.direct_thresholds["result_count_max"]):
            reasons.append("result_count_large")
        if evaluation.domain_diversity >= 0.6 and evaluation.score_gap <= 0.08:
            reasons.append("topic_dispersion_high")
        if evaluation.conflict_detected:
            reasons.append("conflict_detected")
        if needs_traceable:
            reasons.append("traceable_evidence_required")
        return reasons

    def _needs_traceable_evidence(self, query: str) -> bool:
        lowered = str(query or "").lower()
        markers = ("政策", "合规", "法规", "监管", "compliance", "policy", "regulation")
        return any(marker in lowered for marker in markers)
