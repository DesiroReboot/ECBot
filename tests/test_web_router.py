from __future__ import annotations

from src.core.search.query_analyzer import QueryAnalysis
from src.core.search.web_result_evaluator import WebEvaluation
from src.core.search.web_router import WebRouter


def _analysis(need_web_search: bool = True) -> QueryAnalysis:
    return QueryAnalysis(
        temporal_intent_score=0.7,
        domain_relevance_score=0.7,
        oov_entity_score=0.8,
        kb_coverage_score=0.4,
        need_web_search=need_web_search,
        reasons=["temporal_intent_high"],
    )


def test_web_router_prefers_direct_fusion_for_precise_results() -> None:
    router = WebRouter()
    decision = router.route(
        query="最近亚马逊选品趋势",
        analysis=_analysis(),
        evaluation=WebEvaluation(
            result_count=5,
            top1_score=0.95,
            top3_mean=0.81,
            score_gap=0.14,
            domain_diversity=0.2,
            freshness_ratio=0.7,
            noise_ratio=0.1,
            conflict_detected=False,
        ),
    )

    assert decision.fusion_strategy == "direct_fusion"
    assert decision.fallback is False
    assert "direct_fusion_thresholds_met" in decision.reasons


def test_web_router_prefers_rag_fusion_for_large_or_conflicting_results() -> None:
    router = WebRouter()
    decision = router.route(
        query="今年最新平台政策合规变化",
        analysis=_analysis(),
        evaluation=WebEvaluation(
            result_count=16,
            top1_score=0.83,
            top3_mean=0.72,
            score_gap=0.03,
            domain_diversity=0.71,
            freshness_ratio=0.62,
            noise_ratio=0.14,
            conflict_detected=True,
        ),
    )

    assert decision.fusion_strategy == "rag_fusion"
    assert decision.fallback is False
    assert "result_count_large" in decision.reasons
    assert "conflict_detected" in decision.reasons


def test_web_router_fallbacks_when_no_web_results() -> None:
    router = WebRouter()
    decision = router.route(
        query="最近平台政策",
        analysis=_analysis(),
        evaluation=WebEvaluation(
            result_count=0,
            top1_score=0.0,
            top3_mean=0.0,
            score_gap=0.0,
            domain_diversity=0.0,
            freshness_ratio=0.0,
            noise_ratio=0.0,
            conflict_detected=False,
        ),
    )

    assert decision.fusion_strategy == "none"
    assert decision.fallback is True
    assert decision.reasons == ["web_no_results"]
