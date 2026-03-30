from __future__ import annotations

from types import SimpleNamespace

from src.core.search.lite_gate import (
    LOW_RELEVANCE_REASON_CODE,
    build_template_response,
    compute_l1_confidence,
    should_trigger_full_rag,
)


def test_compute_l1_confidence_uses_hits_and_metrics() -> None:
    hits = [SimpleNamespace(score=0.82), SimpleNamespace(score=0.64), SimpleNamespace(score=0.51)]
    trace = {
        "metrics": {
            "coverage_score": 0.7,
            "evidence_count": 3,
        }
    }
    confidence = compute_l1_confidence(hits, trace)
    assert 0.6 < confidence < 1.0


def test_should_trigger_full_rag_respects_threshold() -> None:
    assert should_trigger_full_rag(0.58, 0.58) is True
    assert should_trigger_full_rag(0.579, 0.58) is False


def test_build_template_response_contains_reason_and_query() -> None:
    answer = build_template_response("latest policy", LOW_RELEVANCE_REASON_CODE)
    assert "latest policy" in answer
    assert LOW_RELEVANCE_REASON_CODE in answer
