from __future__ import annotations

from src.core.search.query_analyzer import QueryAnalyzer
from src.core.search.rag_search import SearchResult


def _result(*, source: str, content: str, score: float = 0.8) -> SearchResult:
    return SearchResult(
        file_uuid=f"id-{source}",
        source=source,
        content=content,
        score=score,
        chunk_id=0,
    )


def test_query_analyzer_triggers_for_temporal_intent() -> None:
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze(
        query="最近亚马逊平台政策有什么变化？",
        local_results=[
            _result(source="policy.md", content="亚马逊平台政策更新与合规要求。", score=0.9),
        ],
        search_trace={},
    )

    assert analysis.need_web_search is True
    assert analysis.temporal_intent_score >= 0.6
    assert "temporal_intent_high" in analysis.reasons


def test_query_analyzer_triggers_for_domain_oov_pair() -> None:
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze(
        query="拉布布在亚马逊选品上有什么风险？",
        local_results=[
            _result(
                source="listing-guide.md",
                content="亚马逊选品需要评估供应链、利润和平台合规。",
                score=0.82,
            ),
        ],
        search_trace={},
    )

    assert analysis.domain_relevance_score >= 0.5
    assert analysis.oov_entity_score >= 0.6
    assert analysis.need_web_search is True
    assert "domain_oov_trigger" in analysis.reasons


def test_query_analyzer_triggers_for_low_kb_coverage() -> None:
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze(
        query="跨境电商物流怎么做",
        local_results=[],
        search_trace={},
    )

    assert analysis.kb_coverage_score == 0.0
    assert analysis.need_web_search is True
    assert "kb_coverage_low" in analysis.reasons
