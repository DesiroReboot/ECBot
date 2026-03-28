from __future__ import annotations

from types import SimpleNamespace

from src.config import GenerationConfig
from src.core.bot_agent import ReActAgent
from src.core.search.query_analyzer import QueryAnalyzer
from src.core.search.rag_search import SearchResult
from src.core.search.web_result_evaluator import WebResultEvaluator
from src.core.search.web_router import WebRouter
from src.core.search.web_search_client import WebSearchResult


class _DummySearcher:
    def __init__(self, results: list[SearchResult], trace: dict | None = None) -> None:
        self._results = results
        self._trace = trace or {"fts_recall": [{"source": "local.md"}], "generation": {"branch_errors": {}}}

    def search_with_trace(self, query: str):  # noqa: ARG002
        return list(self._results), dict(self._trace)


def _local_result() -> SearchResult:
    return SearchResult(
        file_uuid="local-1",
        source="local-guide.md",
        content="Amazon product selection should start from demand and profitability signals.",
        score=0.76,
        chunk_id=0,
    )


def _web_result(*, title: str, url: str, snippet: str, score: float) -> WebSearchResult:
    return WebSearchResult(
        title=title,
        url=url,
        snippet=snippet,
        score=score,
        source_domain="news.example.com",
        published_at="2026-03-20",
    )


def _build_agent(
    *,
    web_enabled: bool,
    web_results: list[WebSearchResult],
    local_results: list[SearchResult] | None = None,
    web_error: Exception | None = None,
) -> ReActAgent:
    agent = ReActAgent.__new__(ReActAgent)
    search_cfg = SimpleNamespace(
        web_search_enabled=web_enabled,
        web_search_provider="tavily",
        web_search_timeout=8,
        web_search_retries=1,
        web_search_tavily_api_key="test-key",
        web_search_tavily_base_url="https://api.tavily.com/search",
        web_search_depth="basic",
        web_search_max_results=8,
        web_direct_fusion_thresholds={
            "result_count_max": 8.0,
            "top3_mean_min": 0.72,
            "score_gap_min": 0.08,
            "noise_ratio_max": 0.25,
        },
        web_rag_max_docs=16,
    )
    agent.config = SimpleNamespace(
        search=search_cfg,
        generation=GenerationConfig(mode="template"),
    )
    agent.answer_top_k = 3
    agent.searcher = _DummySearcher(local_results if local_results is not None else [_local_result()])
    agent.manifest_store = SimpleNamespace(get_manifest=lambda: {"version": "ok"})
    agent.query_analyzer = QueryAnalyzer()
    if web_error is None:
        agent.web_search_client = SimpleNamespace(search=lambda query, limit: list(web_results))  # noqa: ARG005
    else:
        agent.web_search_client = SimpleNamespace(search=lambda query, limit: (_ for _ in ()).throw(web_error))  # noqa: ARG005
    agent.web_result_evaluator = WebResultEvaluator()
    agent.web_router = WebRouter(direct_thresholds=search_cfg.web_direct_fusion_thresholds)
    agent.generation_client = SimpleNamespace()
    return agent


def test_run_sync_applies_direct_fusion_and_writes_web_trace() -> None:
    web_rows = [
        _web_result(
            title="Fabric category trend report",
            url="https://news.example.com/a",
            snippet="Recent demand and sourcing indicators show sustained momentum.",
            score=0.95,
        ),
        _web_result(
            title="Cross-border seller category watch",
            url="https://news.example.com/b",
            snippet="Category movement is now concentrated in giftable products.",
            score=0.72,
        ),
        _web_result(
            title="Supply chain update",
            url="https://news.example.com/c",
            snippet="Upstream price volatility has narrowed and lead time improved.",
            score=0.7,
        ),
    ]
    agent = _build_agent(web_enabled=True, web_results=web_rows)
    response = agent.run_sync("latest product selection trend", include_trace=True)

    web_trace = response.trace["search"]["web"]
    assert web_trace["need_web_search"] is True
    assert web_trace["fusion_strategy"] == "direct_fusion"
    assert web_trace["fallback_used"] is False
    assert web_trace["metrics"]["result_count"] == 3


def test_run_sync_fallbacks_to_local_when_web_empty() -> None:
    agent = _build_agent(web_enabled=True, web_results=[])
    response = agent.run_sync("latest amazon category trend", include_trace=True)

    web_trace = response.trace["search"]["web"]
    assert web_trace["need_web_search"] is True
    assert web_trace["fusion_strategy"] == "none"
    assert web_trace["fallback_used"] is True
    assert "web_no_results" in web_trace["reasons"]


def test_kb_empty_auto_triggers_web_fallback_chain() -> None:
    web_rows = [
        _web_result(
            title="Policy change",
            url="https://news.example.com/policy",
            snippet="The platform published a new policy update this week.",
            score=0.88,
        )
    ]
    agent = _build_agent(web_enabled=True, web_results=web_rows, local_results=[])
    response = agent.run_sync("latest platform policy updates", include_trace=True)

    web_trace = response.trace["search"]["web"]
    assert web_trace["need_web_search"] is True
    assert "kb_empty_triggered_web_fallback" in web_trace["reasons"]
    assert web_trace["fallback_used"] is False
    assert all(step.get("stage") != "fallback_answer" for step in response.trace["strategy_execution"])


def test_web_provider_error_records_reason_and_returns_safe_fallback() -> None:
    agent = _build_agent(
        web_enabled=True,
        web_results=[],
        local_results=[],
        web_error=RuntimeError("provider_misconfigured:tavily_api_key_missing"),
    )
    response = agent.run_sync("latest platform policy updates", include_trace=True)

    web_trace = response.trace["search"]["web"]
    assert web_trace["need_web_search"] is True
    assert web_trace["fallback_used"] is True
    assert "web_search_error" in web_trace["reasons"]
    assert "provider_misconfigured" in web_trace["reasons"]
    assert any(step.get("stage") == "fallback_answer" for step in response.trace["strategy_execution"])


def test_run_sync_legacy_web_fallback_field_is_read_once() -> None:
    agent = _build_agent(web_enabled=False, web_results=[])
    agent.search_orchestrator = SimpleNamespace(
        search_with_trace=lambda query: SimpleNamespace(  # noqa: ARG005
            hits=[_local_result()],
            citations=[],
            retrieval_confidence=0.7,
            trace_search={
                "web": {
                    "need_web_search": True,
                    "fusion_strategy": "none",
                    "reasons": ["legacy"],
                    "metrics": {},
                    "fallback": True,
                }
            },
        )
    )

    response = agent.run_sync("legacy trace", include_trace=True)
    web_trace = response.trace["search"]["web"]
    assert web_trace["fallback_used"] is True
    assert response.trace["search"]["compat"]["trace_web_fallback_legacy_read"] is True
