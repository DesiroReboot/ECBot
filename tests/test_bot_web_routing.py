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
        content="亚马逊选品需要先看需求，再评估竞争强度和利润。",
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


def _build_agent(*, web_enabled: bool, web_results: list[WebSearchResult]) -> ReActAgent:
    agent = ReActAgent.__new__(ReActAgent)
    search_cfg = SimpleNamespace(
        web_search_enabled=web_enabled,
        web_search_provider="mock",
        web_search_timeout=8,
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
    agent.searcher = _DummySearcher([_local_result()])
    agent.manifest_store = SimpleNamespace(get_manifest=lambda: {"version": "ok"})
    agent.query_analyzer = QueryAnalyzer()
    agent.web_search_client = SimpleNamespace(search=lambda query, limit: list(web_results))  # noqa: ARG005
    agent.web_result_evaluator = WebResultEvaluator()
    agent.web_router = WebRouter(direct_thresholds=search_cfg.web_direct_fusion_thresholds)
    agent.generation_client = SimpleNamespace()
    return agent


def test_run_sync_applies_direct_fusion_and_writes_web_trace() -> None:
    web_rows = [
        _web_result(
            title="拉布布选品趋势报告",
            url="https://news.example.com/a",
            snippet="最近拉布布相关选品热度上升，供应链开始扩容。",
            score=0.95,
        ),
        _web_result(
            title="跨境卖家选品观察",
            url="https://news.example.com/b",
            snippet="平台类目近期变化主要集中在潮玩和礼品。",
            score=0.72,
        ),
        _web_result(
            title="拉布布供应链动态",
            url="https://news.example.com/c",
            snippet="今年上游原料价格趋稳，交付周期缩短。",
            score=0.7,
        ),
    ]
    agent = _build_agent(web_enabled=True, web_results=web_rows)
    response = agent.run_sync("最近拉布布选品趋势如何？", include_trace=True)

    web_trace = response.trace["search"]["web"]
    assert web_trace["need_web_search"] is True
    assert web_trace["fusion_strategy"] == "direct_fusion"
    assert web_trace["fallback"] is False
    assert web_trace["metrics"]["result_count"] == 3


def test_run_sync_fallbacks_to_local_when_web_empty() -> None:
    agent = _build_agent(web_enabled=True, web_results=[])
    response = agent.run_sync("最近亚马逊选品趋势如何？", include_trace=True)

    web_trace = response.trace["search"]["web"]
    assert web_trace["need_web_search"] is True
    assert web_trace["fusion_strategy"] == "none"
    assert web_trace["fallback"] is True
    assert "web_no_results" in web_trace["reasons"]
