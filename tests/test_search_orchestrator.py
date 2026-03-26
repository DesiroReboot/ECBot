from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.search.orchestrator import SearchOrchestrator
from src.core.search.planner import PlannerOutput
from src.core.search.rag_search import SearchResult


@dataclass
class _StubPlanner:
    output: PlannerOutput

    def plan(self, query: str, *, trace_context: dict[str, Any] | None = None) -> PlannerOutput:  # noqa: ARG002
        return self.output


class _StubRAGSearcher:
    def search_with_trace(self, query: str) -> tuple[list[SearchResult], dict[str, Any]]:  # noqa: ARG002
        return (
            [
                SearchResult(
                    file_uuid="f1",
                    source="kb-a.md",
                    content="内容A",
                    score=0.82,
                    chunk_id=1,
                    source_path="/kb/kb-a.md",
                    section_title="s1",
                ),
                SearchResult(
                    file_uuid="f2",
                    source="kb-b.md",
                    content="内容B",
                    score=0.64,
                    chunk_id=2,
                    source_path="/kb/kb-b.md",
                    section_title="s2",
                ),
            ],
            {"fts_recall": [{"source": "kb-a.md"}], "generation": {"branch_errors": {}}},
        )


class _StubWebSearcher:
    def __init__(self) -> None:
        self.called = 0

    def search_with_trace(
        self,
        query: str,
        *,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # noqa: ARG002
        self.called += 1
        return ([], {"ok": True})


def test_orchestrator_uses_planner_fields_and_skips_web_execution() -> None:
    planner = _StubPlanner(
        output=PlannerOutput(
            plan_id="plan-1",
            need_web_search=True,
            source_route="hybrid",
            fusion_strategy="rag_fusion",
            reasons=["temporal_intent_detected"],
            retrieval_plan={"sources": [{"name": "kb_index", "enabled": True}]},
        )
    )
    web_searcher = _StubWebSearcher()
    orchestrator = SearchOrchestrator(
        planner=planner,
        rag_searcher=_StubRAGSearcher(),
        web_searcher=web_searcher,
        config=None,
    )

    result = orchestrator.search_with_trace("最近平台政策")

    assert web_searcher.called == 0
    assert len(result.hits) == 2
    assert result.trace_search["planner"]["source_route"] == "hybrid"
    assert result.trace_search["planner"]["fusion_strategy"] == "rag_fusion"
    assert result.trace_search["web"]["executed"] is False
    assert result.trace_search["web"]["skip_reason"] == "web_execution_delegated_to_bot_agent"
