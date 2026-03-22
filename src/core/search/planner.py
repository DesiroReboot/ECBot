from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Any, Literal


SourceRoute = Literal["kb_only", "web_only", "hybrid"]
FusionStrategy = Literal["none", "direct_fusion", "rag_fusion"]


@dataclass
class PlannerOutput:
    plan_id: str
    need_web_search: bool
    source_route: SourceRoute
    fusion_strategy: FusionStrategy
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    query_expansion: dict[str, Any] = field(default_factory=dict)
    retrieval_plan: dict[str, Any] = field(default_factory=dict)


class Planner:
    """Planner interface for search routing decisions."""

    def plan(self, query: str, *, trace_context: dict[str, Any] | None = None) -> PlannerOutput:
        raise NotImplementedError


class RulePlanner(Planner):
    """Default lightweight planner.

    Note:
        This planner can decide `need_web_search=true`, but current orchestrator
        keeps Web search in reserved mode and does not execute it yet.
    """

    _TEMPORAL_HINTS = (
        "最新",
        "最近",
        "近期",
        "本周",
        "本月",
        "今年",
        "today",
        "latest",
        "recent",
    )

    def plan(self, query: str, *, trace_context: dict[str, Any] | None = None) -> PlannerOutput:
        query_text = str(query or "").strip()
        lowered = query_text.lower()
        has_temporal_intent = any(token in lowered for token in self._TEMPORAL_HINTS)
        need_web_search = has_temporal_intent
        source_route: SourceRoute = "hybrid" if need_web_search else "kb_only"
        fusion_strategy: FusionStrategy = "direct_fusion" if need_web_search else "none"

        reasons = ["temporal_intent_detected"] if need_web_search else ["kb_default_route"]
        confidence = 0.8 if need_web_search else 0.9
        if not query_text:
            reasons = ["empty_query"]
            source_route = "kb_only"
            fusion_strategy = "none"
            need_web_search = False
            confidence = 0.0

        return PlannerOutput(
            plan_id=self._build_plan_id(query_text=query_text),
            need_web_search=need_web_search,
            source_route=source_route,
            fusion_strategy=fusion_strategy,
            confidence=confidence,
            reasons=reasons,
            query_expansion={"core_terms": [], "bridge_terms": [], "synonyms": []},
            retrieval_plan={
                "sources": [
                    {"name": "kb_index", "enabled": True, "priority": 1},
                    {"name": "web_search", "enabled": need_web_search, "priority": 2},
                ],
                "execution_mode": "cascade",
                "fallback_policy": "web_to_kb",
            },
        )

    def _build_plan_id(self, *, query_text: str) -> str:
        if not query_text:
            return "plan-empty"
        digest = hashlib.md5(query_text.encode("utf-8")).hexdigest()[:12]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"plan-{stamp}-{digest}"
