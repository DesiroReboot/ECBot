from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Any, Literal

from src.core.search.domain_filter import DomainFilter, DomainFilterResult


SourceRoute = Literal["kb_only", "web_only", "hybrid"]
FusionStrategy = Literal["none", "direct_fusion", "rag_fusion"]
RouteMode = Literal["none", "serial", "parallel"]


@dataclass
class PlannerOutput:
    plan_id: str
    need_web_search: bool
    source_route: SourceRoute
    fusion_strategy: FusionStrategy
    allow_rag: bool = True
    filter_reason: str = ""
    domain_relevance_score: float = 1.0
    domain_filter: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    route_mode: RouteMode = "none"
    reasons: list[str] = field(default_factory=list)
    query_expansion: dict[str, Any] = field(default_factory=dict)
    retrieval_plan: dict[str, Any] = field(default_factory=dict)


class Planner:
    """Planner interface for search routing decisions."""

    def plan(self, query: str, *, trace_context: dict[str, Any] | None = None) -> PlannerOutput:
        raise NotImplementedError


class RulePlanner(Planner):
    """Default lightweight planner with optional domain filter."""

    def __init__(
        self,
        *,
        domain_filter_enabled: bool = True,
        domain_filter_threshold: float = 0.45,
        domain_filter_fail_open: bool = True,
        domain_filter: DomainFilter | None = None,
    ) -> None:
        self.domain_filter_enabled = bool(domain_filter_enabled)
        self.domain_filter_fail_open = bool(domain_filter_fail_open)
        self.domain_filter = domain_filter or DomainFilter(threshold=domain_filter_threshold)

    def plan(self, query: str, *, trace_context: dict[str, Any] | None = None) -> PlannerOutput:
        query_text = str(query or "").strip()
        analyzer_view = self._extract_analyzer_view(trace_context)

        need_web_search = bool(analyzer_view["need_web_search"])
        source_route: SourceRoute = "hybrid" if need_web_search else "kb_only"
        route_mode: RouteMode = "serial" if need_web_search else "none"
        # Planner only keeps two-state routing (must/no). Concrete web fusion strategy
        # is decided later in orchestrator by web evaluation.
        fusion_strategy: FusionStrategy = "none"
        reasons = list(analyzer_view["reason_codes"]) or ["analyzer_default_no_web"]
        confidence = 0.78 if need_web_search else 0.9

        if not query_text:
            reasons = ["empty_query"]
            source_route = "kb_only"
            route_mode = "none"
            fusion_strategy = "none"
            need_web_search = False
            confidence = 0.0

        domain_result = self._check_domain_filter(query_text)
        reasons.append(f"domain_filter:{domain_result.reason}")

        if not domain_result.allow_rag:
            need_web_search = False
            source_route = "kb_only"
            route_mode = "none"
            fusion_strategy = "none"
            confidence = 0.0
            reasons.append("rag_blocked_by_domain_filter")

        return PlannerOutput(
            plan_id=self._build_plan_id(query_text=query_text),
            need_web_search=need_web_search,
            source_route=source_route,
            fusion_strategy=fusion_strategy,
            allow_rag=domain_result.allow_rag,
            filter_reason=domain_result.reason,
            domain_relevance_score=domain_result.score,
            domain_filter=domain_result.to_trace_dict(),
            confidence=confidence,
            route_mode=route_mode,
            reasons=reasons,
            query_expansion={"core_terms": [], "bridge_terms": [], "synonyms": []},
            retrieval_plan={
                "sources": [
                    {"name": "kb_index", "enabled": domain_result.allow_rag, "priority": 1},
                    {
                        "name": "web_search",
                        "enabled": domain_result.allow_rag and need_web_search,
                        "priority": 2,
                    },
                ],
                "execution_mode": "serial" if need_web_search else "rag_only",
                "fallback_policy": "phase_a_rag_to_web_on_low_confidence",
            },
        )

    def _extract_analyzer_view(self, trace_context: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(trace_context, dict):
            return {"need_web_search": False, "reason_codes": []}

        raw = trace_context.get("query_analysis")
        if not isinstance(raw, dict):
            return {"need_web_search": False, "reason_codes": []}

        reason_codes = raw.get("reason_codes")
        if not isinstance(reason_codes, list):
            reason_codes = raw.get("reasons")
        normalized_reasons = [
            str(item).strip()
            for item in (reason_codes if isinstance(reason_codes, list) else [])
            if str(item).strip()
        ]
        return {
            "need_web_search": bool(raw.get("need_web_search", False)),
            "reason_codes": normalized_reasons,
        }

    def _check_domain_filter(self, query_text: str) -> DomainFilterResult:
        if not self.domain_filter_enabled:
            return DomainFilterResult(
                allow_rag=True,
                reason="domain_filter_disabled",
                score=1.0,
                threshold=float(self.domain_filter.threshold),
            )
        try:
            return self.domain_filter.check(query_text)
        except Exception:
            if self.domain_filter_fail_open:
                return DomainFilterResult(
                    allow_rag=True,
                    reason="domain_filter_error_fail_open",
                    score=1.0,
                    threshold=float(self.domain_filter.threshold),
                )
            return DomainFilterResult(
                allow_rag=False,
                reason="domain_filter_error_fail_closed",
                score=0.0,
                threshold=float(self.domain_filter.threshold),
            )

    def _build_plan_id(self, *, query_text: str) -> str:
        if not query_text:
            return "plan-empty"
        digest = hashlib.md5(query_text.encode("utf-8")).hexdigest()[:12]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"plan-{stamp}-{digest}"
