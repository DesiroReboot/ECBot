from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from src.core.search.query_preprocessor import QueryIntent, QueryPreprocessor


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class QueryAnalysis:
    temporal_intent_score: float
    domain_relevance_score: float
    oov_entity_score: float
    kb_coverage_score: float
    need_web_search: bool
    reasons: list[str] = field(default_factory=list)
    route_mode: str = "kb_only"
    query_intent: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        signals = {
            "temporal_intent_score": round(float(self.temporal_intent_score), 6),
            "domain_relevance_score": round(float(self.domain_relevance_score), 6),
            "oov_entity_score": round(float(self.oov_entity_score), 6),
            "kb_coverage_score": round(float(self.kb_coverage_score), 6),
        }
        reason_codes = list(self.reasons)
        return {
            "signals": signals,
            "reason_codes": reason_codes,
            "need_web_search": bool(self.need_web_search),
            "route_mode": str(self.route_mode or "kb_only"),
            "query_intent": dict(self.query_intent),
            # Compatibility fields for legacy readers.
            "temporal_intent_score": signals["temporal_intent_score"],
            "domain_relevance_score": signals["domain_relevance_score"],
            "oov_entity_score": signals["oov_entity_score"],
            "kb_coverage_score": signals["kb_coverage_score"],
            "reasons": reason_codes,
        }


class QueryAnalyzer:
    _TEMPORAL_KEYWORDS: tuple[str, ...] = (
        "最近",
        "最新",
        "新政策",
        "政策更新",
        "近期",
        "今年",
        "本月",
        "本周",
        "刚刚",
        "now",
        "latest",
        "recent",
        "this year",
        "this month",
    )
    _DOMAIN_ACTION_KEYWORDS: tuple[str, ...] = (
        "选品",
        "爆品",
        "供应链",
        "关税",
        "平台政策",
        "合规",
        "上架",
        "物流",
        "清关",
        "广告",
        "转化",
        "listing",
        "fba",
    )
    _DOMAIN_CONTEXT_KEYWORDS: tuple[str, ...] = (
        "外贸",
        "跨境",
        "电商",
        "亚马逊",
        "amazon",
        "shopify",
        "temu",
        "tiktok",
        "aliexpress",
        "shein",
    )
    _ENTITY_STOPWORDS: set[str] = {
        "最近",
        "最新",
        "新政策",
        "政策",
        "今年",
        "本月",
        "本周",
        "刚刚",
        "如何",
        "怎么",
        "哪些",
        "什么",
        "可以",
        "需要",
        "请问",
        "问题",
        "建议",
        "外贸",
        "跨境",
        "电商",
        "平台",
    }

    def __init__(self) -> None:
        self.preprocessor = QueryPreprocessor()

    def analyze(
        self,
        *,
        query: str,
        local_results: list[Any],
        search_trace: dict[str, Any] | None = None,
    ) -> QueryAnalysis:
        preprocess = self.preprocessor.process(query)
        lowered = str(preprocess.get("lowered", ""))
        intent = self._coerce_query_intent(preprocess.get("query_intent", {}))
        query_tokens = self._query_tokens(
            intent=intent,
            preprocess_tokens=[str(token) for token in preprocess.get("tokens", []) if str(token).strip()],
        )
        theme_hints = [str(item) for item in preprocess.get("theme_hints", []) if str(item).strip()]
        local_rows = self._coerce_local_rows(local_results)

        temporal_score = self._temporal_intent_score(lowered=lowered, intent=intent)
        domain_score = self._domain_relevance_score(
            lowered=lowered,
            theme_hints=theme_hints,
            intent=intent,
        )
        oov_score = self._oov_entity_score(
            entities=list(intent.get("core_entities", [])),
            local_rows=local_rows,
        )
        kb_coverage_score = self._kb_coverage_score(
            query_tokens=query_tokens,
            local_rows=local_rows,
            search_trace=search_trace or {},
        )

        reasons: list[str] = []
        if not local_rows:
            reasons.append("kb_no_hit")
        if temporal_score >= 0.6:
            reasons.append("temporal_intent_high")
        if domain_score >= 0.5 and oov_score >= 0.6:
            reasons.append("domain_oov_trigger")
        if kb_coverage_score < 0.35:
            reasons.append("kb_coverage_low")
        if temporal_score >= 0.45 and self._is_policy_query(intent, lowered):
            reasons.append("policy_temporal_trigger")
        if intent.get("core_entities") and oov_score >= 0.75:
            reasons.append("new_entity_oov_high")

        need_web_search = bool(reasons) or bool(intent.get("need_web_search", False))
        route_mode = self._route_mode(
            temporal_score=temporal_score,
            oov_score=oov_score,
            kb_coverage_score=kb_coverage_score,
            need_web_search=need_web_search,
            local_rows=local_rows,
            intent_route=str(intent.get("route_mode", "kb_only")),
        )
        if route_mode == "web_dominant":
            reasons.append("route_web_dominant")

        return QueryAnalysis(
            temporal_intent_score=temporal_score,
            domain_relevance_score=domain_score,
            oov_entity_score=oov_score,
            kb_coverage_score=kb_coverage_score,
            need_web_search=need_web_search,
            reasons=reasons,
            route_mode=route_mode,
            query_intent=dict(intent),
        )

    def _coerce_local_rows(self, local_results: list[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in local_results:
            if isinstance(item, dict):
                rows.append(
                    {
                        "source": str(item.get("source", "")),
                        "content": str(item.get("content", "")),
                        "score": float(item.get("score", 0.0)),
                    }
                )
                continue
            rows.append(
                {
                    "source": str(getattr(item, "source", "")),
                    "content": str(getattr(item, "content", "")),
                    "score": float(getattr(item, "score", 0.0)),
                }
            )
        return rows

    def _temporal_intent_score(self, *, lowered: str, intent: QueryIntent) -> float:
        hits = sum(1 for keyword in self._TEMPORAL_KEYWORDS if keyword in lowered)
        hits += len(intent.get("temporal_terms", []))
        if hits >= 2:
            return 0.85
        if hits == 1:
            return 0.65
        if re.search(r"\b20\d{2}\b", lowered):
            return 0.5
        return 0.0

    def _domain_relevance_score(
        self,
        *,
        lowered: str,
        theme_hints: list[str],
        intent: QueryIntent,
    ) -> float:
        entities = list(intent.get("core_entities", []))
        action_hits = sum(1 for token in self._DOMAIN_ACTION_KEYWORDS if token in lowered)
        context_hits = sum(1 for token in self._DOMAIN_CONTEXT_KEYWORDS if token in lowered)
        action_hits += len(intent.get("intent_terms", []))
        context_hits += sum(
            1 for term in intent.get("constraint_terms", []) if term in {"平台政策", "政策", "合规", "规则", "监管"}
        )
        theme_bonus = min(len(theme_hints), 2) * 0.12
        entity_bonus = 0.26 if entities else 0.0
        co_occurrence_bonus = 0.28 if action_hits > 0 and (context_hits > 0 or entities) else 0.0
        score = 0.18 * context_hits + 0.2 * action_hits + theme_bonus + entity_bonus + co_occurrence_bonus
        return _clamp(score)

    def _oov_entity_score(self, *, entities: list[str], local_rows: list[dict[str, Any]]) -> float:
        if not entities:
            return 0.0
        haystack = " ".join(
            (f"{row.get('source', '')} {row.get('content', '')}".lower() for row in local_rows)
        )
        if not haystack.strip():
            return 0.9
        missing = sum(1 for entity in entities if entity.lower() not in haystack)
        missing_ratio = missing / max(len(entities), 1)
        if missing_ratio >= 0.9:
            return 0.9
        if missing_ratio >= 0.6:
            return 0.72
        if missing_ratio >= 0.3:
            return 0.5
        return 0.2

    def _kb_coverage_score(
        self,
        *,
        query_tokens: list[str],
        local_rows: list[dict[str, Any]],
        search_trace: dict[str, Any],
    ) -> float:
        if not local_rows:
            return 0.0
        top_rows = local_rows[:5]
        avg_score = sum(_clamp(float(row.get("score", 0.0))) for row in top_rows) / max(len(top_rows), 1)
        token_hits = 0
        compact_tokens = [token for token in query_tokens if len(token) >= 2]
        for token in compact_tokens:
            if any(token in str(row.get("content", "")).lower() for row in top_rows):
                token_hits += 1
        token_coverage = token_hits / max(len(compact_tokens), 1) if compact_tokens else 0.0
        source_diversity = (
            len({str(row.get("source", "")).strip() for row in top_rows if str(row.get("source", "")).strip()})
            / max(len(top_rows), 1)
        )

        errors = search_trace.get("errors", []) if isinstance(search_trace, dict) else []
        generation = search_trace.get("generation", {}) if isinstance(search_trace, dict) else {}
        branch_errors = generation.get("branch_errors", {}) if isinstance(generation, dict) else {}
        penalty = 0.0
        if isinstance(errors, list) and errors:
            penalty += 0.08
        if isinstance(branch_errors, dict) and branch_errors and not search_trace.get("fts_recall"):
            penalty += 0.12

        score = 0.55 * avg_score + 0.3 * token_coverage + 0.15 * source_diversity - penalty
        return _clamp(score)

    def _route_mode(
        self,
        *,
        temporal_score: float,
        oov_score: float,
        kb_coverage_score: float,
        need_web_search: bool,
        local_rows: list[dict[str, Any]],
        intent_route: str,
    ) -> str:
        if not need_web_search:
            return "kb_only"
        if not local_rows:
            return "web_dominant"
        if temporal_score >= 0.75:
            return "web_dominant"
        if oov_score >= 0.72 and kb_coverage_score <= 0.45:
            return "web_dominant"
        if intent_route == "web_dominant" and temporal_score >= 0.5:
            return "web_dominant"
        return "hybrid"

    def _is_policy_query(self, intent: QueryIntent, lowered: str) -> bool:
        constraints = set(str(item) for item in intent.get("constraint_terms", []))
        if constraints & {"平台政策", "政策", "合规", "监管", "规则", "法规"}:
            return True
        markers = ("政策", "合规", "监管", "规则", "法规", "policy", "compliance")
        return any(marker in lowered for marker in markers)

    def _query_tokens(self, *, intent: QueryIntent, preprocess_tokens: list[str]) -> list[str]:
        merged = (
            list(intent.get("core_entities", []))
            + list(intent.get("intent_terms", []))
            + list(intent.get("constraint_terms", []))
            + preprocess_tokens
        )
        deduped: list[str] = []
        seen: set[str] = set()
        for token in merged:
            text = str(token).strip().lower()
            if len(text) < 2:
                continue
            if text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    def _coerce_query_intent(self, raw: Any) -> QueryIntent:
        if not isinstance(raw, dict):
            return {
                "core_entities": [],
                "intent_terms": [],
                "constraint_terms": [],
                "temporal_terms": [],
                "need_web_search": False,
                "route_mode": "kb_only",
            }

        def _list_value(key: str) -> list[str]:
            value = raw.get(key, [])
            if not isinstance(value, list):
                return []
            return [str(item).strip() for item in value if str(item).strip()]

        route_mode = str(raw.get("route_mode", "kb_only")).strip() or "kb_only"
        if route_mode not in {"kb_only", "hybrid", "web_dominant"}:
            route_mode = "kb_only"
        return {
            "core_entities": _list_value("core_entities"),
            "intent_terms": _list_value("intent_terms"),
            "constraint_terms": _list_value("constraint_terms"),
            "temporal_terms": _list_value("temporal_terms"),
            "need_web_search": bool(raw.get("need_web_search", False)),
            "route_mode": route_mode,
        }
