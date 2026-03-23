from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from src.core.search.query_preprocessor import QueryPreprocessor


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "temporal_intent_score": round(float(self.temporal_intent_score), 6),
            "domain_relevance_score": round(float(self.domain_relevance_score), 6),
            "oov_entity_score": round(float(self.oov_entity_score), 6),
            "kb_coverage_score": round(float(self.kb_coverage_score), 6),
            "need_web_search": bool(self.need_web_search),
            "reasons": list(self.reasons),
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
        query_tokens = [str(token) for token in preprocess.get("tokens", []) if str(token).strip()]
        theme_hints = [str(item) for item in preprocess.get("theme_hints", []) if str(item).strip()]
        local_rows = self._coerce_local_rows(local_results)
        entity_candidates = self._extract_entity_candidates(query)

        temporal_score = self._temporal_intent_score(lowered)
        domain_score = self._domain_relevance_score(
            lowered=lowered,
            theme_hints=theme_hints,
            entities=entity_candidates,
        )
        oov_score = self._oov_entity_score(entities=entity_candidates, local_rows=local_rows)
        kb_coverage_score = self._kb_coverage_score(
            query_tokens=query_tokens,
            local_rows=local_rows,
            search_trace=search_trace or {},
        )

        reasons: list[str] = []
        if temporal_score >= 0.6:
            reasons.append("temporal_intent_high")
        if domain_score >= 0.5 and oov_score >= 0.6:
            reasons.append("domain_oov_trigger")
        if kb_coverage_score < 0.35:
            reasons.append("kb_coverage_low")

        return QueryAnalysis(
            temporal_intent_score=temporal_score,
            domain_relevance_score=domain_score,
            oov_entity_score=oov_score,
            kb_coverage_score=kb_coverage_score,
            need_web_search=bool(reasons),
            reasons=reasons,
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

    def _temporal_intent_score(self, lowered: str) -> float:
        hits = sum(1 for keyword in self._TEMPORAL_KEYWORDS if keyword in lowered)
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
        entities: list[str],
    ) -> float:
        action_hits = sum(1 for token in self._DOMAIN_ACTION_KEYWORDS if token in lowered)
        context_hits = sum(1 for token in self._DOMAIN_CONTEXT_KEYWORDS if token in lowered)
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

    def _extract_entity_candidates(self, query: str) -> list[str]:
        candidates: list[str] = []

        for raw_span in re.findall(r"[\u4e00-\u9fff]{2,16}", query):
            span = raw_span
            for token in self._ENTITY_STOPWORDS:
                span = span.replace(token, " ")
            for part in re.split(r"\s+", span):
                normalized = part.strip()
                if len(normalized) < 2:
                    continue
                if normalized in self._ENTITY_STOPWORDS:
                    continue
                candidates.append(normalized)

        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query):
            normalized = token.strip().lower()
            if normalized in self._ENTITY_STOPWORDS:
                continue
            candidates.append(normalized)

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped[:8]
