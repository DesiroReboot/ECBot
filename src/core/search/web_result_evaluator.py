from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from typing import Any

from src.core.search.web_search_client import WebSearchResult


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class WebEvaluation:
    result_count: int
    top1_score: float
    top3_mean: float
    score_gap: float
    domain_diversity: float
    freshness_ratio: float
    noise_ratio: float
    conflict_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_count": int(self.result_count),
            "top1_score": round(float(self.top1_score), 6),
            "top3_mean": round(float(self.top3_mean), 6),
            "score_gap": round(float(self.score_gap), 6),
            "domain_diversity": round(float(self.domain_diversity), 6),
            "freshness_ratio": round(float(self.freshness_ratio), 6),
            "noise_ratio": round(float(self.noise_ratio), 6),
            "conflict_detected": bool(self.conflict_detected),
        }


class WebResultEvaluator:
    _LOW_QUALITY_URL_MARKERS: tuple[str, ...] = (
        "ads",
        "adservice",
        "redirect",
        "click",
        "utm_",
        "sponsored",
    )
    _RESTRICT_MARKERS: tuple[str, ...] = ("禁止", "限制", "下架", "ban", "restriction", "penalty")
    _RELAX_MARKERS: tuple[str, ...] = ("放宽", "允许", "支持", "恢复", "allow", "approved")

    def evaluate(
        self,
        *,
        query: str,
        results: list[WebSearchResult],
        freshness_window_days: int = 180,
    ) -> WebEvaluation:
        deduped = self._dedupe(results)
        ranked = sorted(
            deduped,
            key=lambda item: self._effective_score(item=item, query=query),
            reverse=True,
        )
        result_count = len(ranked)
        score_series = [self._effective_score(item=item, query=query) for item in ranked]
        top1_score = score_series[0] if score_series else 0.0
        top3_mean = sum(score_series[:3]) / max(min(3, len(score_series)), 1)
        top5_mean = sum(score_series[:5]) / max(min(5, len(score_series)), 1)
        score_gap = top1_score - top5_mean if score_series else 0.0

        unique_domains = {item.source_domain for item in ranked if item.source_domain}
        domain_diversity = len(unique_domains) / max(result_count, 1)
        freshness_ratio = self._freshness_ratio(ranked=ranked, freshness_window_days=freshness_window_days)
        noise_ratio = self._noise_ratio(ranked)
        conflict_detected = self._conflict_detected(ranked)

        return WebEvaluation(
            result_count=result_count,
            top1_score=_clamp(top1_score),
            top3_mean=_clamp(top3_mean),
            score_gap=_clamp(score_gap),
            domain_diversity=_clamp(domain_diversity),
            freshness_ratio=_clamp(freshness_ratio),
            noise_ratio=_clamp(noise_ratio),
            conflict_detected=conflict_detected,
        )

    def _dedupe(self, results: list[WebSearchResult]) -> list[WebSearchResult]:
        deduped: list[WebSearchResult] = []
        seen: set[str] = set()
        for item in results:
            key = item.url or f"{item.title}|{item.snippet}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _effective_score(self, *, item: WebSearchResult, query: str) -> float:
        if item.score > 0:
            return _clamp(float(item.score))
        query_terms = set(re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", query.lower()))
        text = f"{item.title} {item.snippet}".lower()
        if not query_terms:
            return 0.0
        hit = sum(1 for term in query_terms if term in text)
        base = hit / max(len(query_terms), 1)
        title_bonus = 0.1 if any(term in item.title.lower() for term in query_terms) else 0.0
        return _clamp(base + title_bonus)

    def _freshness_ratio(self, *, ranked: list[WebSearchResult], freshness_window_days: int) -> float:
        if not ranked:
            return 0.0
        threshold = date.today() - timedelta(days=max(1, freshness_window_days))
        fresh = 0
        for item in ranked:
            published = self._parse_date(item.published_at)
            if published is not None and published >= threshold:
                fresh += 1
        return fresh / max(len(ranked), 1)

    def _parse_date(self, value: str) -> date | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except Exception:
            try:
                return date.fromisoformat(text[:10])
            except Exception:
                return None

    def _noise_ratio(self, ranked: list[WebSearchResult]) -> float:
        if not ranked:
            return 0.0
        noisy = 0
        for item in ranked:
            snippet = str(item.snippet or "").strip()
            title = str(item.title or "").strip()
            url = str(item.url or "").lower()
            too_short = len(snippet) < 24
            low_info = len(set(re.findall(r"[a-z0-9\u4e00-\u9fff]", snippet.lower()))) < 10
            ad_like = any(marker in url for marker in self._LOW_QUALITY_URL_MARKERS)
            title_noise = title.lower().startswith(("广告", "推广", "sponsored"))
            if too_short or (low_info and ad_like) or title_noise:
                noisy += 1
        return noisy / max(len(ranked), 1)

    def _conflict_detected(self, ranked: list[WebSearchResult]) -> bool:
        if len(ranked) <= 1:
            return False
        text = " ".join(f"{row.title} {row.snippet}".lower() for row in ranked)
        has_restrict = any(marker in text for marker in self._RESTRICT_MARKERS)
        has_relax = any(marker in text for marker in self._RELAX_MARKERS)
        return has_restrict and has_relax
