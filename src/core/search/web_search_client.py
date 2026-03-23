from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from urllib.parse import urlparse


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0
    source_domain: str = ""
    published_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "WebSearchResult":
        url = str(payload.get("url", "")).strip()
        domain = str(payload.get("source_domain", "")).strip() or _domain_from_url(url)
        published_at = _normalize_published_at(payload.get("published_at"))
        return cls(
            title=str(payload.get("title", "")).strip(),
            url=url,
            snippet=str(payload.get("snippet", "")).strip(),
            score=float(payload.get("score", 0.0) or 0.0),
            source_domain=domain,
            published_at=published_at,
            metadata=dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata", {}), dict)
            else {},
        )


def _domain_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return (parsed.netloc or "").lower().strip()


def _normalize_published_at(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        return text[:10]


class WebSearchClient:
    def __init__(
        self,
        *,
        provider: str = "mock",
        timeout: int = 8,
        search_impl: Callable[[str, int], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.provider = str(provider or "mock").strip().lower()
        self.timeout = max(1, int(timeout))
        self._search_impl = search_impl

    def search(self, query: str, *, limit: int = 12) -> list[WebSearchResult]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []
        safe_limit = max(1, int(limit))
        if self._search_impl is not None:
            payloads = self._search_impl(normalized_query, safe_limit)
            return self._coerce_results(payloads, safe_limit)
        if self.provider in {"", "none", "disabled", "off", "mock"}:
            return []
        # Provider hooks are intentionally conservative in this phase:
        # keep default behavior deterministic and compatible, then wire real provider separately.
        return []

    def _coerce_results(self, payloads: list[dict[str, Any]], limit: int) -> list[WebSearchResult]:
        rows: list[WebSearchResult] = []
        seen: set[str] = set()
        for payload in payloads:
            result = WebSearchResult.from_payload(payload)
            dedup_key = result.url or f"{result.title}|{result.snippet}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            if not result.source_domain:
                result.source_domain = _domain_from_url(result.url)
            rows.append(result)
            if len(rows) >= limit:
                break
        return rows
