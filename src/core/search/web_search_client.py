from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from time import sleep
from typing import Any, Callable
from urllib import parse, request
from urllib.error import HTTPError
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
        max_retries: int = 1,
        tavily_api_key: str = "",
        tavily_base_url: str = "https://api.tavily.com/search",
        tavily_search_depth: str = "basic",
        max_results: int = 8,
        session: Any | None = None,
        search_impl: Callable[[str, int], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.provider = str(provider or "mock").strip().lower()
        self.timeout = max(1, int(timeout))
        self.max_retries = max(0, int(max_retries))
        self.tavily_api_key = str(tavily_api_key or "").strip()
        self.tavily_base_url = str(tavily_base_url or "https://api.tavily.com/search").strip()
        self.tavily_search_depth = str(tavily_search_depth or "basic").strip().lower()
        if self.tavily_search_depth not in {"basic", "advanced"}:
            self.tavily_search_depth = "basic"
        self.max_results = max(1, int(max_results))
        self._session = session
        self._search_impl = search_impl

    def search(self, query: str, *, limit: int = 12) -> list[WebSearchResult]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []
        safe_limit = min(max(1, int(limit)), self.max_results)
        if self._search_impl is not None:
            payloads = self._search_impl(normalized_query, safe_limit)
            return self._coerce_results(payloads, safe_limit)
        if self.provider in {"", "none", "disabled", "off", "mock"}:
            return []
        if self.provider == "tavily":
            payloads = self._search_tavily(normalized_query, safe_limit)
            return self._coerce_results(payloads, safe_limit)
        return []

    def _search_tavily(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not self.tavily_api_key or self.tavily_api_key.upper().startswith("YOUR_"):
            raise RuntimeError("provider_misconfigured:tavily_api_key_missing")
        url = self._safe_http_url(self.tavily_base_url)
        payload = self._build_tavily_request(query, limit)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.tavily_api_key}",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post_json(url=url, payload=payload, headers=headers)
                rows = response.get("results", [])
                if not isinstance(rows, list):
                    return []
                normalized: list[dict[str, Any]] = []
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    normalized.append(self._normalize_tavily_item(item))
                return normalized
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    sleep(min(0.8 * (attempt + 1), 2.0))
                    continue
                break
        raise RuntimeError(f"tavily_search_failed:{last_error}")

    def _build_tavily_request(self, query: str, limit: int) -> dict[str, Any]:
        safe_limit = min(max(1, int(limit)), self.max_results)
        return {
            "query": str(query).strip(),
            "max_results": safe_limit,
            "search_depth": self.tavily_search_depth,
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
        }

    def _normalize_tavily_item(self, item: dict[str, Any]) -> dict[str, Any]:
        url = str(item.get("url", "")).strip()
        snippet = self._extract_tavily_snippet(item)
        title = str(item.get("title", "")).strip()
        score_raw = item.get("score", 0.0)
        try:
            score = float(score_raw or 0.0)
        except Exception:
            score = 0.0
        published_at = _normalize_published_at(
            item.get("published_date")
            or item.get("published_at")
            or item.get("date")
        )
        metadata = {
            "provider": "tavily",
            "raw_content": str(item.get("raw_content", "")).strip(),
            "favicon": str(item.get("favicon", "")).strip(),
        }
        metadata = {key: value for key, value in metadata.items() if value}
        return {
            "title": title,
            "url": url,
            "snippet": snippet,
            "score": score,
            "source_domain": _domain_from_url(url),
            "published_at": published_at,
            "metadata": metadata,
        }

    def _extract_tavily_snippet(self, item: dict[str, Any]) -> str:
        for key in ("content", "snippet", "description", "raw_content"):
            value = str(item.get(key, "")).strip()
            if value:
                return value
        return ""

    def _post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        session = self._session
        if session is not None and hasattr(session, "post"):
            response = session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            status_code = int(getattr(response, "status_code", 200))
            if status_code >= 400:
                text = str(getattr(response, "text", "")).strip()
                raise RuntimeError(f"http_{status_code}:{text[:200]}")
            json_method = getattr(response, "json", None)
            if callable(json_method):
                loaded = json_method()
                return loaded if isinstance(loaded, dict) else {}
            body = str(getattr(response, "text", "")).strip()
            if not body:
                return {}
            loaded = json.loads(body)
            return loaded if isinstance(loaded, dict) else {}

        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:  # nosec B310
                body = response.read().decode("utf-8", errors="ignore")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"http_{exc.code}:{body[:200]}") from exc
        if not body.strip():
            return {}
        loaded = json.loads(body)
        return loaded if isinstance(loaded, dict) else {}

    def _safe_http_url(self, raw_url: str) -> str:
        parsed = parse.urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"unsupported url scheme: {parsed.scheme!r}")
        if not parsed.netloc:
            raise ValueError("missing url host")
        return raw_url

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
