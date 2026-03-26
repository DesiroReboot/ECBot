from __future__ import annotations

import pytest

from src.core.search.web_search_client import WebSearchClient


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self) -> dict:
        return dict(self._payload)


class _FakeSession:
    def __init__(self, payload: dict):
        self.payload = payload
        self.last_request: dict = {}

    def post(self, url: str, *, json: dict, headers: dict, timeout: int) -> _FakeResponse:
        self.last_request = {
            "url": url,
            "json": dict(json),
            "headers": dict(headers),
            "timeout": timeout,
        }
        return _FakeResponse(status_code=200, payload=self.payload)


def test_build_tavily_request_clamps_limit_and_depth() -> None:
    client = WebSearchClient(
        provider="tavily",
        tavily_api_key="test-key",
        tavily_search_depth="advanced",
        max_results=8,
    )
    payload = client._build_tavily_request("latest amazon policy", 99)

    assert payload["query"] == "latest amazon policy"
    assert payload["max_results"] == 8
    assert payload["search_depth"] == "advanced"
    assert payload["include_answer"] is False


def test_normalize_tavily_item_handles_missing_fields() -> None:
    client = WebSearchClient(provider="tavily", tavily_api_key="test-key")
    normalized = client._normalize_tavily_item(
        {
            "title": "Policy update",
            "url": "https://example.com/news",
            "content": "A platform policy changed this week.",
            "score": "0.81",
            "published_date": "2026-03-20T10:00:00Z",
        }
    )

    assert normalized["title"] == "Policy update"
    assert normalized["snippet"] == "A platform policy changed this week."
    assert normalized["score"] == pytest.approx(0.81)
    assert normalized["source_domain"] == "example.com"
    assert normalized["published_at"] == "2026-03-20"


def test_search_tavily_dedupes_and_respects_limit() -> None:
    session = _FakeSession(
        payload={
            "results": [
                {
                    "title": "A",
                    "url": "https://example.com/a",
                    "content": "content-a",
                    "score": 0.9,
                },
                {
                    "title": "A-duplicate",
                    "url": "https://example.com/a",
                    "content": "content-a-dup",
                    "score": 0.7,
                },
                {
                    "title": "B",
                    "url": "https://example.com/b",
                    "content": "content-b",
                    "score": 0.6,
                },
            ]
        }
    )
    client = WebSearchClient(
        provider="tavily",
        tavily_api_key="test-key",
        session=session,
        max_results=10,
    )

    rows = client.search("query", limit=2)

    assert len(rows) == 2
    assert rows[0].url == "https://example.com/a"
    assert rows[1].url == "https://example.com/b"
    assert session.last_request["json"]["max_results"] == 2


def test_search_tavily_raises_for_missing_api_key() -> None:
    client = WebSearchClient(provider="tavily", tavily_api_key="")

    with pytest.raises(RuntimeError, match="provider_misconfigured"):
        client.search("latest trend")
