from __future__ import annotations

from src.core.search.query_preprocessor import QueryPreprocessor


def test_extract_keywords_returns_deduped_non_numeric_tokens() -> None:
    pre = QueryPreprocessor()
    keywords = pre.extract_keywords("最近 2026 跨境电商 平台政策 最新 变化", top_k=4)
    assert len(keywords) <= 4
    assert "2026" not in keywords
    assert len(set(keywords)) == len(keywords)


def test_extract_keywords_falls_back_to_available_tokens() -> None:
    pre = QueryPreprocessor()
    keywords = pre.extract_keywords("hello world", top_k=3)
    assert keywords


def test_query_intent_preserves_entities_and_progress_keywords() -> None:
    pre = QueryPreprocessor()
    query = "在最新的平台政策下，如何打造类似 哭哭马 拉布布 的爆款"
    processed = pre.process(query)
    intent = processed["query_intent"]

    assert "哭哭马" in intent["core_entities"]
    assert "拉布布" in intent["core_entities"]
    assert "最新" in intent["temporal_terms"]
    assert intent["need_web_search"] is True
    assert intent["route_mode"] in {"hybrid", "web_dominant"}

    progress_keywords = pre.extract_progress_keywords(query, top_k=4)
    assert "哭哭马" in progress_keywords
    assert "拉布布" in progress_keywords
    assert "在最" not in progress_keywords
