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
