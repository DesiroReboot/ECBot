from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from src.core.search.rag_search import RAGSearcher


def test_search_degrades_to_fts_when_vector_branch_fails() -> None:
    db_path = Path.cwd() / f"ragv2-search-{uuid4().hex}.db"
    try:
        searcher = RAGSearcher(
            db_path=str(db_path),
            top_k=3,
            fts_top_k=3,
            vec_top_k=3,
            embedding_provider="local",
            embedding_base_url="",
            embedding_api_key="",
        )

        fts_results = [
            {
                "file_uuid": "doc-1",
                "chunk_id": 0,
                "source": "trade-guide.md",
                "source_path": "/kb/trade-guide.md",
                "section_title": "shipping",
                "doc_type": "text",
                "content": "Amazon logistics and shipping timeline details.",
                "fts_rank": 1,
                "fts_raw_score": -0.5,
            }
        ]

        hybrid_meta = {
            "vector_meta": {"candidate_pool": 0, "error": "vector timeout"},
            "branch_errors": {"vec": "vector timeout"},
        }

        with patch.object(searcher.hybrid, "retrieve", return_value=(fts_results, [], hybrid_meta)):
            results, trace = searcher.search_with_trace("amazon shipping")

        assert len(results) == 1
        assert results[0].file_uuid == "doc-1"
        assert [path["source"] for path in results[0].retrieval_paths] == ["fts"]

        assert trace["vector_recall"] == []
        assert trace["generation"]["branch_errors"]["vec"] == "vector timeout"
        assert any(msg.startswith("vec:") for msg in trace.get("errors", []))
    finally:
        if db_path.exists():
            db_path.unlink()
