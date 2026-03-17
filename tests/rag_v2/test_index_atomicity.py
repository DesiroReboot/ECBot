from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from src.RAG.config.kbase_config import KBaseConfig
from src.RAG.indexing.indexer import Indexer


def _snapshot_rows(db_path: Path, file_uuid: str) -> dict[str, list[tuple]]:
    with sqlite3.connect(str(db_path)) as conn:
        chunks = conn.execute(
            """
            SELECT chunk_id, content, content_hash
            FROM chunks
            WHERE file_uuid = ?
            ORDER BY chunk_id
            """,
            (file_uuid,),
        ).fetchall()
        fts = conn.execute(
            """
            SELECT chunk_id, content
            FROM fts_index
            WHERE file_uuid = ?
            ORDER BY chunk_id
            """,
            (file_uuid,),
        ).fetchall()
        vec = conn.execute(
            """
            SELECT chunk_id, length(embedding)
            FROM vec_index
            WHERE file_uuid = ?
            ORDER BY chunk_id
            """,
            (file_uuid,),
        ).fetchall()
    return {"chunks": chunks, "fts": fts, "vec": vec}


def test_index_document_rolls_back_atomically_on_midway_failure() -> None:
    db_path = Path.cwd() / f"ragv2-atomic-{uuid4().hex}.db"
    try:
        config = KBaseConfig(
            db_path=str(db_path),
            chunk_size=64,
            chunk_overlap=0,
            embedding_provider="local",
            embedding_base_url="",
            embedding_api_key="",
        )
        indexer = Indexer(str(db_path), config)

        file_uuid = "doc-atomic-1"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                INSERT INTO files (uuid, filename, filepath, file_hash, file_size)
                VALUES (?, ?, ?, ?, ?)
                """,
                (file_uuid, "old.txt", "/tmp/old.txt", "hash-old", 123),
            )
            conn.commit()

        old_content = "alpha beta gamma " * 30
        indexer.index_document(
            file_uuid=file_uuid,
            content=old_content,
            source="old.txt",
            source_path="/tmp/old.txt",
        )
        baseline = _snapshot_rows(db_path, file_uuid)
        assert baseline["chunks"]
        assert baseline["fts"]
        assert baseline["vec"]

        new_content = "new content should not be persisted " * 30
        with patch.object(indexer.embedding, "serialize", side_effect=RuntimeError("forced-failure")):
            with pytest.raises(RuntimeError, match="forced-failure"):
                indexer.index_document(
                    file_uuid=file_uuid,
                    content=new_content,
                    source="new.txt",
                    source_path="/tmp/new.txt",
                )

        after = _snapshot_rows(db_path, file_uuid)
        assert after == baseline
    finally:
        try:
            if db_path.exists():
                db_path.unlink()
        except PermissionError:
            pass
