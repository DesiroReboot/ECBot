from __future__ import annotations

import hashlib
import sqlite3
from typing import Any

from src.RAG.config.kbase_config import KBaseConfig
from src.RAG.reader.chunker import Chunker
from src.RAG.reader.embedding_client import EmbeddingClient
from src.RAG.storage.sqlite_conn import connect, transaction
from src.RAG.storage.sqlite_schema import ensure_schema


class Indexer:
    def __init__(self, db_path: str, config: KBaseConfig):
        self.db_path = db_path
        self.config = config
        self.chunker = Chunker(config.chunk_size, config.chunk_overlap)
        self.embedding = EmbeddingClient(config)
        with connect(self.db_path) as conn:
            ensure_schema(conn)

    def _chunk_content(self, content: str) -> list[str]:
        return self.chunker.split(content)

    def index_document(
        self,
        file_uuid: str,
        content: str,
        *,
        source: str = "",
        source_path: str = "",
        section_title: str = "",
        doc_type: str = "text",
    ) -> dict[str, Any]:
        chunks = self._chunk_content(content)
        embeddings: list[list[float]] | None = None
        embedding_error: str | None = None
        if chunks:
            try:
                embeddings = self.embedding.embed_texts(chunks)
            except Exception as exc:
                embeddings = None
                embedding_error = str(exc)

        with transaction(self.db_path) as conn:
            self._delete_document_index_tx(conn, file_uuid)
            for chunk_id, chunk_text in enumerate(chunks):
                content_hash = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()
                conn.execute(
                    """
                    INSERT INTO chunks (
                        file_uuid, chunk_id, source_filename, source_path, section_title, doc_type,
                        content, content_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_uuid,
                        chunk_id,
                        source,
                        source_path,
                        section_title,
                        doc_type,
                        chunk_text,
                        content_hash,
                    ),
                )
                try:
                    conn.execute(
                        """
                        INSERT INTO fts_index (content, source, section_title, file_uuid, chunk_id)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (chunk_text, source, section_title, file_uuid, chunk_id),
                    )
                except sqlite3.OperationalError:
                    conn.execute(
                        """
                        INSERT INTO fts_index (content, file_uuid, chunk_id)
                        VALUES (?, ?, ?)
                        """,
                        (chunk_text, file_uuid, chunk_id),
                    )
            if embeddings is not None:
                for chunk_id, vec in enumerate(embeddings):
                    try:
                        conn.execute(
                            """
                            INSERT INTO vec_index (file_uuid, chunk_id, embedding, source, source_path)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                file_uuid,
                                chunk_id,
                                self.embedding.serialize(vec),
                                source,
                                source_path,
                            ),
                        )
                    except sqlite3.OperationalError:
                        conn.execute(
                            """
                            INSERT INTO vec_index (file_uuid, chunk_id, embedding)
                            VALUES (?, ?, ?)
                            """,
                            (file_uuid, chunk_id, self.embedding.serialize(vec)),
                        )

        return {
            "chunks_written": len(chunks),
            "vectors_written": len(chunks) if embeddings is not None else 0,
            "index_status": "ready" if embeddings is not None else "partial",
            "last_error": embedding_error,
        }

    def delete_document_index(self, file_uuid: str) -> None:
        with transaction(self.db_path) as conn:
            self._delete_document_index_tx(conn, file_uuid)

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        with connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            fts_columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(fts_index)").fetchall()
            }
            has_source = "source" in fts_columns
            source_expr = "fts_index.source" if has_source else "COALESCE(files.filename, '')"
            rows = conn.execute(
                """
                SELECT
                    fts_index.file_uuid AS file_uuid,
                    fts_index.chunk_id AS chunk_id,
                    """
                + source_expr
                + """
                    AS source,
                    COALESCE(files.filepath, '') AS source_path,
                    fts_index.content AS content,
                    bm25(fts_index) AS fts_raw_score
                FROM fts_index
                LEFT JOIN files ON files.uuid = fts_index.file_uuid
                WHERE fts_index MATCH ?
                ORDER BY bm25(fts_index)
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()

        results = []
        for rank, row in enumerate(rows, start=1):
            results.append(
                {
                    "file_uuid": row["file_uuid"],
                    "chunk_id": int(row["chunk_id"]),
                    "source": row["source"],
                    "source_path": row["source_path"],
                    "content": row["content"],
                    "fts_rank": rank,
                    "fts_raw_score": float(row["fts_raw_score"]),
                }
            )
        return results

    def get_index_stats(self) -> dict[str, int]:
        with connect(self.db_path) as conn:
            files = self._safe_count(conn, "files")
            chunks = self._safe_count(conn, "chunks")
            fts_docs = self._safe_count(conn, "fts_index")
            vec_rows = self._safe_count(conn, "vec_index")
        return {
            "indexed_files": files,
            "indexed_chunks": chunks,
            "fts_documents": fts_docs,
            "vec_rows": vec_rows,
        }

    def _delete_document_index_tx(self, conn: sqlite3.Connection, file_uuid: str) -> None:
        conn.execute("DELETE FROM fts_index WHERE file_uuid = ?", (file_uuid,))
        conn.execute("DELETE FROM vec_index WHERE file_uuid = ?", (file_uuid,))
        try:
            conn.execute("DELETE FROM chunks WHERE file_uuid = ?", (file_uuid,))
        except sqlite3.OperationalError:
            pass

    def _safe_count(self, conn: sqlite3.Connection, table_name: str) -> int:
        try:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        except sqlite3.OperationalError:
            return 0
