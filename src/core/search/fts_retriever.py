from __future__ import annotations

import sqlite3
from typing import Any


class FTSRetriever:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def retrieve(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        try:
            with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                    )
                }
                has_chunks = "chunks" in tables
                fts_columns = {row[1] for row in conn.execute("PRAGMA table_info(fts_index)")}
                file_columns = (
                    {row[1] for row in conn.execute("PRAGMA table_info(files)")}
                    if "files" in tables
                    else set()
                )
                joins = []
                source_expr = "fts_index.source" if "source" in fts_columns else "NULL"
                source_path_expr = "NULL"
                section_expr = "fts_index.section_title" if "section_title" in fts_columns else "NULL"
                doc_type_expr = "NULL"
                if has_chunks:
                    joins.append(
                        """
                        LEFT JOIN chunks
                            ON chunks.file_uuid = fts_index.file_uuid
                           AND chunks.chunk_id = fts_index.chunk_id
                        """
                    )
                    source_expr = f"COALESCE(chunks.source_filename, {source_expr})"
                    source_path_expr = "chunks.source_path"
                    section_expr = f"COALESCE(chunks.section_title, {section_expr})"
                    doc_type_expr = "chunks.doc_type"
                if "files" in tables:
                    joins.append("LEFT JOIN files ON files.uuid = fts_index.file_uuid")
                    if "filename" in file_columns:
                        source_expr = f"COALESCE({source_expr}, files.filename, '')"
                    if "filepath" in file_columns:
                        source_path_expr = f"COALESCE({source_path_expr}, files.filepath, '')"
                    if "doc_type" in file_columns:
                        doc_type_expr = f"COALESCE({doc_type_expr}, files.doc_type, 'text')"
                if "filename" not in file_columns:
                    source_expr = f"COALESCE({source_expr}, '')"
                if "filepath" not in file_columns:
                    source_path_expr = f"COALESCE({source_path_expr}, '')"
                if "doc_type" not in file_columns:
                    doc_type_expr = f"COALESCE({doc_type_expr}, 'text')"
                section_expr = f"COALESCE({section_expr}, '')"

                sql = f"""
                    SELECT
                        fts_index.file_uuid AS file_uuid,
                        fts_index.chunk_id AS chunk_id,
                        {source_expr} AS source,
                        {source_path_expr} AS source_path,
                        {section_expr} AS section_title,
                        {doc_type_expr} AS doc_type,
                        fts_index.content AS content,
                        bm25(fts_index) AS fts_raw_score
                    FROM fts_index
                    {' '.join(joins)}
                    WHERE fts_index MATCH ?
                    ORDER BY bm25(fts_index)
                    LIMIT ?
                """
                rows = conn.execute(
                    sql,
                    (query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            results.append(
                {
                    "file_uuid": row["file_uuid"],
                    "chunk_id": int(row["chunk_id"]),
                    "source": row["source"],
                    "source_path": row["source_path"],
                    "section_title": row["section_title"],
                    "doc_type": row["doc_type"],
                    "content": row["content"],
                    "fts_rank": rank,
                    "fts_raw_score": float(row["fts_raw_score"]),
                }
            )
        return results
