from __future__ import annotations

import re
import sqlite3
from typing import Any


class FTSRetriever:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def retrieve(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        match_query = self._build_match_query(query)
        if not match_query:
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
                doc_chunk_count_expr = "1"
                if has_chunks:
                    joins.append(
                        """
                        LEFT JOIN chunks
                            ON chunks.file_uuid = fts_index.file_uuid
                           AND chunks.chunk_id = fts_index.chunk_id
                        """
                    )
                    joins.append(
                        """
                        LEFT JOIN (
                            SELECT file_uuid, COUNT(*) AS doc_chunk_count
                            FROM chunks
                            GROUP BY file_uuid
                        ) chunk_stats ON chunk_stats.file_uuid = fts_index.file_uuid
                        """
                    )
                    source_expr = f"COALESCE(chunks.source_filename, {source_expr})"
                    source_path_expr = "chunks.source_path"
                    section_expr = f"COALESCE(chunks.section_title, {section_expr})"
                    doc_type_expr = "chunks.doc_type"
                    doc_chunk_count_expr = "COALESCE(chunk_stats.doc_chunk_count, 1)"
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
                        {doc_chunk_count_expr} AS doc_chunk_count,
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
                    (match_query, limit),
                ).fetchall()

                if not rows:
                    like_terms = self._build_like_terms(match_query)
                    if like_terms:
                        like_score_expr = " + ".join(
                            "(CASE WHEN LOWER(fts_index.content) LIKE ? THEN 1 ELSE 0 END)"
                            for _ in like_terms
                        )
                        like_sql = f"""
                            SELECT
                                fts_index.file_uuid AS file_uuid,
                                fts_index.chunk_id AS chunk_id,
                                {source_expr} AS source,
                                {source_path_expr} AS source_path,
                                {section_expr} AS section_title,
                                {doc_type_expr} AS doc_type,
                                {doc_chunk_count_expr} AS doc_chunk_count,
                                fts_index.content AS content,
                                ({like_score_expr}) AS fts_raw_score
                            FROM fts_index
                            {' '.join(joins)}
                            WHERE ({like_score_expr}) > 0
                            ORDER BY fts_raw_score DESC
                            LIMIT ?
                        """
                        like_params = [f"%{term}%" for term in like_terms]
                        rows = conn.execute(
                            like_sql,
                            like_params + like_params + [limit],
                        ).fetchall()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f"fts query failed: query={query!r}, match_query={match_query!r}, error={exc}"
            ) from exc

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
                    "doc_chunk_count": int(row["doc_chunk_count"]),
                    "content": row["content"],
                    "fts_rank": rank,
                    "fts_raw_score": float(row["fts_raw_score"]),
                }
            )
        return results

    def _build_match_query(self, query: str) -> str:
        normalized = re.sub(r"\s+", " ", str(query)).strip().lower()
        # Use explicit tokens to avoid FTS syntax errors from punctuation in raw user queries.
        ascii_tokens = re.findall(r"[a-z0-9_]{2,}", normalized)
        cjk_spans = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)

        tokens: list[str] = list(ascii_tokens)
        for span in cjk_spans:
            if len(span) <= 4:
                tokens.append(span)
            for idx in range(0, len(span) - 1):
                tokens.append(span[idx : idx + 2])

        if tokens:
            deduped: list[str] = []
            seen: set[str] = set()
            for token in tokens:
                if token in seen:
                    continue
                seen.add(token)
                deduped.append(token)
            return " ".join(deduped[:32])

        fallback = re.sub(r"[^a-z0-9_\u4e00-\u9fff]+", " ", normalized)
        fallback = re.sub(r"\s+", " ", fallback).strip()
        return fallback

    def _build_like_terms(self, match_query: str) -> list[str]:
        terms = [token.strip() for token in str(match_query).split() if token.strip()]
        filtered = [term for term in terms if len(term) >= 2]
        if not filtered:
            filtered = terms
        deduped: list[str] = []
        seen: set[str] = set()
        for term in filtered:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
        return deduped[:8]
