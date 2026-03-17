from __future__ import annotations

import sqlite3
from typing import Any

from src.RAG.storage.sqlite_conn import connect
from src.RAG.storage.sqlite_schema import ensure_schema


class FileMapper:
    def __init__(self, db_path: str, ensure_db_schema: bool = True):
        self.db_path = db_path
        if ensure_db_schema:
            with connect(self.db_path) as conn:
                ensure_schema(conn)

    def save_file(
        self,
        uuid: str,
        filename: str,
        filepath: str,
        category: str | None,
        summary: str,
        file_hash: str,
        file_size: int,
        *,
        doc_type: str = "text",
        parse_status: str = "ready",
        index_status: str = "ready",
        last_error: str | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO files (
                        uuid, filename, filepath, category, summary, file_hash, file_size,
                        doc_type, parse_status, index_status, last_error, last_scanned_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(filepath) DO UPDATE SET
                        uuid=excluded.uuid,
                        filename=excluded.filename,
                        category=excluded.category,
                        summary=excluded.summary,
                        file_hash=excluded.file_hash,
                        file_size=excluded.file_size,
                        doc_type=excluded.doc_type,
                        parse_status=excluded.parse_status,
                        index_status=excluded.index_status,
                        last_error=excluded.last_error,
                        last_scanned_at=CURRENT_TIMESTAMP,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        uuid,
                        filename,
                        filepath,
                        category or "uncategorized",
                        summary,
                        file_hash,
                        file_size,
                        doc_type,
                        parse_status,
                        index_status,
                        last_error,
                    ),
                )
            except sqlite3.OperationalError:
                # Compatibility fallback for legacy minimal files table.
                conn.execute(
                    """
                    INSERT INTO files (
                        uuid, filename, filepath, category, summary, file_hash, file_size, updated_at, last_scanned_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(uuid) DO UPDATE SET
                        filename=excluded.filename,
                        filepath=excluded.filepath,
                        category=excluded.category,
                        summary=excluded.summary,
                        file_hash=excluded.file_hash,
                        file_size=excluded.file_size,
                        updated_at=CURRENT_TIMESTAMP,
                        last_scanned_at=CURRENT_TIMESTAMP
                    """,
                    (
                        uuid,
                        filename,
                        filepath,
                        category or "uncategorized",
                        summary,
                        file_hash,
                        file_size,
                    ),
                )
            conn.commit()

    def get_file(self, uuid: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM files WHERE uuid = ?", (uuid,)).fetchone()
            return dict(row) if row else None

    def get_file_by_path(self, filepath: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM files WHERE filepath = ?", (filepath,)).fetchone()
            return dict(row) if row else None

    def get_all_files(self) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM files ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def update_category(self, uuid: str, category: str) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE files SET category = ?, updated_at = CURRENT_TIMESTAMP WHERE uuid = ?",
                (category, uuid),
            )
            conn.commit()

    def delete_file(self, uuid: str) -> bool:
        with connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM files WHERE uuid = ?", (uuid,))
            conn.commit()
            return cursor.rowcount > 0

    def search_by_filename(self, keyword: str, limit: int = 20) -> list[dict[str, Any]]:
        pattern = f"%{keyword}%"
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM files
                WHERE filename LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (pattern, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def count_ready_chunks(self, uuid: str) -> int:
        with connect(self.db_path) as conn:
            try:
                row = conn.execute(
                    "SELECT COUNT(*) AS count FROM chunks WHERE file_uuid = ?",
                    (uuid,),
                ).fetchone()
                return int(row["count"]) if row else 0
            except sqlite3.OperationalError:
                return 0

    def update_index_status(self, uuid: str, index_status: str, last_error: str | None = None) -> None:
        with connect(self.db_path) as conn:
            try:
                conn.execute(
                    """
                    UPDATE files
                    SET index_status = ?, last_error = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE uuid = ?
                    """,
                    (index_status, last_error, uuid),
                )
            except sqlite3.OperationalError:
                # Legacy tables may not have status columns.
                conn.execute(
                    "UPDATE files SET updated_at = CURRENT_TIMESTAMP WHERE uuid = ?",
                    (uuid,),
                )
            conn.commit()

    def count_by_category(self) -> dict[str, int]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT category, COUNT(*) AS count
                FROM files
                GROUP BY category
                """
            ).fetchall()
            return {str(row["category"] or "uncategorized"): int(row["count"]) for row in rows}

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
