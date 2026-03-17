from __future__ import annotations

import json
from typing import Any

from src.RAG.storage.sqlite_conn import connect
from src.RAG.storage.sqlite_schema import ensure_schema


class ConflictResolver:
    def __init__(self, db_path: str, ensure_db_schema: bool = True):
        self.db_path = db_path
        if ensure_db_schema:
            with connect(self.db_path) as conn:
                ensure_schema(conn)

    def report_conflict(
        self,
        topic: str,
        conflicting_sources: list[str],
        priority: str = "medium",
    ) -> None:
        payload = json.dumps(conflicting_sources, ensure_ascii=False)
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO conflicts (topic, conflicting_sources, priority)
                VALUES (?, ?, ?)
                """,
                (topic, payload, priority),
            )
            conn.commit()

    def detect_conflicts(self) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM conflicts
                WHERE resolution_status = 'detected'
                ORDER BY created_at DESC
                """
            ).fetchall()
            return [self._normalize_row(row) for row in rows]

    def resolve_conflict(self, conflict_id: int, resolution_note: str) -> bool:
        with connect(self.db_path) as conn:
            try:
                cursor = conn.execute(
                    """
                    UPDATE conflicts
                    SET resolution_status = 'resolved',
                        resolution_note = ?,
                        resolved_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (resolution_note, conflict_id),
                )
            except Exception:
                cursor = conn.execute(
                    """
                    UPDATE conflicts
                    SET resolution_status = 'resolved',
                        resolved_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (conflict_id,),
                )
            conn.commit()
            return cursor.rowcount > 0

    def get_conflict_stats(self) -> dict[str, int]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT resolution_status, COUNT(*) AS count
                FROM conflicts
                GROUP BY resolution_status
                """
            ).fetchall()
        stats = {"detected": 0, "resolved": 0}
        for row in rows:
            stats[str(row["resolution_status"])] = int(row["count"])
        return stats

    def _normalize_row(self, row: Any) -> dict[str, Any]:
        payload = dict(row)
        raw_sources = payload.get("conflicting_sources") or "[]"
        try:
            payload["conflicting_sources"] = json.loads(raw_sources)
        except json.JSONDecodeError:
            payload["conflicting_sources"] = []
        return payload
