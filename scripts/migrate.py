from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.RAG.storage.sqlite_conn import connect  # noqa: E402
from src.RAG.storage.sqlite_schema import ensure_schema  # noqa: E402


TABLES = ["files", "chunks", "fts_index", "vec_index", "index_manifest", "conflicts"]


def _status(db_path: str) -> dict[str, Any]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        existing = {str(row[0]) for row in rows}
        details: dict[str, Any] = {}
        for table in TABLES:
            has_table = table in existing
            details[table] = {"exists": has_table, "rows": 0}
            if has_table:
                try:
                    details[table]["rows"] = int(
                        conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    )
                except sqlite3.OperationalError:
                    details[table]["rows"] = 0
        return {
            "db_path": str(Path(db_path).resolve()),
            "tables": details,
            "ready": all(details[name]["exists"] for name in TABLES),
        }


def _upgrade(db_path: str) -> dict[str, Any]:
    with connect(db_path) as conn:
        ensure_schema(conn)
    return _status(db_path)


def _downgrade(db_path: str) -> dict[str, Any]:
    drop_sql = [
        "DROP TABLE IF EXISTS index_manifest",
        "DROP TABLE IF EXISTS conflicts",
        "DROP TABLE IF EXISTS vec_index",
        "DROP TABLE IF EXISTS fts_index",
        "DROP TABLE IF EXISTS chunks",
        "DROP TABLE IF EXISTS files",
    ]
    with connect(db_path) as conn:
        for sql in drop_sql:
            conn.execute(sql)
        conn.commit()
    return _status(db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple migration entrypoint for RAG SQLite schema")
    parser.add_argument("--db", default="DB/ec_bot.db", help="SQLite database path")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show migration status")
    subparsers.add_parser("upgrade", help="Apply latest schema")

    downgrade_parser = subparsers.add_parser("downgrade", help="Drop managed schema objects")
    downgrade_parser.add_argument(
        "--yes",
        action="store_true",
        help="Acknowledge destructive downgrade operation",
    )

    args = parser.parse_args()
    if args.command == "status":
        result = _status(args.db)
    elif args.command == "upgrade":
        result = _upgrade(args.db)
    else:
        if not args.yes:
            raise SystemExit("downgrade is destructive; re-run with --yes")
        result = _downgrade(args.db)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
