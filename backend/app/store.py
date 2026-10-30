from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


class AnalysisStore:
    """SQLite store with a per-thread persistent connection.

    Opening a new connection per request is measurably slow under load.
    Using thread-local storage satisfies SQLite's thread-safety requirement
    while avoiding repeated open/close overhead on every API call.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")   # safe with WAL; faster writes
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA cache_size=-8000")     # 8 MB page cache per connection
        return connection

    def _get_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._get_connection()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS comparisons (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    previous_id TEXT NOT NULL REFERENCES analyses(id),
                    revised_id TEXT NOT NULL REFERENCES analyses(id),
                    response_json TEXT NOT NULL
                )"""
            )

    def save_analysis(self, response: dict[str, Any]) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO analyses(id, created_at, response_json) VALUES (?, ?, ?)",
                (response["id"], created_at, json.dumps(response, separators=(",", ":"))),
            )

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT response_json FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
        return json.loads(row["response_json"]) if row else None

    def save_comparison(self, response: dict[str, Any]) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO comparisons(id, created_at, previous_id, revised_id, response_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    response["revised_id"],
                    datetime.now(timezone.utc).isoformat(),
                    response["previous_id"],
                    response["revised_id"],
                    json.dumps(response, separators=(",", ":")),
                ),
            )

    def purge_expired(self, retention_days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id FROM analyses WHERE created_at < ?", (cutoff,)
            ).fetchall()
            expired_ids = [row["id"] for row in rows]
            if expired_ids:
                connection.executemany(
                    "DELETE FROM comparisons WHERE previous_id = ? OR revised_id = ?",
                    [(analysis_id, analysis_id) for analysis_id in expired_ids],
                )
                connection.executemany(
                    "DELETE FROM analyses WHERE id = ?",
                    [(analysis_id,) for analysis_id in expired_ids],
                )
            return len(expired_ids)
