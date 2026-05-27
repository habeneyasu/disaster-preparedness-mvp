"""SQLite data access for query log persistence."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.core.config import settings

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    district TEXT NOT NULL,
    hazard_type TEXT,
    raw_report TEXT NOT NULL,
    summary TEXT NOT NULL,
    predicted_risk TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    map_path TEXT NOT NULL
);
"""


def _resolve_db_path() -> Path:
    """Resolve the SQLite file path from DATABASE_URL or DB_PATH."""
    url = settings.DATABASE_URL
    if url.startswith("sqlite:///"):
        return Path(url.removeprefix("sqlite:///"))
    return settings.DB_PATH


@contextmanager
def get_db_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row dict access; commits on success."""
    settings.ensure_data_dir()
    conn = sqlite3.connect(str(_resolve_db_path()))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the query_log table and enable WAL mode for concurrent reads."""
    with get_db_connection() as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute("PRAGMA journal_mode=WAL")


def insert_query_record(
    district: str,
    hazard_type: str | None,
    raw_report: str,
    summary: str,
    predicted_risk: str,
    confidence_score: float,
    map_path: str,
) -> int:
    """Insert a query log row and return its primary key."""
    timestamp = datetime.now(timezone.utc).isoformat()
    insert_sql = """
        INSERT INTO query_log (
            timestamp, district, hazard_type, raw_report,
            summary, predicted_risk, confidence_score, map_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_db_connection() as conn:
        cursor = conn.execute(
            insert_sql,
            (
                timestamp,
                district,
                hazard_type,
                raw_report,
                summary,
                predicted_risk,
                confidence_score,
                map_path,
            ),
        )
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("Insert into query_log did not return a row id")
        return int(row_id)


def fetch_query_history(limit: int | None = None) -> list[dict[str, Any]]:
    """Return the most recent query log entries, newest first."""
    if limit is None:
        limit = settings.HISTORY_DEFAULT_LIMIT
    safe_limit = max(1, min(limit, settings.HISTORY_MAX_LIMIT))

    select_sql = """
        SELECT id, timestamp, district, hazard_type, raw_report,
               summary, predicted_risk, confidence_score, map_path
        FROM query_log
        ORDER BY id DESC
        LIMIT ?
    """
    with get_db_connection() as conn:
        rows = conn.execute(select_sql, (safe_limit,)).fetchall()
        return [dict(row) for row in rows]
