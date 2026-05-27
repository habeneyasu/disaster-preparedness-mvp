"""SQLite persistence for query_log."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.core.config import settings

_SCHEMA = """
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

_INSERT = """
INSERT INTO query_log (
    timestamp, district, hazard_type, raw_report,
    summary, predicted_risk, confidence_score, map_path
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT = """
SELECT id, timestamp, district, hazard_type, raw_report,
       summary, predicted_risk, confidence_score, map_path
FROM query_log ORDER BY id DESC LIMIT ?
"""


def _db_path() -> Path:
    url = settings.DATABASE_URL
    if url.startswith("sqlite:///"):
        return Path(url.removeprefix("sqlite:///"))
    return settings.DB_PATH


@contextmanager
def get_db_connection() -> Iterator[sqlite3.Connection]:
    settings.ensure_data_dir()
    conn = sqlite3.connect(str(_db_path()))
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
    with get_db_connection() as conn:
        conn.execute(_SCHEMA)
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
    with get_db_connection() as conn:
        cur = conn.execute(
            _INSERT,
            (
                datetime.now(timezone.utc).isoformat(),
                district,
                hazard_type,
                raw_report,
                summary,
                predicted_risk,
                confidence_score,
                map_path,
            ),
        )
        if cur.lastrowid is None:
            raise RuntimeError("Insert into query_log did not return a row id")
        return int(cur.lastrowid)


def fetch_query_history(limit: int | None = None) -> list[dict[str, Any]]:
    limit = settings.HISTORY_DEFAULT_LIMIT if limit is None else limit
    safe = max(1, min(limit, settings.HISTORY_MAX_LIMIT))
    with get_db_connection() as conn:
        return [dict(r) for r in conn.execute(_SELECT, (safe,)).fetchall()]
