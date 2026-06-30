"""Append-only audit log backed by SQLite.

Every attribution decision and every appeal becomes a row. The log is
**append-only**: an appeal is written as a *new* row sharing the original's
``content_id`` (event_type='appeal', status='under_review') rather than mutating
the original classification row. This preserves the full decision history, which
is exactly what a human reviewer needs when they open the appeal queue.

The "current status" of a piece of content is therefore the status on its most
recent row (see ``latest_for_content``). Each public function takes the DB path
explicitly so tests can point at a temp database. (planning.md "Audit log".)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type        TEXT NOT NULL,          -- 'classification' | 'appeal'
    content_id        TEXT NOT NULL,
    creator_id        TEXT,
    timestamp         TEXT NOT NULL,          -- ISO-8601 UTC
    attribution       TEXT,                   -- likely_human | uncertain | likely_ai
    confidence        REAL,                   -- combined ai_likelihood
    llm_score         REAL,                   -- signal 1
    stylometric_score REAL,                   -- signal 2
    status            TEXT NOT NULL,          -- 'classified' | 'under_review'
    appeal_reasoning  TEXT,
    text_snippet      TEXT                    -- first chars of the submission
);
CREATE INDEX IF NOT EXISTS idx_audit_content ON audit_log(content_id);
"""

_SNIPPET_LEN = 280


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: str) -> None:
    """Create the audit table if it does not exist (idempotent)."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def log_classification(
    db_path: str,
    *,
    content_id: str,
    creator_id: Optional[str],
    attribution: str,
    confidence: float,
    llm_score: Optional[float],
    stylometric_score: Optional[float],
    text_snippet: Optional[str] = None,
    status: str = "classified",
) -> None:
    """Append a classification decision."""
    snippet = (text_snippet or "")[:_SNIPPET_LEN] or None
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit_log (
                event_type, content_id, creator_id, timestamp, attribution,
                confidence, llm_score, stylometric_score, status,
                appeal_reasoning, text_snippet
            ) VALUES ('classification', ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                content_id,
                creator_id,
                _now(),
                attribution,
                confidence,
                llm_score,
                stylometric_score,
                status,
                snippet,
            ),
        )


def log_appeal(
    db_path: str,
    *,
    content_id: str,
    creator_id: Optional[str],
    appeal_reasoning: str,
    original: dict[str, Any],
) -> None:
    """Append an appeal as a new row sharing ``content_id``.

    The original decision's scores are copied onto the appeal row so a reviewer
    sees the appeal *alongside* what was decided, in one record.
    """
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit_log (
                event_type, content_id, creator_id, timestamp, attribution,
                confidence, llm_score, stylometric_score, status,
                appeal_reasoning, text_snippet
            ) VALUES ('appeal', ?, ?, ?, ?, ?, ?, ?, 'under_review', ?, ?)
            """,
            (
                content_id,
                creator_id,
                _now(),
                original.get("attribution"),
                original.get("confidence"),
                original.get("llm_score"),
                original.get("stylometric_score"),
                appeal_reasoning,
                original.get("text_snippet"),
            ),
        )


def get_recent(db_path: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent entries (newest first) as plain dicts."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_all(db_path: str) -> list[dict[str, Any]]:
    """Return every entry (oldest first) — used by analytics aggregation."""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
    return [dict(row) for row in rows]


def latest_for_content(db_path: str, content_id: str) -> Optional[dict[str, Any]]:
    """Most recent row for a content_id, or None if it was never classified.

    Used by /appeal to confirm the content exists and to pull the original
    decision's scores for the appeal record.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM audit_log WHERE content_id = ? ORDER BY id DESC LIMIT 1",
            (content_id,),
        ).fetchone()
    return dict(row) if row else None
