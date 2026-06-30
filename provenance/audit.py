"""Append-only audit log backed by SQLite.

Every attribution decision and every appeal becomes a row. The log is
**append-only**: an appeal is written as a *new* row sharing the original's
``content_id`` (event_type='appeal', status='under_review') rather than mutating
the original classification row. This preserves the full decision history, which
is exactly what a human reviewer needs when they open the appeal queue.

The "current status" of a piece of content is therefore the status on its most
recent row (see ``latest_for_content``). Each public function takes a ``Database``
(see ``provenance.db``) so connection handling lives in one place and multi-step
operations can run atomically. (planning.md "Audit log".)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from provenance.db import Database

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type        TEXT NOT NULL,          -- 'classification' | 'appeal'
    content_id        TEXT NOT NULL,
    content_type      TEXT NOT NULL DEFAULT 'text',  -- 'text' | 'metadata'
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

_INSERT = """
    INSERT INTO audit_log (
        event_type, content_id, content_type, creator_id, timestamp,
        attribution, confidence, llm_score, stylometric_score, status,
        appeal_reasoning, text_snippet
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db: Database) -> None:
    """Create the audit table if it does not exist (idempotent)."""
    with db.transaction() as conn:
        conn.executescript(_SCHEMA)


def log_classification(
    db: Database,
    *,
    content_id: str,
    creator_id: Optional[str],
    attribution: str,
    confidence: float,
    llm_score: Optional[float],
    stylometric_score: Optional[float],
    text_snippet: Optional[str] = None,
    status: str = "classified",
    content_type: str = "text",
) -> None:
    """Append a classification decision."""
    snippet = (text_snippet or "")[:_SNIPPET_LEN] or None
    with db.transaction() as conn:
        conn.execute(
            _INSERT,
            ("classification", content_id, content_type, creator_id, _now(),
             attribution, confidence, llm_score, stylometric_score, status, None, snippet),
        )


def _latest_row(conn, content_id: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM audit_log WHERE content_id = ? ORDER BY id DESC LIMIT 1",
        (content_id,),
    ).fetchone()
    return dict(row) if row else None


def record_appeal(
    db: Database,
    *,
    content_id: str,
    appeal_reasoning: str,
) -> Optional[dict[str, Any]]:
    """Atomically verify the content exists and append the appeal.

    The "does this content exist?" read and the appeal insert happen in a **single
    transaction**, so they can't interleave with another writer. Returns the
    original decision row (for the caller's response), or ``None`` if the
    ``content_id`` was never classified — in which case nothing is written.

    The original decision's creator and scores are copied onto the appeal row so a
    reviewer sees the appeal *alongside* what was decided, in one record.
    """
    with db.transaction() as conn:
        original = _latest_row(conn, content_id)
        if original is None:
            return None
        conn.execute(
            _INSERT,
            ("appeal", content_id, original.get("content_type", "text"),
             original.get("creator_id"), _now(),
             original.get("attribution"), original.get("confidence"),
             original.get("llm_score"), original.get("stylometric_score"),
             "under_review", appeal_reasoning, original.get("text_snippet")),
        )
        return original


def get_recent(db: Database, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent entries (newest first) as plain dicts."""
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_all(db: Database) -> list[dict[str, Any]]:
    """Return every entry (oldest first) — used by analytics aggregation."""
    with db.connection() as conn:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
    return [dict(row) for row in rows]


def latest_for_content(db: Database, content_id: str) -> Optional[dict[str, Any]]:
    """Most recent row for a content_id, or None if it was never classified."""
    with db.connection() as conn:
        return _latest_row(conn, content_id)
