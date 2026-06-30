"""Provenance certificate (stretch feature) — the "Verified Human Creator" credential.

This is a credential of **provenance/accountability**, not a claim that an algorithm
proved humanity. A creator earns it by (1) registering a personal writing baseline
from a few of their own samples and (2) accepting an authorship pledge. We
deliberately do NOT gate it on passing the AI detector — that detector is an
acknowledged-unreliable signal, and gating on it would deny the credential to the
formal / non-native human writers most prone to false positives (the people the
system protects). See planning.md "Provenance certificate".

Storage is a ``verified_creators`` table in the same SQLite DB. The enrolled
baseline (mean of the three stylometric sub-scores across the samples) is reused to
score how well a later submission matches the creator's established style.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any, Optional

from provenance.db import Database
from provenance.signals.stylometry import stylometric_members

MIN_SAMPLES = 3
MIN_SAMPLE_WORDS = 20  # above stylometry's ~15-word floor so the baseline is meaningful
CREDENTIAL = "Verified Human Creator"
STATEMENT = (
    "This creator has registered an authorship baseline and attests that this is "
    "their own original work."
)
_BASELINE_KEYS = ("function_words", "punctuation", "burstiness")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verified_creators (
    creator_id            TEXT PRIMARY KEY,
    verified_at           TEXT NOT NULL,
    samples_count         INTEGER NOT NULL,
    baseline_function_words REAL NOT NULL,
    baseline_punctuation    REAL NOT NULL,
    baseline_burstiness     REAL NOT NULL
);
"""


class EnrollmentError(ValueError):
    """Raised when a verification request is invalid (maps to HTTP 400)."""


def init_db(db: Database) -> None:
    with db.transaction() as conn:
        conn.executescript(_SCHEMA)


def _baseline_from_samples(samples: list[str]) -> dict[str, float]:
    """Mean stylometric fingerprint across the enrolment samples."""
    per_sample = [stylometric_members(s) for s in samples]
    return {k: statistics.fmean(m[k] for m in per_sample) for k in _BASELINE_KEYS}


def enroll(
    db: Database,
    creator_id: str,
    samples: Any,
    *,
    pledge_accepted: bool,
) -> dict[str, Any]:
    """Register a creator's baseline and grant the credential.

    Raises ``EnrollmentError`` if the pledge isn't accepted or there aren't at least
    ``MIN_SAMPLES`` samples of ``MIN_SAMPLE_WORDS`` words. Re-enrolling updates the
    stored baseline.
    """
    if not isinstance(creator_id, str) or not creator_id.strip():
        raise EnrollmentError("creator_id is required.")
    if not pledge_accepted:
        raise EnrollmentError("The authorship pledge must be accepted (pledge_accepted=true).")
    if not isinstance(samples, list):
        raise EnrollmentError("samples must be a list of writing samples.")

    valid = [s for s in samples if isinstance(s, str) and len(s.split()) >= MIN_SAMPLE_WORDS]
    if len(valid) < MIN_SAMPLES:
        raise EnrollmentError(
            f"At least {MIN_SAMPLES} writing samples of >= {MIN_SAMPLE_WORDS} words "
            f"each are required to establish a baseline."
        )

    baseline = _baseline_from_samples(valid)
    verified_at = datetime.now(timezone.utc).isoformat()
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO verified_creators (
                creator_id, verified_at, samples_count,
                baseline_function_words, baseline_punctuation, baseline_burstiness
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(creator_id) DO UPDATE SET
                verified_at=excluded.verified_at,
                samples_count=excluded.samples_count,
                baseline_function_words=excluded.baseline_function_words,
                baseline_punctuation=excluded.baseline_punctuation,
                baseline_burstiness=excluded.baseline_burstiness
            """,
            (creator_id, verified_at, len(valid),
             baseline["function_words"], baseline["punctuation"], baseline["burstiness"]),
        )

    return {
        "creator_id": creator_id,
        "credential": CREDENTIAL,
        "verified_at": verified_at,
        "samples_enrolled": len(valid),
        "baseline": {k: round(v, 4) for k, v in baseline.items()},
        "statement": STATEMENT,
    }


def get_certificate(db: Database, creator_id: str) -> Optional[dict[str, Any]]:
    """Return the stored credential + baseline for a creator, or None."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM verified_creators WHERE creator_id = ?", (creator_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "creator_id": row["creator_id"],
        "credential": CREDENTIAL,
        "verified_at": row["verified_at"],
        "samples_enrolled": row["samples_count"],
        "baseline": {
            "function_words": row["baseline_function_words"],
            "punctuation": row["baseline_punctuation"],
            "burstiness": row["baseline_burstiness"],
        },
        "statement": STATEMENT,
    }


def is_verified(db: Database, creator_id: str) -> bool:
    return get_certificate(db, creator_id) is not None


def baseline_consistency(baseline: dict[str, float], members: dict[str, float]) -> float:
    """How well a submission's fingerprint matches the enrolled baseline (1 = identical)."""
    diff = statistics.fmean(abs(baseline[k] - members[k]) for k in _BASELINE_KEYS)
    return round(max(0.0, 1.0 - diff), 4)


def count_verified(db: Database) -> int:
    with db.connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM verified_creators").fetchone()[0]
