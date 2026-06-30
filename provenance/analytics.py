"""Analytics dashboard (stretch feature) — aggregate metrics over the audit log.

Everything needed is already captured in the append-only audit log, so this is
pure aggregation with no new data collection. ``summarize`` is a standalone pure
function (easy to unit-test without a database); ``compute`` wires it to the log.

Metrics (≥3, per the stretch rubric):
- **detection pattern** — attribution distribution + the AI-vs-human verdict ratio
- **appeal rate** — appeals ÷ classifications
- **average confidence** and **uncertain rate** — extra health metrics that fit this
  system's "honest uncertainty" stance.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from provenance import audit
from provenance.db import Database


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate audit-log ``entries`` into dashboard metrics.

    Each submission is one ``classification`` row (unique content_id); each appeal
    is one ``appeal`` row. An empty log yields zeros / ``null`` rather than dividing
    by zero.
    """
    classifications = [e for e in entries if e.get("event_type") == "classification"]
    appeals = [e for e in entries if e.get("event_type") == "appeal"]
    n = len(classifications)

    by_attr = Counter(e.get("attribution") for e in classifications)
    ai = by_attr.get("likely_ai", 0)
    human = by_attr.get("likely_human", 0)
    uncertain = by_attr.get("uncertain", 0)

    avg_conf = (
        round(statistics.fmean(e["confidence"] for e in classifications), 4) if n else None
    )

    return {
        "total_submissions": n,
        "total_appeals": len(appeals),
        # 1) detection pattern
        "detection_pattern": {
            "likely_ai": ai,
            "likely_human": human,
            "uncertain": uncertain,
            "ai_vs_human_ratio": _ratio(ai, human),  # null if no human verdicts yet
            "percent_ai": _ratio(100 * ai, n),
            "percent_human": _ratio(100 * human, n),
            "percent_uncertain": _ratio(100 * uncertain, n),
        },
        # 2) appeal rate
        "appeal_rate": _ratio(len(appeals), n),
        # 3) extra metrics
        "average_confidence": avg_conf,
        "uncertain_rate": _ratio(uncertain, n),
    }


def compute(db: Database) -> dict[str, Any]:
    """Compute dashboard metrics from the audit log."""
    return summarize(audit.get_all(db))
