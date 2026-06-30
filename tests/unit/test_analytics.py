"""UNIT TESTS — analytics aggregation, provenance.analytics.summarize.

Scope: the pure aggregation function in isolation (no DB). Verifies the detection
pattern, appeal rate, and extra metrics are computed correctly, and that an empty
log degrades to zeros / null instead of dividing by zero.
"""

from __future__ import annotations

from provenance.analytics import summarize


def _c(attribution, confidence):
    return {"event_type": "classification", "attribution": attribution, "confidence": confidence}


def _appeal():
    return {"event_type": "appeal", "attribution": "likely_ai", "confidence": 0.8}


def test_empty_log_is_safe():
    out = summarize([])
    assert out["total_submissions"] == 0
    assert out["appeal_rate"] is None
    assert out["average_confidence"] is None
    assert out["detection_pattern"]["ai_vs_human_ratio"] is None


def test_detection_pattern_counts_and_ratio():
    entries = [
        _c("likely_ai", 0.9),
        _c("likely_ai", 0.8),
        _c("likely_human", 0.1),
        _c("uncertain", 0.55),
    ]
    out = summarize(entries)
    dp = out["detection_pattern"]
    assert (dp["likely_ai"], dp["likely_human"], dp["uncertain"]) == (2, 1, 1)
    assert dp["ai_vs_human_ratio"] == 2.0  # 2 AI : 1 human
    assert out["total_submissions"] == 4


def test_appeal_rate_and_extra_metrics():
    entries = [
        _c("likely_ai", 0.9),
        _c("likely_human", 0.1),
        _c("uncertain", 0.6),
        _c("uncertain", 0.5),
        _appeal(),  # one appeal against 4 classifications
    ]
    out = summarize(entries)
    assert out["total_appeals"] == 1
    assert out["appeal_rate"] == 0.25            # 1 / 4
    assert out["uncertain_rate"] == 0.5          # 2 / 4
    assert out["average_confidence"] == round((0.9 + 0.1 + 0.6 + 0.5) / 4, 4)


def test_appeals_excluded_from_classification_counts():
    # appeal rows must not be counted as submissions/classifications
    out = summarize([_c("likely_ai", 0.9), _appeal()])
    assert out["total_submissions"] == 1
    assert out["detection_pattern"]["likely_ai"] == 1
