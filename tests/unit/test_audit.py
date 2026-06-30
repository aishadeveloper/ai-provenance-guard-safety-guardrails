"""UNIT TESTS — append-only audit log, provenance.audit.

Scope: the SQLite log in isolation against a temp Database. Verifies entries are
written with the required structured fields, that appeals are *appended* (not
mutated onto the original row) and recorded atomically, and that
``latest_for_content`` reflects the newest status. The append-only guarantee is
core to the audit design.
"""

from __future__ import annotations

from provenance import audit


def _classify(db, content_id="c1", creator_id="u1", attribution="likely_human"):
    audit.log_classification(
        db,
        content_id=content_id,
        creator_id=creator_id,
        attribution=attribution,
        confidence=0.2,
        llm_score=0.15,
        stylometric_score=0.25,
        text_snippet="hello world",
    )


def test_init_is_idempotent(db):
    audit.init_db(db)
    audit.init_db(db)  # second call must not raise
    assert audit.get_recent(db) == []


def test_log_classification_writes_structured_entry(db):
    audit.init_db(db)
    _classify(db)
    entries = audit.get_recent(db)
    assert len(entries) == 1
    e = entries[0]
    assert e["event_type"] == "classification"
    assert e["content_id"] == "c1"
    assert e["creator_id"] == "u1"
    assert e["status"] == "classified"
    assert e["confidence"] == 0.2
    assert e["llm_score"] == 0.15
    assert e["stylometric_score"] == 0.25
    assert e["timestamp"]  # ISO string present


def test_appeal_is_appended_not_mutated(db):
    audit.init_db(db)
    _classify(db, attribution="likely_ai")

    original = audit.record_appeal(db, content_id="c1", appeal_reasoning="I wrote this myself.")
    assert original is not None  # content existed

    entries = audit.get_recent(db)
    assert len(entries) == 2  # appended, original row preserved

    latest = audit.latest_for_content(db, "c1")
    assert latest["event_type"] == "appeal"
    assert latest["status"] == "under_review"
    assert latest["appeal_reasoning"] == "I wrote this myself."
    # original decision context copied onto the appeal row for the reviewer
    assert latest["attribution"] == "likely_ai"
    assert latest["creator_id"] == "u1"  # derived from the original

    # the original classification row still says 'classified'
    classification_rows = [e for e in entries if e["event_type"] == "classification"]
    assert classification_rows[0]["status"] == "classified"


def test_record_appeal_unknown_content_writes_nothing(db):
    audit.init_db(db)
    result = audit.record_appeal(db, content_id="ghost", appeal_reasoning="please review")
    assert result is None
    assert audit.get_recent(db) == []  # nothing written when content doesn't exist


def test_latest_for_content_none_when_unknown(db):
    audit.init_db(db)
    assert audit.latest_for_content(db, "does-not-exist") is None


def test_get_recent_is_newest_first(db):
    audit.init_db(db)
    _classify(db, content_id="a")
    _classify(db, content_id="b")
    entries = audit.get_recent(db)
    assert [e["content_id"] for e in entries] == ["b", "a"]
