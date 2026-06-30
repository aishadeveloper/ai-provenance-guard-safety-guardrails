"""UNIT TESTS — append-only audit log, provenance.audit.

Scope: the SQLite log in isolation against a temp DB. Verifies entries are
written with the required structured fields, that appeals are *appended* (not
mutated onto the original row), and that ``latest_for_content`` reflects the
newest status. The append-only guarantee is core to the audit design.
"""

from __future__ import annotations

from provenance import audit


def _classify(db_path, content_id="c1", creator_id="u1", attribution="likely_human"):
    audit.log_classification(
        db_path,
        content_id=content_id,
        creator_id=creator_id,
        attribution=attribution,
        confidence=0.2,
        llm_score=0.15,
        stylometric_score=0.25,
        text_snippet="hello world",
    )


def test_init_is_idempotent(db_path):
    audit.init_db(db_path)
    audit.init_db(db_path)  # second call must not raise
    assert audit.get_recent(db_path) == []


def test_log_classification_writes_structured_entry(db_path):
    audit.init_db(db_path)
    _classify(db_path)
    entries = audit.get_recent(db_path)
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


def test_appeal_is_appended_not_mutated(db_path):
    audit.init_db(db_path)
    _classify(db_path, attribution="likely_ai")
    original = audit.latest_for_content(db_path, "c1")

    audit.log_appeal(
        db_path,
        content_id="c1",
        creator_id="u1",
        appeal_reasoning="I wrote this myself.",
        original=original,
    )

    entries = audit.get_recent(db_path)
    assert len(entries) == 2  # appended, original row preserved

    latest = audit.latest_for_content(db_path, "c1")
    assert latest["event_type"] == "appeal"
    assert latest["status"] == "under_review"
    assert latest["appeal_reasoning"] == "I wrote this myself."
    # original decision context copied onto the appeal row for the reviewer
    assert latest["attribution"] == "likely_ai"

    # the original classification row still says 'classified'
    classification_rows = [e for e in entries if e["event_type"] == "classification"]
    assert classification_rows[0]["status"] == "classified"


def test_latest_for_content_none_when_unknown(db_path):
    audit.init_db(db_path)
    assert audit.latest_for_content(db_path, "does-not-exist") is None


def test_get_recent_is_newest_first(db_path):
    audit.init_db(db_path)
    _classify(db_path, content_id="a")
    _classify(db_path, content_id="b")
    entries = audit.get_recent(db_path)
    assert [e["content_id"] for e in entries] == ["b", "a"]
