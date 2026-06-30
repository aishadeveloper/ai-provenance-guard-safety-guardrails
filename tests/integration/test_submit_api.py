"""INTEGRATION TESTS — /submit and /log over the Flask app.

Scope: the HTTP layer wired to the real pipeline, real audit log (temp DB), and a
stubbed Groq client (via the ``client`` fixture). Verifies the response contract
and that every submission is persisted to the audit log and surfaced by /log.
These assert the *contract* (keys, types, persistence), so they survive as the
pipeline internals grow in later milestones.
"""

from __future__ import annotations

from provenance.app import create_app


def test_submit_returns_full_contract(client):
    resp = client.post("/submit", json={"text": "a poem about the sea", "creator_id": "u1"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data["content_id"], str) and data["content_id"]
    assert data["attribution"] in {"likely_human", "uncertain", "likely_ai"}
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["label"], str) and data["label"]


def test_submit_persists_to_audit_log(client):
    resp = client.post("/submit", json={"text": "hello there world", "creator_id": "u1"})
    content_id = resp.get_json()["content_id"]

    log = client.get("/log").get_json()["entries"]
    assert len(log) == 1
    entry = log[0]
    assert entry["content_id"] == content_id
    assert entry["creator_id"] == "u1"
    assert entry["event_type"] == "classification"
    assert entry["status"] == "classified"


def test_submit_rejects_missing_text(client):
    resp = client.post("/submit", json={"creator_id": "u1"})
    assert resp.status_code == 400
    assert "text" in resp.get_json()["error"].lower()


def test_submit_rejects_blank_text(client):
    resp = client.post("/submit", json={"text": "   ", "creator_id": "u1"})
    assert resp.status_code == 400


def test_submit_rejects_missing_creator_id(client):
    resp = client.post("/submit", json={"text": "real content here"})
    assert resp.status_code == 400
    assert "creator_id" in resp.get_json()["error"].lower()


def test_log_is_empty_before_any_submission(client):
    assert client.get("/log").get_json()["entries"] == []


def test_ai_stub_drives_attribution(db_path, fake_ai_client):
    """A high AI-likelihood stub should yield a likely_ai attribution end-to-end."""
    app = create_app(db_path=db_path, llm_client=fake_ai_client, testing=True)
    c = app.test_client()
    data = c.post("/submit", json={"text": "uniform polished prose", "creator_id": "u2"}).get_json()
    assert data["attribution"] == "likely_ai"
    assert data["confidence"] >= 0.7
