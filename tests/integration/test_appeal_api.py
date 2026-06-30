"""INTEGRATION TESTS — /appeal over the Flask app.

Scope: the appeals workflow end-to-end against a temp DB. Verifies an appeal sets
status to 'under_review', is appended to the audit log alongside the original
decision, returns confirmation, and rejects malformed or unknown appeals.
"""

from __future__ import annotations


def _submit(client, text="some submitted writing here", creator_id="creator-1"):
    return client.post("/submit", json={"text": text, "creator_id": creator_id}).get_json()["content_id"]


def test_appeal_updates_status_and_logs(client):
    content_id = _submit(client)

    resp = client.post(
        "/appeal",
        json={"content_id": content_id, "creator_reasoning": "I wrote this myself."},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["content_id"] == content_id
    assert body["status"] == "under_review"

    # the appeal must be visible in the log next to the original decision
    log = client.get("/log").get_json()["entries"]
    appeal_rows = [e for e in log if e["event_type"] == "appeal" and e["content_id"] == content_id]
    assert len(appeal_rows) == 1
    assert appeal_rows[0]["status"] == "under_review"
    assert appeal_rows[0]["appeal_reasoning"] == "I wrote this myself."

    # original classification row is preserved (append-only)
    class_rows = [e for e in log if e["event_type"] == "classification" and e["content_id"] == content_id]
    assert class_rows[0]["status"] == "classified"


def test_appeal_unknown_content_returns_404(client):
    resp = client.post(
        "/appeal",
        json={"content_id": "nonexistent-id", "creator_reasoning": "please review"},
    )
    assert resp.status_code == 404


def test_appeal_requires_content_id(client):
    resp = client.post("/appeal", json={"creator_reasoning": "missing id"})
    assert resp.status_code == 400


def test_appeal_requires_reasoning(client):
    content_id = _submit(client)
    resp = client.post("/appeal", json={"content_id": content_id})
    assert resp.status_code == 400
