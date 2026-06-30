"""INTEGRATION TESTS — GET /analytics over the Flask app.

Scope: the analytics endpoint wired to real submissions and an appeal (temp DB,
stubbed Groq). Verifies the dashboard reflects actual activity and exposes the
three required metric families.
"""

from __future__ import annotations

from provenance.app import create_app
from tests.conftest import FakeGroqClient


def test_empty_analytics(client):
    out = client.get("/analytics").get_json()
    assert out["total_submissions"] == 0
    assert out["appeal_rate"] is None


def test_analytics_reflects_activity(db_path):
    # deterministic AI stub so every submission classifies as likely_ai
    app = create_app(db_path=db_path, llm_client=FakeGroqClient(
        '{"verdict": "ai", "ai_likelihood": 0.95, "reasoning": "x"}'),
        testing=True, ratelimit_enabled=False)
    c = app.test_client()

    ai_text = (
        "Maintaining a healthy work-life balance is essential for productivity and "
        "well-being. First, it is important to establish clear boundaries. Second, "
        "prioritizing tasks effectively can help reduce stress and improve focus."
    )
    cid = c.post("/submit", json={"text": ai_text, "creator_id": "u1"}).get_json()["content_id"]
    c.post("/submit", json={"text": ai_text, "creator_id": "u2"})
    c.post("/appeal", json={"content_id": cid, "creator_reasoning": "I wrote this."})

    out = c.get("/analytics").get_json()
    assert out["total_submissions"] == 2
    assert out["total_appeals"] == 1
    assert out["appeal_rate"] == 0.5
    # all three required metric families are present
    assert "detection_pattern" in out
    assert out["detection_pattern"]["likely_ai"] == 2
    assert "appeal_rate" in out
    assert out["average_confidence"] is not None
