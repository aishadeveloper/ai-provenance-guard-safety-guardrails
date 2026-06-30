"""INTEGRATION TESTS — /verify, /certificate, and the verified badge on /submit.

Scope: the provenance-certificate workflow end-to-end over the Flask app (temp DB,
stubbed Groq). Verifies a creator can earn the credential, that it is looked up,
that a verified creator's /submit response carries a distinct `provenance` badge,
and that an unverified creator's does not.
"""

from __future__ import annotations

SAMPLES = [
    ("I spent the whole weekend repotting plants and somehow got soil in places soil should "
     "never be, and now every windowsill in the flat smells faintly of damp earth."),
    ("The bakery on the corner changed hands again, and honestly the new sourdough is a clear "
     "downgrade, though the woman at the till is much friendlier than the last lot were."),
    ("My neighbour's cat has decided my windowsill is hers now and frankly I lack the will to "
     "argue, so most mornings we just sit there together watching the buses go past slowly."),
]


def test_verify_earns_certificate(client):
    resp = client.post(
        "/verify",
        json={"creator_id": "writer-1", "samples": SAMPLES, "pledge_accepted": True},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["credential"] == "Verified Human Creator"
    assert body["samples_enrolled"] == 3


def test_verify_rejects_without_pledge(client):
    resp = client.post("/verify", json={"creator_id": "w", "samples": SAMPLES, "pledge_accepted": False})
    assert resp.status_code == 400


def test_verify_rejects_too_few_samples(client):
    resp = client.post("/verify", json={"creator_id": "w", "samples": SAMPLES[:2], "pledge_accepted": True})
    assert resp.status_code == 400


def test_certificate_lookup(client):
    client.post("/verify", json={"creator_id": "writer-1", "samples": SAMPLES, "pledge_accepted": True})
    found = client.get("/certificate?creator_id=writer-1")
    assert found.status_code == 200
    assert found.get_json()["credential"] == "Verified Human Creator"

    missing = client.get("/certificate?creator_id=ghost")
    assert missing.status_code == 404
    assert missing.get_json()["verified_human_creator"] is False


def test_verified_submit_carries_badge(client):
    client.post("/verify", json={"creator_id": "writer-1", "samples": SAMPLES, "pledge_accepted": True})
    resp = client.post("/submit", json={"text": "a fresh piece of my own writing here", "creator_id": "writer-1"})
    prov = resp.get_json()["provenance"]
    assert prov["verified_human_creator"] is True
    assert prov["badge"] == "✓ Verified Human Creator"
    assert 0.0 <= prov["baseline_consistency"] <= 1.0


def test_unverified_submit_has_no_badge(client):
    resp = client.post("/submit", json={"text": "a fresh piece of my own writing here", "creator_id": "stranger"})
    prov = resp.get_json()["provenance"]
    assert prov["verified_human_creator"] is False
    assert "badge" not in prov
