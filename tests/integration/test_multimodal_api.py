"""INTEGRATION TESTS — multi-modal /submit (content_type='metadata').

Scope: the image-metadata modality end-to-end over the same /submit endpoint
(temp DB). Verifies a metadata payload is attributed and returns the standard
result shape, that it is logged with content_type='metadata', and that bad inputs
are rejected.
"""

from __future__ import annotations

AI_IMAGE = {"software": "Midjourney v6", "format": "PNG", "width": 1024, "height": 1024}
REAL_PHOTO = {"exif": {"Make": "Canon", "Model": "EOS R6", "ExposureTime": "1/250",
                       "FNumber": 2.8, "ISO": 200, "GPSLatitude": "51.5"}}


def test_metadata_submission_returns_result(client):
    resp = client.post("/submit", json={
        "content_type": "metadata", "creator_id": "studio-1", "metadata": AI_IMAGE,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["content_type"] == "metadata"
    assert data["attribution"] == "likely_ai"
    assert set(data["signals"]) == {"generator_signature", "camera_fingerprint", "content_credential"}
    assert isinstance(data["label"], str) and data["label"]


def test_real_photo_metadata_scores_human(client):
    data = client.post("/submit", json={
        "content_type": "metadata", "creator_id": "photog-1", "metadata": REAL_PHOTO,
    }).get_json()
    assert data["attribution"] == "likely_human"


def test_metadata_submission_is_logged_with_content_type(client):
    client.post("/submit", json={"content_type": "metadata", "creator_id": "studio-1", "metadata": AI_IMAGE})
    log = client.get("/log").get_json()["entries"]
    assert log[0]["content_type"] == "metadata"


def test_text_submission_defaults_content_type_text(client):
    client.post("/submit", json={"text": "a normal text submission here", "creator_id": "u1"})
    log = client.get("/log").get_json()["entries"]
    assert log[0]["content_type"] == "text"


def test_metadata_requires_object(client):
    resp = client.post("/submit", json={"content_type": "metadata", "creator_id": "u1", "metadata": "not-an-object"})
    assert resp.status_code == 400


def test_unsupported_content_type_rejected(client):
    resp = client.post("/submit", json={"content_type": "audio", "creator_id": "u1", "metadata": {}})
    assert resp.status_code == 400
