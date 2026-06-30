"""UNIT TESTS — multi-modal image-metadata signals, provenance.signals.metadata.

Scope: the three deterministic image-metadata signals and their ensemble combination
in isolation. Verifies each signal fires on the right cue, that an AI-generated image's
metadata scores higher than a real photo's, and that empty/signal-free metadata lands
neutral (no evidence -> uncertain).
"""

from __future__ import annotations

from provenance.signals import metadata as md

AI_IMAGE = {"software": "Midjourney v6", "format": "PNG"}
AI_C2PA = {"c2pa": {"digital_source_type": "trainedAlgorithmicMedia", "claim_generator": "DALL-E"}}
REAL_PHOTO = {
    "exif": {
        "Make": "Canon", "Model": "EOS R6", "LensModel": "RF24-70mm",
        "ExposureTime": "1/250", "FNumber": 2.8, "ISO": 200,
        "GPSLatitude": "51.5", "DateTimeOriginal": "2025:08:01 14:22:10",
    }
}


def test_generator_signature_detects_named_tool():
    assert md.generator_signature(AI_IMAGE) >= 0.9
    assert md.generator_signature(REAL_PHOTO) == 0.5  # no generator named


def test_generator_signature_detects_c2pa_ai_source():
    assert md.generator_signature(AI_C2PA) >= 0.9


def test_camera_fingerprint_human_for_photo_abstains_without_exif():
    assert md.camera_fingerprint(REAL_PHOTO) <= 0.2     # rich EXIF -> human
    # no camera EXIF -> abstain (0.5), NOT positive AI evidence (false-positive guard)
    assert md.camera_fingerprint(AI_IMAGE) == 0.5


def test_content_credential_reads_assertions():
    assert md.content_credential(AI_C2PA) >= 0.9
    assert md.content_credential({"c2pa": {"ai_generated": False, "type": "digitalCapture"}}) <= 0.1
    assert md.content_credential(REAL_PHOTO) == 0.5     # no credential present


def test_ai_image_scores_higher_than_real_photo():
    ai = md.metadata_signal(AI_IMAGE)["ai_likelihood"]
    photo = md.metadata_signal(REAL_PHOTO)["ai_likelihood"]
    assert ai > 0.7
    assert photo < 0.3


def test_empty_metadata_is_uncertain():
    out = md.metadata_signal({})
    assert out["ai_likelihood"] == 0.5
    assert set(out["signals"]) == {"generator_signature", "camera_fingerprint", "content_credential"}


def test_conflicting_metadata_pulls_toward_uncertain():
    # an AI generator tag AND a full camera fingerprint disagree -> should not be confident
    conflicted = {"software": "Stable Diffusion", **REAL_PHOTO}
    score = md.metadata_signal(conflicted)["ai_likelihood"]
    assert 0.35 <= score <= 0.7
