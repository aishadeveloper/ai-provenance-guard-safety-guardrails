"""UNIT TESTS — provenance certificate logic, provenance.certificates.

Scope: enrollment validation, baseline computation, consistency math, and lookup,
against a temp DB. No Flask. The certificate is a provenance/accountability
credential earned via baseline enrollment + pledge — these tests pin that contract.
"""

from __future__ import annotations

import pytest

from provenance import certificates

SAMPLE_A = ("I spent the whole weekend repotting plants and somehow got soil in places soil "
            "should never be, and now every windowsill in the flat smells faintly of damp earth.")
SAMPLE_B = ("The bakery on the corner changed hands again, and honestly the new sourdough is a "
            "clear downgrade, though the woman at the till is much friendlier than the last lot.")
SAMPLE_C = ("My neighbour's cat has decided my windowsill is hers now and frankly I lack the will "
            "to argue, so most mornings we just sit there together watching the buses go past.")


def _samples():
    return [SAMPLE_A, SAMPLE_B, SAMPLE_C]


def test_enroll_grants_credential(db_path):
    certificates.init_db(db_path)
    cert = certificates.enroll(db_path, "writer-1", _samples(), pledge_accepted=True)
    assert cert["credential"] == "Verified Human Creator"
    assert cert["samples_enrolled"] == 3
    assert set(cert["baseline"]) == {"function_words", "punctuation", "burstiness"}
    assert certificates.is_verified(db_path, "writer-1")


def test_enroll_requires_pledge(db_path):
    certificates.init_db(db_path)
    with pytest.raises(certificates.EnrollmentError):
        certificates.enroll(db_path, "writer-1", _samples(), pledge_accepted=False)


def test_enroll_requires_minimum_samples(db_path):
    certificates.init_db(db_path)
    with pytest.raises(certificates.EnrollmentError):
        certificates.enroll(db_path, "writer-1", [SAMPLE_A, SAMPLE_B], pledge_accepted=True)


def test_enroll_rejects_too_short_samples(db_path):
    certificates.init_db(db_path)
    short = ["too short", "also short", "still short"]
    with pytest.raises(certificates.EnrollmentError):
        certificates.enroll(db_path, "writer-1", short, pledge_accepted=True)


def test_unverified_creator_has_no_certificate(db_path):
    certificates.init_db(db_path)
    assert certificates.get_certificate(db_path, "nobody") is None
    assert certificates.is_verified(db_path, "nobody") is False


def test_reenroll_updates_baseline(db_path):
    certificates.init_db(db_path)
    certificates.enroll(db_path, "writer-1", _samples(), pledge_accepted=True)
    cert2 = certificates.enroll(
        db_path, "writer-1",
        [SAMPLE_A + " " + SAMPLE_B, SAMPLE_B + " " + SAMPLE_C, SAMPLE_C + " " + SAMPLE_A],
        pledge_accepted=True,
    )
    assert cert2["samples_enrolled"] == 3
    assert certificates.count_verified(db_path) == 1  # updated, not duplicated


def test_baseline_consistency_is_one_for_identical_fingerprint():
    baseline = {"function_words": 0.4, "punctuation": 0.3, "burstiness": 0.5}
    assert certificates.baseline_consistency(baseline, dict(baseline)) == 1.0


def test_baseline_consistency_drops_with_deviation():
    baseline = {"function_words": 0.2, "punctuation": 0.2, "burstiness": 0.2}
    far = {"function_words": 0.9, "punctuation": 0.9, "burstiness": 0.9}
    assert certificates.baseline_consistency(baseline, far) < 0.5
