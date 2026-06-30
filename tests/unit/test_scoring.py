"""UNIT TESTS — confidence scoring, provenance.scoring.

Scope: the signal-blending math in isolation. The live system blends via
``combine_ensemble`` (text uses 4 members, image-metadata 3), so the weighted
blend, the conflict pull toward 0.5, clamping, and weighting are pinned directly
on ``combine_ensemble`` below. The thin 2-signal ``combine`` wrapper delegates to
it, so it gets exactly two dedicated tests — its equivalence to the ensemble and
its one extra branch (None stylometry) — rather than re-proving the same bands.
"""

from __future__ import annotations

import pytest

from provenance import scoring
from provenance.config import LIKELY_AI_MIN, LIKELY_HUMAN_MAX


# --- combine_ensemble: the blending math (single source of truth) ----------

def test_ensemble_all_members_agree_high_is_confident_ai():
    score = scoring.combine_ensemble([(0.9, 0.55), (0.8, 0.2), (0.85, 0.15), (0.8, 0.1)])
    assert score >= LIKELY_AI_MIN


def test_ensemble_all_members_agree_low_is_confident_human():
    score = scoring.combine_ensemble([(0.1, 0.55), (0.2, 0.2), (0.15, 0.15), (0.1, 0.1)])
    assert score < LIKELY_HUMAN_MAX


def test_ensemble_split_members_pull_into_uncertain():
    score = scoring.combine_ensemble([(0.9, 0.55), (0.1, 0.45)])
    assert LIKELY_HUMAN_MAX <= score < LIKELY_AI_MIN


def test_ensemble_neutral_member_does_not_create_conflict():
    # one strong-AI member + one perfectly neutral member: the neutral one carries
    # no evidence, so it must NOT pull the result toward uncertain.
    with_neutral = scoring.combine_ensemble([(0.9, 0.5), (0.5, 0.5)])
    assert with_neutral == pytest.approx(0.7)  # plain weighted mean, no pull


def test_ensemble_respects_weights():
    heavy_ai = scoring.combine_ensemble([(0.9, 0.9), (0.1, 0.1)])
    heavy_human = scoring.combine_ensemble([(0.9, 0.1), (0.1, 0.9)])
    assert heavy_ai > 0.6
    assert heavy_human < 0.4


def test_ensemble_empty_or_zero_weight_is_neutral():
    assert scoring.combine_ensemble([]) == 0.5
    assert scoring.combine_ensemble([(0.9, 0.0)]) == 0.5


def test_ensemble_result_is_clamped():
    assert 0.0 <= scoring.combine_ensemble([(1.0, 0.6), (1.0, 0.4)]) <= 1.0
    assert 0.0 <= scoring.combine_ensemble([(0.0, 0.6), (0.0, 0.4)]) <= 1.0


# --- combine (2-signal wrapper): only what the ensemble tests don't cover ---

def test_two_signal_combine_is_ensemble_special_case():
    # combine() delegates to combine_ensemble, so it inherits all the bands above.
    from provenance.config import W_LLM, W_STYLO

    assert scoring.combine(0.9, 0.1) == pytest.approx(
        scoring.combine_ensemble([(0.9, W_LLM), (0.1, W_STYLO)])
    )


def test_none_stylometry_falls_back_to_llm():
    # the one branch unique to combine(): a missing second signal -> just the LLM
    assert scoring.combine(0.73, None) == pytest.approx(0.73)
