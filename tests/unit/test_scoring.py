"""UNIT TESTS — confidence scoring, provenance.scoring.

Scope: the signal-blending math in isolation. Verifies the weighted blend, the
disagreement pull toward 0.5, clamping, and the None-stylometry fallback. The
disagreement-pull behavior is the heart of the system's honesty guarantee, so it
is pinned here directly.
"""

from __future__ import annotations

import pytest

from provenance import scoring
from provenance.config import LIKELY_AI_MIN, LIKELY_HUMAN_MAX


def test_agreeing_high_signals_stay_confident_ai():
    score = scoring.combine(0.9, 0.85)
    assert score >= LIKELY_AI_MIN


def test_agreeing_low_signals_stay_confident_human():
    score = scoring.combine(0.1, 0.15)
    assert score < LIKELY_HUMAN_MAX


def test_disagreement_pulls_into_uncertain_band():
    # one signal screams AI, the other screams human -> must NOT be confident
    score = scoring.combine(0.9, 0.1)
    assert LIKELY_HUMAN_MAX <= score < LIKELY_AI_MIN


def test_full_agreement_at_midpoint_is_midpoint():
    assert scoring.combine(0.5, 0.5) == pytest.approx(0.5)


def test_none_stylometry_falls_back_to_llm():
    assert scoring.combine(0.73, None) == pytest.approx(0.73)


def test_result_is_clamped():
    assert 0.0 <= scoring.combine(1.0, 1.0) <= 1.0
    assert 0.0 <= scoring.combine(0.0, 0.0) <= 1.0


def test_more_disagreement_pulls_harder():
    mild = scoring.combine(0.8, 0.6)   # blended 0.72, small gap
    severe = scoring.combine(0.9, 0.1)  # blended 0.58, large gap
    # severe disagreement should sit closer to 0.5 than the mild case is to its blend
    assert abs(severe - 0.5) < abs(mild - 0.5)


# --- ensemble (3+ signals, stretch feature) --------------------------------

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


def test_two_signal_combine_is_ensemble_special_case():
    from provenance.config import W_LLM, W_STYLO

    assert scoring.combine(0.9, 0.1) == pytest.approx(
        scoring.combine_ensemble([(0.9, W_LLM), (0.1, W_STYLO)])
    )
