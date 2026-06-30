"""REGRESSION TESTS — ensemble detection invariants (stretch feature).

Scope: guards the ensemble's structural contract — that it remains a documented
3-or-more-signal weighted ensemble with weights that form a proper weighting
(sum to 1.0). If someone drops a member or breaks the weighting, this fails.
"""

from __future__ import annotations

import pytest

from provenance.config import ENSEMBLE_WEIGHTS


def test_ensemble_has_at_least_three_signals():
    assert len(ENSEMBLE_WEIGHTS) >= 3


def test_ensemble_includes_both_semantic_and_structural_members():
    # the LLM (semantic) plus at least two structural stylometric members
    assert "llm" in ENSEMBLE_WEIGHTS
    structural = set(ENSEMBLE_WEIGHTS) - {"llm"}
    assert len(structural) >= 2


def test_ensemble_weights_sum_to_one():
    assert sum(ENSEMBLE_WEIGHTS.values()) == pytest.approx(1.0)


def test_ensemble_weights_are_positive():
    assert all(w > 0 for w in ENSEMBLE_WEIGHTS.values())
