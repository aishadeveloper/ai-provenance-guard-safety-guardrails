"""Confidence scoring — blend the two signals into one calibrated AI-likelihood.

Both signals report on the same 0–1 "how AI-like" scale, so blending is a
weighted average plus a **disagreement pull**: when the semantic (LLM) and
structural (stylometry) signals conflict, the result is pulled toward 0.5
(uncertain) instead of reporting a confident-looking average. A confident verdict
therefore requires the two independent signals to *agree* — which is exactly the
honesty property the project asks for. (planning.md "Confidence scoring".)
"""

from __future__ import annotations

from typing import Optional, Sequence

from provenance.config import DISAGREEMENT_PULL, W_LLM, W_STYLO


def combine_ensemble(
    members: Sequence[tuple[float, float]],
    *,
    pull: float = DISAGREEMENT_PULL,
) -> float:
    """Combine N weighted detector scores into one AI-likelihood, clamped to [0, 1].

    ``members`` is a sequence of ``(score, weight)`` pairs (each score is a 0–1
    AI-likelihood). The result is the weighted mean pulled toward 0.5 in proportion
    to how much the members' evidence **cancels**:

      net   = |Σ wᵢ·(sᵢ − 0.5)|     (how strongly they agree on a direction)
      gross = Σ wᵢ·|sᵢ − 0.5|        (total evidence, regardless of direction)
      conflict = 1 − net/gross       (0 = all lean the same way, 1 = evidence cancels)

    A member near 0.5 has small ``|sᵢ − 0.5|``, so it neither manufactures nor masks
    conflict. This is the N-member generalization of the 2-signal disagreement pull.
    """
    active = [(s, w) for s, w in members if w > 0]
    total_w = sum(w for _, w in active)
    if total_w == 0:
        return 0.5

    mean = sum(s * w for s, w in active) / total_w
    net = abs(sum((s - 0.5) * w for s, w in active)) / total_w
    gross = sum(abs(s - 0.5) * w for s, w in active) / total_w
    conflict = 0.0 if gross == 0 else 1.0 - (net / gross)

    pulled = mean + (0.5 - mean) * conflict * pull
    return max(0.0, min(1.0, pulled))


def combine(
    llm_score: float,
    stylometric_score: Optional[float],
    *,
    w_llm: float = W_LLM,
    w_stylo: float = W_STYLO,
    pull: float = DISAGREEMENT_PULL,
) -> float:
    """Blend two AI-likelihoods into one, clamped to [0, 1].

    If ``stylometric_score`` is None (e.g. signal unavailable), fall back to the
    LLM score alone rather than inventing a value.
    """
    if stylometric_score is None:
        return max(0.0, min(1.0, llm_score))

    # The 2-signal blend is the special case of the N-member ensemble rule.
    return combine_ensemble(
        [(llm_score, w_llm), (stylometric_score, w_stylo)], pull=pull
    )
