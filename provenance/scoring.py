"""Confidence scoring — blend the two signals into one calibrated AI-likelihood.

Both signals report on the same 0–1 "how AI-like" scale, so blending is a
weighted average plus a **disagreement pull**: when the semantic (LLM) and
structural (stylometry) signals conflict, the result is pulled toward 0.5
(uncertain) instead of reporting a confident-looking average. A confident verdict
therefore requires the two independent signals to *agree* — which is exactly the
honesty property the project asks for. (planning.md "Confidence scoring".)
"""

from __future__ import annotations

from typing import Optional

from provenance.config import DISAGREEMENT_PULL, W_LLM, W_STYLO


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

    total_w = w_llm + w_stylo
    blended = (w_llm * llm_score + w_stylo * stylometric_score) / total_w

    # Pull toward 0.5 only on *genuine* conflict: the signals must sit on
    # opposite sides of 0.5 (one says AI, the other says human). A merely
    # non-committal signal near 0.5 is not a conflict and must not drag a
    # confident signal down. Conflict strength is set by the *weaker* signal's
    # distance from 0.5 (scaled to 0..1) — a barely-leaning opposite signal
    # pulls little; two strongly-opposed signals pull hard.
    d_llm = llm_score - 0.5
    d_stylo = stylometric_score - 0.5
    if d_llm * d_stylo < 0:  # opposite sides of 0.5
        conflict = min(abs(d_llm), abs(d_stylo)) * 2.0
        blended = blended + (0.5 - blended) * conflict * pull

    return max(0.0, min(1.0, blended))
