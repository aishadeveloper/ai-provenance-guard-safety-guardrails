"""Transparency labels — maps an AI-likelihood score to a reader-facing label.

Three variants, selected by the asymmetric thresholds in ``config``. The exact
display text is fixed here (and reproduced verbatim in the README) so wording
cannot silently drift; a regression test pins it. Design principles: never a hard
authorship claim ("likely"/"signs of", never "this was written by AI"); every
label admits fallibility; the AI label names the appeal path; the uncertain label
explains *why* it is uncertain. (planning.md "Transparency labels".)
"""

from __future__ import annotations

from provenance.config import LIKELY_AI_MIN, LIKELY_HUMAN_MAX

LIKELY_HUMAN = "likely_human"
UNCERTAIN = "uncertain"
LIKELY_AI = "likely_ai"

LABEL_TEXT = {
    LIKELY_HUMAN: (
        "Likely human-written. Our automated check found little sign of AI "
        "generation in this piece. Automated checks aren't perfect — this is a "
        "signal, not a verdict."
    ),
    UNCERTAIN: (
        "Uncertain. Our check couldn't confidently tell whether a person wrote "
        "this or AI generated it. This is common for short pieces, or for "
        "writing a person wrote and then used AI to edit. No conclusion is being "
        "drawn."
    ),
    LIKELY_AI: (
        "Likely AI-generated. Our automated check found strong signs this text "
        "was produced or heavily shaped by AI. This is an automated assessment, "
        "not a certainty — if you wrote this yourself, you can appeal."
    ),
}


def attribution_for(score: float) -> str:
    """Map a 0–1 AI-likelihood to one attribution band.

    Bands are contiguous: ``score < 0.40`` is likely_human, ``0.40 <= score <
    0.70`` is uncertain, ``score >= 0.70`` is likely_ai.
    """
    if score < LIKELY_HUMAN_MAX:
        return LIKELY_HUMAN
    if score < LIKELY_AI_MIN:
        return UNCERTAIN
    return LIKELY_AI


def label_text(attribution: str) -> str:
    """Return the exact reader-facing text for an attribution band."""
    return LABEL_TEXT[attribution]


def classify_label(score: float) -> tuple[str, str]:
    """Convenience: ``score`` -> ``(attribution, label_text)``."""
    attribution = attribution_for(score)
    return attribution, label_text(attribution)
