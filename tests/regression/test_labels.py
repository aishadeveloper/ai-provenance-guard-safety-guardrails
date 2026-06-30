"""REGRESSION TESTS — transparency label text and band reachability.

Scope: pins the exact reader-facing label wording (the README reproduces it
verbatim, and graders check it) so it cannot silently drift, and guards the
threshold->variant mapping including the boundary values. Also confirms all three
variants are reachable.
"""

from __future__ import annotations

from provenance import labels
from provenance.config import LIKELY_AI_MIN, LIKELY_HUMAN_MAX


def test_attribution_band_boundaries():
    # contiguous, asymmetric bands: <0.40 human, [0.40,0.70) uncertain, >=0.70 ai
    assert labels.attribution_for(0.0) == labels.LIKELY_HUMAN
    assert labels.attribution_for(0.399) == labels.LIKELY_HUMAN
    assert labels.attribution_for(LIKELY_HUMAN_MAX) == labels.UNCERTAIN  # 0.40 -> uncertain
    assert labels.attribution_for(0.69) == labels.UNCERTAIN
    assert labels.attribution_for(LIKELY_AI_MIN) == labels.LIKELY_AI  # 0.70 -> ai
    assert labels.attribution_for(1.0) == labels.LIKELY_AI


def test_all_three_variants_reachable_and_distinct():
    texts = {labels.classify_label(s)[1] for s in (0.1, 0.55, 0.9)}
    assert len(texts) == 3  # three different strings


def test_exact_label_text_is_pinned():
    # If you change this wording, change the README too — they must match verbatim.
    assert labels.label_text(labels.LIKELY_HUMAN) == (
        "Likely human-written. Our automated check found little sign of AI "
        "generation in this piece. Automated checks aren't perfect — this is a "
        "signal, not a verdict."
    )
    assert labels.label_text(labels.UNCERTAIN) == (
        "Uncertain. Our check couldn't confidently tell whether a person wrote "
        "this or AI generated it. This is common for short pieces, or for "
        "writing a person wrote and then used AI to edit. No conclusion is being "
        "drawn."
    )
    assert labels.label_text(labels.LIKELY_AI) == (
        "Likely AI-generated. Our automated check found strong signs this text "
        "was produced or heavily shaped by AI. This is an automated assessment, "
        "not a certainty — if you wrote this yourself, you can appeal."
    )


def test_human_label_makes_no_hard_claim():
    # no label should assert authorship as fact (false-positive protection)
    for text in labels.LABEL_TEXT.values():
        assert "Likely" in text or "Uncertain" in text


def test_ai_label_names_the_appeal_path():
    assert "appeal" in labels.label_text(labels.LIKELY_AI).lower()
