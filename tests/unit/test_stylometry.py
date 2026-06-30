"""UNIT TESTS — Signal 2 (stylometry), provenance.signals.stylometry.

Scope: the deterministic stylometric pipeline in isolation. Verifies feature
extraction, the metric -> sub-score mappings, the Burrows's Delta sub-score, and
the short-text fallback. Stylometry is fully deterministic, so these assert
concrete ordering between known-AI and known-human text without any LLM.
"""

from __future__ import annotations

from provenance.signals import stylometry as st

HUMAN = (
    "ok so i finally tried that new ramen place downtown and honestly? underwhelming. "
    "the broth was fine but they put WAY too much sodium in it and i was thirsty for "
    "like three hours after. probably won't go back unless someone drags me there."
)
AI = (
    "Artificial intelligence represents a transformative paradigm shift in modern "
    "society. It is important to note that while the benefits of AI are numerous, it is "
    "equally essential to consider the ethical implications. Furthermore, stakeholders "
    "across various sectors must collaborate to ensure responsible deployment."
)


def test_word_and_sentence_extraction():
    assert st._words("Hello, world! It's fine.") == ["hello", "world", "it's", "fine"]
    assert len(st._sentences("One. Two! Three?")) == 3


def test_burstiness_none_for_single_sentence():
    assert st._burstiness("just one sentence here") is None


def test_short_text_is_neutral():
    result = st.stylometric_signal("too short")
    assert result["ai_likelihood"] == 0.5
    assert "note" in result["detail"]


def test_ai_text_scores_higher_than_human_text():
    ai_score = st.stylometric_signal(AI)["ai_likelihood"]
    human_score = st.stylometric_signal(HUMAN)["ai_likelihood"]
    assert ai_score > human_score


def test_detail_exposes_all_three_subscores():
    detail = st.stylometric_signal(AI)["detail"]
    assert set(detail) == {"function_words", "punctuation", "burstiness"}
    for v in detail.values():
        assert 0.0 <= v <= 1.0


def test_build_reference_produces_distinct_anchors():
    ref = st.build_reference([HUMAN], [AI])
    # the function-word profiles for the two classes should not be identical
    assert ref.fw_human_profile != ref.fw_ai_profile


def test_delta_subscore_is_directional():
    """Delta sub-score should be lower for human-styled text than AI-styled text."""
    ref = st.default_reference()
    assert st._delta_subscore(HUMAN, ref) < st._delta_subscore(AI, ref)
