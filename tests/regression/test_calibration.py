"""REGRESSION TESTS — stylometry calibration on the spec's reference texts.

Scope: guards the *calibration* of Signal 2 against drift. Uses the four labelled
sample texts from the project spec (Milestone 4). Stylometry is deterministic, so
these pin the relative ordering the signal must preserve — if a future change to
the metrics, weights, or reference corpus silently re-orders clearly-human vs
clearly-AI text, this fails. (Full end-to-end confidence numbers, which depend on
the live LLM, are documented in the README rather than asserted here.)
"""

from __future__ import annotations

import pytest

from provenance.signals.stylometry import stylometric_signal

# The four labelled inputs from the spec (Milestone 4 test set).
AI_ESSAY = (
    "Artificial intelligence represents a transformative paradigm shift in modern "
    "society. It is important to note that while the benefits of AI are numerous, it is "
    "equally essential to consider the ethical implications. Furthermore, stakeholders "
    "across various sectors must collaborate to ensure responsible deployment."
)
HUMAN_RAMEN = (
    "ok so i finally tried that new ramen place downtown and honestly? underwhelming. "
    "the broth was fine but they put WAY too much sodium in it and i was thirsty for "
    "like three hours after. my friend got the spicy version and said it was better. "
    "probably won't go back unless someone drags me there"
)
FORMAL_HUMAN = (
    "The relationship between monetary policy and asset price inflation has been "
    "extensively studied in the literature. Central banks face a fundamental tension "
    "between their mandate for price stability and the unintended consequences of "
    "prolonged low interest rates on equity and real estate valuations."
)
EDITED_AI = (
    "I've been thinking a lot about remote work lately. There are genuine tradeoffs - "
    "flexibility and no commute on one side, isolation and blurred work-life boundaries "
    "on the other. Studies show productivity varies widely by individual and role type."
)


@pytest.fixture(scope="module")
def scores():
    return {name: stylometric_signal(text)["ai_likelihood"]
            for name, text in {
                "ai_essay": AI_ESSAY,
                "human_ramen": HUMAN_RAMEN,
                "formal_human": FORMAL_HUMAN,
                "edited_ai": EDITED_AI,
            }.items()}


def test_clear_ai_separates_from_clear_human(scores):
    assert scores["ai_essay"] > scores["human_ramen"]


def test_casual_human_is_the_most_human(scores):
    # the casual, irregular human text should be the least AI-like of the four
    assert scores["human_ramen"] == min(scores.values())


def test_formal_human_drifts_toward_ai(scores):
    # documents the known false-positive direction: formal human writing looks
    # more AI-like to stylometry than casual human writing does
    assert scores["formal_human"] > scores["human_ramen"]


def test_all_scores_in_unit_interval(scores):
    assert all(0.0 <= v <= 1.0 for v in scores.values())
