"""Classification pipeline — orchestrates signals -> scoring -> label.

This is the stable seam the HTTP layer depends on. Through the milestones its
internals grow (M3: Signal 1 only; M4: + Signal 2 and blended scoring) while its
output contract stays fixed, so the /submit route never has to change.
"""

from __future__ import annotations

from typing import Any, Optional

import statistics

from provenance import labels, scoring
from provenance.config import ENSEMBLE_WEIGHTS
from provenance.signals.llm import llm_signal
from provenance.signals.stylometry import stylometric_members


def classify(text: str, *, llm_client: Optional[Any] = None) -> dict[str, Any]:
    """Run the full detection pipeline on ``text`` and return a decision dict.

    Keys: ``llm_score`` (semantic member), ``stylometric_score`` (equal-weight mean
    of the structural members, kept for the audit log), ``confidence`` (the 4-member
    weighted ensemble), ``signals`` (per-member 0–1 scores), ``attribution``,
    ``label``, plus diagnostics (``verdict``, ``llm_reasoning``, ``llm_error``).

    The ensemble members are the LLM plus the three stylometric properties, each
    independently weighted (config.ENSEMBLE_WEIGHTS) and combined by
    ``scoring.combine_ensemble`` with a conflict-scaled pull toward uncertainty.
    """
    llm = llm_signal(text, client=llm_client)
    llm_score = llm["ai_likelihood"]

    stylo = stylometric_members(text)  # function_words, punctuation, burstiness
    stylometric_score = statistics.fmean(stylo.values())  # aggregate for the log

    members = {"llm": llm_score, **stylo}
    confidence = scoring.combine_ensemble(
        [(score, ENSEMBLE_WEIGHTS[name]) for name, score in members.items()]
    )

    attribution, label = labels.classify_label(confidence)
    return {
        "llm_score": llm_score,
        "stylometric_score": stylometric_score,
        "confidence": confidence,
        "signals": {name: round(score, 4) for name, score in members.items()},
        "attribution": attribution,
        "label": label,
        "verdict": llm["verdict"],
        "llm_reasoning": llm["reasoning"],
        "llm_error": llm["error"],
    }
