"""Classification pipeline — orchestrates signals -> scoring -> label.

This is the stable seam the HTTP layer depends on. Through the milestones its
internals grow (M3: Signal 1 only; M4: + Signal 2 and blended scoring) while its
output contract stays fixed, so the /submit route never has to change.
"""

from __future__ import annotations

from typing import Any, Optional

from provenance import labels, scoring
from provenance.signals.llm import llm_signal
from provenance.signals.stylometry import stylometric_signal


def classify(text: str, *, llm_client: Optional[Any] = None) -> dict[str, Any]:
    """Run the full detection pipeline on ``text`` and return a decision dict.

    Keys: ``llm_score`` (signal 1), ``stylometric_score`` (signal 2),
    ``confidence`` (blended), ``attribution``, ``label``, plus diagnostics
    (``verdict``, ``llm_reasoning``, ``llm_error``, ``stylometric_detail``).

    Both signals run on the raw text; ``scoring.combine`` blends them with a
    disagreement pull toward uncertainty; the blended score selects the label.
    """
    llm = llm_signal(text, client=llm_client)
    llm_score = llm["ai_likelihood"]

    stylo = stylometric_signal(text)
    stylometric_score = stylo["ai_likelihood"]

    confidence = scoring.combine(llm_score, stylometric_score)

    attribution, label = labels.classify_label(confidence)
    return {
        "llm_score": llm_score,
        "stylometric_score": stylometric_score,
        "confidence": confidence,
        "attribution": attribution,
        "label": label,
        "verdict": llm["verdict"],
        "llm_reasoning": llm["reasoning"],
        "llm_error": llm["error"],
        "stylometric_detail": stylo["detail"],
    }
