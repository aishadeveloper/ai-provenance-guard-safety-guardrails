"""Classification pipeline — orchestrates signals -> scoring -> label.

This is the stable seam the HTTP layer depends on. Through the milestones its
internals grow (M3: Signal 1 only; M4: + Signal 2 and blended scoring) while its
output contract stays fixed, so the /submit route never has to change.
"""

from __future__ import annotations

from typing import Any, Optional

from provenance import labels
from provenance.signals.llm import llm_signal


def classify(text: str, *, llm_client: Optional[Any] = None) -> dict[str, Any]:
    """Run the detection pipeline on ``text`` and return a decision dict.

    Keys: ``llm_score``, ``stylometric_score``, ``confidence``, ``attribution``,
    ``label``, plus LLM diagnostics (``verdict``, ``llm_reasoning``,
    ``llm_error``).

    M3 status: confidence is **signal-1-only** — it is exactly the LLM's
    AI-likelihood. The stylometry signal and the blended score land in M4, at
    which point ``stylometric_score`` stops being ``None`` and ``confidence``
    becomes the blend.
    """
    llm = llm_signal(text, client=llm_client)
    llm_score = llm["ai_likelihood"]

    stylometric_score: Optional[float] = None
    confidence = llm_score

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
    }
