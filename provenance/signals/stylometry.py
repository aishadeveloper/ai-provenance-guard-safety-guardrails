"""Signal 2 — stylometric heuristics (the *structural* signal).

Deterministic, pure-Python. Captures structural/lexical-habit uniformity: humans
write unevenly, AI trends smooth and regular. Three metrics, each a *distinct*
property so they aren't redundant (planning.md "Detection signals"):

1. **Function-word profile** — Burrows's Delta distance to a known-human vs
   known-AI reference profile. Function words are the strongest classical
   authorship signal because writers use them unconsciously; AI uses them at more
   uniform rates than an individual human's idiosyncratic pattern.
2. **Punctuation density** — punctuation marks per sentence.
3. **Sentence-length variance (burstiness)** — std of sentence lengths; a named
   pillar of AI-text detection (humans burst long/short, machines trend uniform).

Each metric is reduced to a 0–1 "how AI-like" sub-score; the signal score is
their equal-weight mean. Metrics 2 and 3 are mapped by linear interpolation
between the human-mean and AI-mean *anchors* computed from the bundled reference
corpora — so the direction of each metric is learned from data, not hardcoded.

Blind spot: polished/formal humans look "too clean" (false-positive risk);
lightly-edited AI drifts toward "human"; unreliable on very short text; blind to
meaning. That is exactly why it is paired with the semantic LLM signal.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from provenance.config import FUNCTION_WORDS

_WORD_RE = re.compile(r"[a-z']+")
_SENT_RE = re.compile(r"[.!?]+")
_PUNCT_CHARS = set(",;:-—")

_FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"
_NEUTRAL = 0.5
_MIN_WORDS = 15  # below this, metrics are too noisy to trust; we still return 0.5s


# --- low-level feature extraction ------------------------------------------

def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _function_word_rates(text: str) -> list[float]:
    """Rate (per word) of each configured function word, as a fixed-length vector."""
    words = _words(text)
    total = len(words)
    if total == 0:
        return [0.0] * len(FUNCTION_WORDS)
    counts = {w: 0 for w in FUNCTION_WORDS}
    for w in words:
        if w in counts:
            counts[w] += 1
    return [counts[w] / total for w in FUNCTION_WORDS]


def _punctuation_density(text: str) -> float:
    n_sent = len(_sentences(text))
    if n_sent == 0:
        return 0.0
    n_punct = sum(1 for ch in text if ch in _PUNCT_CHARS)
    return n_punct / n_sent


def _burstiness(text: str) -> Optional[float]:
    """Std of sentence lengths (in words). None if fewer than two sentences."""
    lengths = [len(_words(s)) for s in _sentences(text)]
    if len(lengths) < 2:
        return None
    return statistics.pstdev(lengths)


# --- reference model (Burrows's Delta basis + anchors) ----------------------

@dataclass
class ReferenceModel:
    fw_mean: list[float]          # per-word mean rate across all reference docs
    fw_std: list[float]           # per-word std (>=epsilon)
    fw_human_profile: list[float] # mean z-vector of human docs
    fw_ai_profile: list[float]    # mean z-vector of AI docs
    punct_human: float
    punct_ai: float
    burst_human: float
    burst_ai: float


def _zvector(rates: Sequence[float], mean: Sequence[float], std: Sequence[float]) -> list[float]:
    return [(r - m) / s for r, m, s in zip(rates, mean, std)]


def build_reference(human_texts: Sequence[str], ai_texts: Sequence[str]) -> ReferenceModel:
    """Compute the Delta standardization basis and per-class anchors."""
    all_texts = list(human_texts) + list(ai_texts)
    rate_vectors = [_function_word_rates(t) for t in all_texts]

    n_features = len(FUNCTION_WORDS)
    fw_mean, fw_std = [], []
    for j in range(n_features):
        column = [vec[j] for vec in rate_vectors]
        m = statistics.fmean(column)
        s = statistics.pstdev(column) if len(column) > 1 else 0.0
        fw_mean.append(m)
        fw_std.append(s if s > 1e-9 else 1e-9)  # guard constant features

    def class_profile(texts: Sequence[str]) -> list[float]:
        zvecs = [_zvector(_function_word_rates(t), fw_mean, fw_std) for t in texts]
        return [statistics.fmean(col) for col in zip(*zvecs)]

    return ReferenceModel(
        fw_mean=fw_mean,
        fw_std=fw_std,
        fw_human_profile=class_profile(human_texts),
        fw_ai_profile=class_profile(ai_texts),
        punct_human=statistics.fmean([_punctuation_density(t) for t in human_texts]),
        punct_ai=statistics.fmean([_punctuation_density(t) for t in ai_texts]),
        burst_human=statistics.fmean([b for t in human_texts if (b := _burstiness(t)) is not None]),
        burst_ai=statistics.fmean([b for t in ai_texts if (b := _burstiness(t)) is not None]),
    )


def _load_corpus(subdir: str) -> list[str]:
    folder = _FIXTURES / subdir
    return [p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.txt"))]


_DEFAULT_REFERENCE: Optional[ReferenceModel] = None


def default_reference() -> ReferenceModel:
    """Lazily build (and cache) the reference model from the bundled corpora."""
    global _DEFAULT_REFERENCE
    if _DEFAULT_REFERENCE is None:
        _DEFAULT_REFERENCE = build_reference(_load_corpus("human"), _load_corpus("ai"))
    return _DEFAULT_REFERENCE


# --- metric -> sub-score ----------------------------------------------------

def _interp(x: float, human_anchor: float, ai_anchor: float) -> float:
    """Map x to 0–1 by where it sits between the human and AI anchors (clamped).

    0 = at/closer-than the human anchor, 1 = at/beyond the AI anchor. Works in
    either direction since it's derived from the two data-driven anchors.
    """
    if math.isclose(human_anchor, ai_anchor):
        return _NEUTRAL
    return max(0.0, min(1.0, (x - human_anchor) / (ai_anchor - human_anchor)))


def _delta_subscore(text: str, ref: ReferenceModel) -> float:
    z = _zvector(_function_word_rates(text), ref.fw_mean, ref.fw_std)
    d_human = statistics.fmean(abs(zi - hi) for zi, hi in zip(z, ref.fw_human_profile))
    d_ai = statistics.fmean(abs(zi - ai) for zi, ai in zip(z, ref.fw_ai_profile))
    if d_human + d_ai == 0:
        return _NEUTRAL
    return d_human / (d_human + d_ai)  # closer to AI profile -> higher


def stylometric_signal(text: str, *, reference: Optional[ReferenceModel] = None) -> dict[str, Any]:
    """Return ``{ai_likelihood, detail}`` for the stylometry signal.

    ``ai_likelihood`` is the equal-weight mean of the three sub-scores. On text
    too short to measure reliably, sub-scores fall back to 0.5 (uncertain) rather
    than producing a confident-but-meaningless number.
    """
    ref = reference or default_reference()
    words = _words(text)

    if len(words) < _MIN_WORDS:
        detail = {"function_words": _NEUTRAL, "punctuation": _NEUTRAL, "burstiness": _NEUTRAL,
                  "note": f"text shorter than {_MIN_WORDS} words; stylometry unreliable"}
        return {"ai_likelihood": _NEUTRAL, "detail": detail}

    fw = _delta_subscore(text, ref)
    punct = _interp(_punctuation_density(text), ref.punct_human, ref.punct_ai)

    burst_raw = _burstiness(text)
    if burst_raw is None:
        burst = _NEUTRAL  # single sentence: can't measure variance
    else:
        # high variance = human, low = AI, so AI-likeness rises as variance falls
        burst = _interp(burst_raw, ref.burst_human, ref.burst_ai)

    combined = statistics.fmean([fw, punct, burst])
    detail = {"function_words": round(fw, 4), "punctuation": round(punct, 4),
              "burstiness": round(burst, 4)}
    return {"ai_likelihood": combined, "detail": detail}
