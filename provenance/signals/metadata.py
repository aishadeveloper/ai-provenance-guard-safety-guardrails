"""Multi-modal (stretch) — image *provenance metadata* signals.

A second content type beyond text: structured image metadata, attributed with
signals appropriate to images (not stylometry) and combined with the SAME
``scoring.combine_ensemble`` used for text. Three signals, each a 0–1 AI-likelihood:

1. **generator_signature** — metadata names a known AI image generator, or carries a
   C2PA ``trainedAlgorithmicMedia`` tag. Direct, high-precision when present.
2. **camera_fingerprint** — presence of real-camera EXIF (make/model/exposure/GPS).
   Real photos carry it; AI images usually don't. Blind spot: stripped metadata on a
   real photo looks AI here, so it is never decisive alone.
3. **content_credential** — a C2PA / Content Credentials assertion of AI generation
   vs camera capture.

See planning.md "Multi-modal support".
"""

from __future__ import annotations

from typing import Any, Iterator

from provenance import scoring
from provenance.config import (
    AI_IMAGE_GENERATORS,
    CAMERA_EXIF_KEYS,
    METADATA_ENSEMBLE_WEIGHTS,
)

_NEUTRAL = 0.5


def _walk(obj: Any) -> Iterator[tuple[str, Any]]:
    """Yield (key, value) for every key in a nested dict/list structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k), v
            yield from _walk(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def _all_strings(obj: Any) -> str:
    """All string values/keys in the structure, lowercased and concatenated."""
    parts: list[str] = []
    for k, v in _walk(obj):
        parts.append(k)
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts).lower()


def _present_keys(obj: Any) -> set[str]:
    """Lowercased, separator-stripped set of all keys present (nested included)."""
    return {k.lower().replace("_", "").replace(" ", "") for k, _ in _walk(obj)}


def generator_signature(metadata: dict[str, Any]) -> float:
    blob = _all_strings(metadata)
    if "trainedalgorithmicmedia" in blob.replace(" ", "").replace("_", ""):
        return 0.97
    if any(name in blob for name in AI_IMAGE_GENERATORS):
        return 0.95
    return _NEUTRAL  # no generator evidence either way


def camera_fingerprint(metadata: dict[str, Any]) -> float:
    keys = _present_keys(metadata)
    n = len(CAMERA_EXIF_KEYS & keys)
    if n >= 4:
        return 0.12  # rich camera fingerprint -> almost certainly a real photo
    if n >= 2:
        return 0.30
    if n == 1:
        return 0.45
    # No camera fields: ABSTAIN (return neutral). Absence isn't positive AI
    # evidence — a metadata-stripped real photo would otherwise be wrongly flagged,
    # the exact false-positive this project avoids.
    return _NEUTRAL


def content_credential(metadata: dict[str, Any]) -> float:
    cc = None
    for k, v in _walk(metadata):
        if k.lower().replace("_", "").replace(" ", "") in ("c2pa", "contentcredentials", "contentcredential"):
            cc = v
            break
    if cc is None:
        return _NEUTRAL

    blob = _all_strings(cc).replace(" ", "").replace("_", "")
    # explicit boolean assertion
    for k, v in _walk(cc):
        if k.lower().replace("_", "").replace(" ", "") in ("aigenerated", "isaigenerated"):
            return 0.97 if v else 0.05
    if "trainedalgorithmicmedia" in blob or any(g.replace(" ", "") in blob for g in AI_IMAGE_GENERATORS):
        return 0.97
    if "digitalcapture" in blob or "camera" in blob:
        return 0.05
    return _NEUTRAL


def metadata_signal(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return ``{ai_likelihood, signals}`` for the image-metadata modality.

    Signals that find no evidence return ``_NEUTRAL`` (0.5) and **abstain** — only
    the informative signals vote in the ensemble. If every signal abstains (e.g.
    empty or signal-free metadata), the verdict is 0.5 (uncertain). The per-signal
    breakdown still reports all three for transparency.
    """
    members = {
        "generator_signature": generator_signature(metadata),
        "camera_fingerprint": camera_fingerprint(metadata),
        "content_credential": content_credential(metadata),
    }
    informative = [
        (score, METADATA_ENSEMBLE_WEIGHTS[name])
        for name, score in members.items()
        if score != _NEUTRAL
    ]
    confidence = scoring.combine_ensemble(informative) if informative else _NEUTRAL
    return {
        "ai_likelihood": confidence,
        "signals": {name: round(score, 4) for name, score in members.items()},
    }
