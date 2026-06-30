"""Provenance Guard — multi-signal AI-attribution backend.

See ``planning.md`` for the full design. Package layout:

- ``config``            tunable thresholds, weights, and the function-word list
- ``signals.llm``       Signal 1 — Groq LLM classification (semantic)
- ``signals.stylometry``Signal 2 — stylometric heuristics (structural)
- ``scoring``           blends the two signals into one calibrated score
- ``labels``            maps a score to a transparency-label variant + text
- ``audit``             append-only SQLite audit log
- ``pipeline``          orchestrates signals → scoring → label
- ``app``               Flask app factory and HTTP routes
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
