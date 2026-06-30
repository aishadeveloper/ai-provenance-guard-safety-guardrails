"""Central tunables for Provenance Guard.

Every "magic number" in the system lives here so the calibration story is in one
place and the README can point at it. Values trace directly to ``planning.md``.
"""

# --- Attribution thresholds (planning.md "Transparency labels") -------------
# Bands are contiguous and asymmetric on purpose: the bar to call something
# "likely AI" (0.70) sits further from the middle than the "likely human" bar
# (0.40), giving human writers more room before any accusation, because a false
# positive (accusing a real person) is the costliest error on a creative platform.
#
#   score  < 0.40            -> likely_human
#   0.40 <= score < 0.70     -> uncertain
#   score >= 0.70            -> likely_ai
LIKELY_HUMAN_MAX = 0.40
LIKELY_AI_MIN = 0.70

# --- Signal blend weights (planning.md "Confidence scoring") ----------------
# The LLM signal is the more holistic judge but is uncalibrated/overconfident;
# stylometry is deterministic but weak on short text. We lean slightly on the
# LLM and use a disagreement pull (below) to stay honest when they conflict.
W_LLM = 0.60
W_STYLO = 0.40

# When the two signals disagree, pull the blended score toward 0.5 (uncertain)
# rather than reporting a confident-looking average. 0 = ignore disagreement,
# 1 = a fully-disagreeing pair lands exactly on 0.5.
DISAGREEMENT_PULL = 0.6

# --- Groq (Signal 1) --------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"

# Cap the text sent to the LLM signal. Signal 2 (stylometry) still analyzes the
# FULL text — only the Groq input is truncated. This bounds input tokens per
# /submit so a long document can't blow Groq's per-minute token limit (12,000
# tokens/min on the free tier as of 2026-06-29; limits may change — see the Groq
# console). ~1,200 words is roughly ~1,500 tokens, comfortably within one window.
LLM_MAX_INPUT_WORDS = 1200

# --- Stylometry (Signal 2) --------------------------------------------------
# A fixed, curated list of common English function words used as the Burrows's
# Delta basis. Kept on the lower end (~40) because the platform sees short
# pieces (poems, excerpts) where rarer words give unstable rates. The SAME list
# is used every time so per-text rates are comparable. (planning.md, Signal 2.)
FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "it",
    "for", "on", "with", "as", "but", "his", "they", "at", "be", "this",
    "from", "or", "by", "an", "not", "are", "we", "you", "he", "she",
    "which", "their", "all", "will", "would", "there", "what", "so", "if", "more",
]

# Database location for the audit log (overridable via the PROVENANCE_DB env var
# or the app factory). Gitignored — see .gitignore.
DEFAULT_DB_PATH = "audit_log.db"

# --- Rate limiting (planning.md "Rate limiting") ----------------------------
# /submit is an expensive write (it calls Groq), so it sits in the strict tier.
# 10/min = burst protection; 100/day = sustained-abuse + Groq-cost cap.
SUBMIT_RATE_LIMITS = "10 per minute;100 per day"
