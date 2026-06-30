"""Flask app factory and HTTP routes.

Uses an app factory so tests can inject a temp DB path, a stubbed LLM client, and
toggle/override rate limiting. Routes:

- ``POST /submit``  classify text -> {content_id, attribution, confidence, label}
- ``POST /appeal``  contest a classification -> status 'under_review', logged
- ``GET  /log``     recent structured audit entries (documentation/grading)
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from provenance import analytics, audit
from provenance.config import DEFAULT_DB_PATH, SUBMIT_RATE_LIMITS
from provenance.pipeline import classify


def create_app(
    *,
    db_path: Optional[str] = None,
    llm_client: Optional[Any] = None,
    testing: bool = False,
    ratelimit_enabled: bool = True,
    submit_rate_limits: Optional[str] = None,
) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = testing
    app.config["RATELIMIT_ENABLED"] = ratelimit_enabled

    db_path = db_path or os.environ.get("PROVENANCE_DB", DEFAULT_DB_PATH)
    app.config["PROVENANCE_DB"] = db_path
    audit.init_db(db_path)

    # Rate limiting (planning.md): /submit is an expensive LLM-backed write, so it
    # sits in the strict tier. Keyed per client IP via in-memory storage.
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[],
        storage_uri="memory://",
    )
    # When RATELIMIT_ENABLED is False, Flask-Limiter's init_app returns early and
    # does NOT register the limiter in app.extensions, so without our own
    # reference the limiter is garbage-collected and the route's weakref to it
    # dies (ReferenceError at request time). Keep it alive for the app's lifetime.
    app.extensions["provenance_limiter"] = limiter
    submit_limits = submit_rate_limits or SUBMIT_RATE_LIMITS

    @app.post("/submit")
    @limiter.limit(submit_limits)
    def submit():
        body = request.get_json(silent=True) or {}
        text = body.get("text")
        creator_id = body.get("creator_id")

        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Field 'text' is required and must be a non-empty string."}), 400
        if not isinstance(creator_id, str) or not creator_id.strip():
            return jsonify({"error": "Field 'creator_id' is required and must be a non-empty string."}), 400

        result = classify(text, llm_client=llm_client)
        content_id = str(uuid.uuid4())

        audit.log_classification(
            db_path,
            content_id=content_id,
            creator_id=creator_id,
            attribution=result["attribution"],
            confidence=result["confidence"],
            llm_score=result["llm_score"],
            stylometric_score=result["stylometric_score"],
            text_snippet=text,
        )

        return (
            jsonify(
                {
                    "content_id": content_id,
                    "attribution": result["attribution"],
                    "confidence": round(result["confidence"], 4),
                    "label": result["label"],
                    "signals": result["signals"],  # per-member ensemble breakdown
                }
            ),
            200,
        )

    @app.post("/appeal")
    def appeal():
        body = request.get_json(silent=True) or {}
        content_id = body.get("content_id")
        creator_reasoning = body.get("creator_reasoning")

        if not isinstance(content_id, str) or not content_id.strip():
            return jsonify({"error": "Field 'content_id' is required."}), 400
        if not isinstance(creator_reasoning, str) or not creator_reasoning.strip():
            return jsonify({"error": "Field 'creator_reasoning' is required and must be a non-empty string."}), 400

        original = audit.latest_for_content(db_path, content_id)
        if original is None:
            return jsonify({"error": f"No content found with id '{content_id}'."}), 404

        audit.log_appeal(
            db_path,
            content_id=content_id,
            creator_id=original.get("creator_id"),
            appeal_reasoning=creator_reasoning,
            original=original,
        )

        return (
            jsonify(
                {
                    "content_id": content_id,
                    "status": "under_review",
                    "message": "Appeal received. This content has been logged for human review.",
                }
            ),
            200,
        )

    @app.get("/log")
    def get_log():
        limit = request.args.get("limit", default=50, type=int)
        return jsonify({"entries": audit.get_recent(db_path, limit=limit)}), 200

    @app.get("/analytics")
    def get_analytics():
        # Aggregate dashboard metrics over the audit log (stretch feature).
        return jsonify(analytics.compute(db_path)), 200

    @app.errorhandler(429)
    def ratelimit_handler(exc):
        # Return structured JSON (not Flask-Limiter's default HTML) so clients and
        # the README evidence see a clean 429 body.
        return (
            jsonify(
                {
                    "error": "rate_limit_exceeded",
                    "message": f"Submission rate limit exceeded ({exc.description}). Please slow down.",
                }
            ),
            429,
        )

    return app
