"""Flask app factory and HTTP routes.

Uses an app factory so tests can inject a temp DB path, a stubbed LLM client, and
toggle/override rate limiting. Routes:

- ``POST /submit``  classify text -> {content_id, attribution, confidence, label}
- ``POST /appeal``  contest a classification -> status 'under_review', logged
- ``GET  /log``     recent structured audit entries (documentation/grading)
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Optional

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from provenance import analytics, audit, certificates
from provenance.config import DEFAULT_DB_PATH, SUBMIT_RATE_LIMITS
from provenance.db import Database
from provenance.pipeline import classify, classify_metadata


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
    # One Database owns connection handling for the whole app; data-access modules
    # receive it rather than each opening their own connection.
    db = Database(db_path)
    audit.init_db(db)
    certificates.init_db(db)

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
        creator_id = body.get("creator_id")
        content_type = (body.get("content_type") or "text").lower()

        if not isinstance(creator_id, str) or not creator_id.strip():
            return jsonify({"error": "Field 'creator_id' is required and must be a non-empty string."}), 400

        # Dispatch on content type — the same endpoint handles text and the
        # multi-modal image-metadata path (stretch). Both return the same shape.
        if content_type == "text":
            text = body.get("text")
            if not isinstance(text, str) or not text.strip():
                return jsonify({"error": "Field 'text' is required and must be a non-empty string."}), 400
            result = classify(text, llm_client=llm_client)
            snippet = text
            llm_score, stylometric_score = result["llm_score"], result["stylometric_score"]
        elif content_type == "metadata":
            metadata = body.get("metadata")
            if not isinstance(metadata, dict) or not metadata:
                return jsonify({"error": "Field 'metadata' is required and must be a non-empty object."}), 400
            result = classify_metadata(metadata)
            snippet = json.dumps(metadata, sort_keys=True)
            llm_score, stylometric_score = None, None
        else:
            return jsonify({"error": f"Unsupported content_type '{content_type}'. Use 'text' or 'metadata'."}), 400

        content_id = str(uuid.uuid4())

        # Provenance certificate (stretch): a verified creator's content carries a
        # distinct credential badge, separate from the content's transparency label.
        cert = certificates.get_certificate(db, creator_id)
        if cert:
            provenance = {
                "verified_human_creator": True,
                "badge": "✓ Verified Human Creator",
                "statement": cert["statement"],
            }
            # baseline consistency is a stylometric (text) notion only
            if content_type == "text":
                members = {k: result["signals"][k] for k in ("function_words", "punctuation", "burstiness")}
                provenance["baseline_consistency"] = certificates.baseline_consistency(cert["baseline"], members)
        else:
            provenance = {"verified_human_creator": False}

        audit.log_classification(
            db,
            content_id=content_id,
            creator_id=creator_id,
            attribution=result["attribution"],
            confidence=result["confidence"],
            llm_score=llm_score,
            stylometric_score=stylometric_score,
            text_snippet=snippet,
            content_type=content_type,
        )

        return (
            jsonify(
                {
                    "content_id": content_id,
                    "content_type": content_type,
                    "attribution": result["attribution"],
                    "confidence": round(result["confidence"], 4),
                    "label": result["label"],
                    "signals": result["signals"],  # per-signal breakdown (modality-specific)
                    "provenance": provenance,       # verified-creator credential
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

        # Read-the-original and write-the-appeal happen atomically in one
        # transaction inside record_appeal; returns None if the content is unknown.
        original = audit.record_appeal(
            db,
            content_id=content_id,
            appeal_reasoning=creator_reasoning,
        )
        if original is None:
            return jsonify({"error": f"No content found with id '{content_id}'."}), 404

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

    @app.post("/verify")
    def verify():
        body = request.get_json(silent=True) or {}
        try:
            certificate = certificates.enroll(
                db,
                body.get("creator_id"),
                body.get("samples"),
                pledge_accepted=bool(body.get("pledge_accepted", False)),
            )
        except certificates.EnrollmentError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(certificate), 200

    @app.get("/certificate")
    def get_cert():
        creator_id = request.args.get("creator_id", type=str)
        if not creator_id:
            return jsonify({"error": "Query param 'creator_id' is required."}), 400
        cert = certificates.get_certificate(db, creator_id)
        if cert is None:
            return jsonify({"verified_human_creator": False, "creator_id": creator_id}), 404
        return jsonify(cert), 200

    @app.get("/log")
    def get_log():
        limit = request.args.get("limit", default=50, type=int)
        return jsonify({"entries": audit.get_recent(db, limit=limit)}), 200

    @app.get("/analytics")
    def get_analytics():
        # Aggregate dashboard metrics over the audit log (stretch feature).
        return jsonify(analytics.compute(db)), 200

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
