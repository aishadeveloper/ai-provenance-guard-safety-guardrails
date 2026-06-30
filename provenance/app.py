"""Flask app factory and HTTP routes.

Uses an app factory so tests can inject a temp DB path and a stubbed LLM client.
Routes implemented in M3: ``POST /submit`` and ``GET /log``. ``POST /appeal`` and
rate limiting are added in M5.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

from flask import Flask, jsonify, request

from provenance import audit
from provenance.config import DEFAULT_DB_PATH
from provenance.pipeline import classify


def create_app(
    *,
    db_path: Optional[str] = None,
    llm_client: Optional[Any] = None,
    testing: bool = False,
) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = testing

    db_path = db_path or os.environ.get("PROVENANCE_DB", DEFAULT_DB_PATH)
    app.config["PROVENANCE_DB"] = db_path
    audit.init_db(db_path)

    @app.post("/submit")
    def submit():
        body = request.get_json(silent=True) or {}
        text = body.get("text")
        creator_id = body.get("creator_id")

        if not isinstance(text, str) or not text.strip():
            return (
                jsonify({"error": "Field 'text' is required and must be a non-empty string."}),
                400,
            )
        if not isinstance(creator_id, str) or not creator_id.strip():
            return (
                jsonify({"error": "Field 'creator_id' is required and must be a non-empty string."}),
                400,
            )

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
                }
            ),
            200,
        )

    @app.get("/log")
    def get_log():
        limit = request.args.get("limit", default=50, type=int)
        return jsonify({"entries": audit.get_recent(db_path, limit=limit)}), 200

    return app
