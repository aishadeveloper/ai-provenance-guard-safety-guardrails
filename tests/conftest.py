"""Shared test fixtures and stubs.

A ``FakeGroqClient`` lets unit/integration tests exercise the LLM signal with no
network and fully deterministic output: it mimics the slice of the Groq client we
use (``client.chat.completions.create(...).choices[0].message.content``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from provenance.app import create_app


class FakeGroqClient:
    """Minimal stand-in for groq.Groq returning canned JSON content.

    Pass ``content`` (a JSON string the model would have returned) or set
    ``raise_exc`` to simulate an API failure.
    """

    def __init__(self, content: str = '{"verdict": "human", "ai_likelihood": 0.2, "reasoning": "stub"}',
                 raise_exc: Exception | None = None):
        self._content = content
        self._raise = raise_exc
        # the client exposes .chat.completions.create — point both at self
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        if self._raise is not None:
            raise self._raise
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def db_path(tmp_path):
    """An isolated, throwaway SQLite path for each test."""
    return str(tmp_path / "audit_test.db")


@pytest.fixture
def fake_human_client():
    return FakeGroqClient('{"verdict": "human", "ai_likelihood": 0.15, "reasoning": "casual, irregular"}')


@pytest.fixture
def fake_ai_client():
    return FakeGroqClient('{"verdict": "ai", "ai_likelihood": 0.9, "reasoning": "uniform, polished"}')


@pytest.fixture
def client(db_path, fake_human_client):
    """A Flask test client wired to a temp DB and a deterministic (human) LLM stub."""
    app = create_app(db_path=db_path, llm_client=fake_human_client, testing=True)
    return app.test_client()
