"""UNIT TESTS — Signal 1 (LLM classification), provenance.signals.llm.

Scope: the pure parsing/normalization/fallback logic of ``llm_signal`` in
isolation, using a stub client. No network, no Flask. Verifies the contract the
rest of the system relies on: a dict with a clamped ai_likelihood, and graceful
degradation to a neutral 0.5 on every failure mode (an unavailable signal must
never become a confident accusation).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from provenance.signals.llm import llm_signal
from tests.conftest import FakeGroqClient


def test_parses_valid_json_response():
    client = FakeGroqClient('{"verdict": "ai", "ai_likelihood": 0.83, "reasoning": "uniform"}')
    result = llm_signal("some text", client=client)
    assert result["verdict"] == "ai"
    assert result["ai_likelihood"] == 0.83
    assert result["reasoning"] == "uniform"
    assert result["error"] is None


def test_clamps_out_of_range_likelihood():
    client = FakeGroqClient('{"verdict": "ai", "ai_likelihood": 1.7, "reasoning": "x"}')
    result = llm_signal("t", client=client)
    assert result["ai_likelihood"] == 1.0


def test_derives_likelihood_from_verdict_when_number_missing():
    client = FakeGroqClient('{"verdict": "human", "reasoning": "no number here"}')
    result = llm_signal("t", client=client)
    assert result["ai_likelihood"] == pytest.approx(0.2)
    assert result["verdict"] == "human"


def test_malformed_json_falls_back_to_neutral():
    client = FakeGroqClient("this is not json at all")
    result = llm_signal("t", client=client)
    assert result["ai_likelihood"] == 0.5
    assert result["verdict"] == "uncertain"
    assert result["error"] == "malformed_json"


def test_non_object_json_falls_back_to_neutral():
    client = FakeGroqClient("[1, 2, 3]")  # valid JSON, wrong shape
    result = llm_signal("t", client=client)
    assert result["ai_likelihood"] == 0.5
    assert result["error"] == "bad_shape"


def test_api_exception_falls_back_to_neutral():
    client = FakeGroqClient(raise_exc=RuntimeError("network down"))
    result = llm_signal("t", client=client)
    assert result["ai_likelihood"] == 0.5
    assert result["error"] == "RuntimeError"


def test_no_client_and_no_api_key_returns_neutral(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = llm_signal("t", client=None)
    assert result["ai_likelihood"] == 0.5
    assert result["error"] == "no_api_key"


def test_long_input_is_truncated_before_sending_to_groq():
    """Bounds Groq input tokens: only the first LLM_MAX_INPUT_WORDS words are sent."""
    from provenance.config import LLM_MAX_INPUT_WORDS

    long_text = " ".join(["word"] * (LLM_MAX_INPUT_WORDS + 500))
    client = FakeGroqClient('{"verdict": "ai", "ai_likelihood": 0.6, "reasoning": "x"}')
    llm_signal(long_text, client=client)

    sent = client.last_kwargs["messages"][1]["content"]  # the user message
    assert len(sent.split()) == LLM_MAX_INPUT_WORDS


def test_short_input_is_sent_unchanged():
    client = FakeGroqClient('{"verdict": "human", "ai_likelihood": 0.2, "reasoning": "x"}')
    llm_signal("just a few words here", client=client)
    assert client.last_kwargs["messages"][1]["content"] == "just a few words here"
