"""
Deterministic offline stand-in for an OpenAI-compatible client.

Select it with ``api_provider="mock"`` (no API key, no network) to exercise the full
GSAM / ACE pipelines end to end. Every call returns the same well-formed JSON object
that satisfies the generator, reflector and curator output parsers.
"""
import json
from types import SimpleNamespace

DEFAULT_REPLY = {
    "reasoning": "mock reasoning",
    "final_answer": "mock",
    "node_ids": [],
    "bullet_ids": [],
    "error_identification": "mock",
    "root_cause_analysis": "mock",
    "correct_approach": "mock",
    "key_insight": "mock insight",
    "bullet_tags": [],
    "node_tags": [],
    "operations": [],
}


class _Completions:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        content = json.dumps(self.reply)
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        return SimpleNamespace(choices=[choice], usage=usage)


class MockLLMClient:
    """Minimal ``client.chat.completions.create`` implementation."""

    def __init__(self, reply=None):
        self.completions = _Completions(dict(reply or DEFAULT_REPLY))
        self.chat = SimpleNamespace(completions=self.completions)
