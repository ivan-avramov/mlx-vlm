"""P369 round-6 F21: a zero-budget request reports no fabricated token / logprob.

Fork-only file. The continuous-batching step turns the zero-budget sentinel into a
``StreamingToken`` (token 0, logprob 0.0, ``token_count == 0``); the chat endpoints
built a logprob entry from it. Logprob entries exist only for real tokens, streaming
and not.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server
import mlx_vlm.server.generation as server_generation


def _sse_objects(body):
    out = []
    for line in body.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :].strip()
            if payload not in ("[DONE]", ""):
                out.append(json.loads(payload))
    return out


def _install(monkeypatch, items):
    class Fake:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, *a, **kw):
            return None

        def _cpu_preprocess(self, *a, **kw):
            return {"input_ids": mx.zeros((1, 4), dtype=mx.int32)}

        def generate(self, prompt=None, images=None, audio=None, args=None, **kw):
            ctx = server_generation.GenerationContext(
                uid=1, prompt_tokens=4, backend=server_generation.BACKEND_CACHED_SESSION
            )
            return ctx, iter(items)

    monkeypatch.setattr(server.runtime, "response_generator", Fake())
    return patch.object(
        server,
        "get_cached_model",
        return_value=(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(model_type="qwen2_vl"),
        ),
    )


ZERO = server_generation.StreamingToken(
    text="", token=0, logprobs=0.0, finish_reason="length", token_count=0
)
REAL = [
    server_generation.StreamingToken(
        text="Hi", token=5, logprobs=-0.5, finish_reason=None, token_count=1
    ),
    server_generation.StreamingToken(
        text="", token=6, logprobs=-0.25, finish_reason="length", token_count=1
    ),
]


@pytest.fixture
def client():
    with TestClient(server.app) as c:
        yield c


def _body(stream, max_tokens):
    return {
        "model": "demo",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": max_tokens,
        "logprobs": True,
        "stream": stream,
    }


def test_f21_non_streaming_zero_budget_has_no_logprob_entries(client, monkeypatch):
    with _install(monkeypatch, [ZERO]):
        r = client.post("/v1/chat/completions", json=_body(False, 0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["usage"]["completion_tokens"] == 0
    lp = body["choices"][0].get("logprobs")
    assert not lp or not lp.get("content")


def test_f21_streaming_zero_budget_has_no_logprob_entries(client, monkeypatch):
    with _install(monkeypatch, [ZERO]):
        r = client.post("/v1/chat/completions", json=_body(True, 0))
    assert r.status_code == 200, r.text
    for obj in _sse_objects(r.text):
        for choice in obj.get("choices", []):
            lp = choice.get("logprobs")
            assert not lp or not lp.get("content")


def test_f21_control_real_tokens_keep_their_logprobs(client, monkeypatch):
    with _install(monkeypatch, list(REAL)):
        r = client.post("/v1/chat/completions", json=_body(False, 2))
    assert r.status_code == 200, r.text
    content = r.json()["choices"][0]["logprobs"]["content"]
    assert [round(e["logprob"], 2) for e in content] == [-0.5, -0.25]
