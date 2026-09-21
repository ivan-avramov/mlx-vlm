"""Fork-only (C99, after the C91 cold review): every usage consumer must read
``StreamingToken.token_count`` rather than count chunks.

Drives the REAL ``_process_cached_request`` with the saved C89 calibration shape
(one generated token, then a finalization chunk that repeats ``generation_tokens=1``
with ``finish_reason='length'``) through the HTTP endpoints and asserts that the
reported completion/output tokens are exactly 1 on every endpoint, streaming and
not. Before C99, ``/v1/responses`` and ``/v1/messages`` non-streaming summed one
per chunk and reported 2 — the same defect C91 fixed at the bridge.
"""

import json
import sys
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server
import mlx_vlm.server.generation as server_generation
from mlx_vlm.generate.common import GenerationResult

from test_cached_tokens_reporting import _bare_response_generator

PROMPT_TOKENS = 3210  # the C89 calibration prompt


def _chunk(text, token, generation_tokens, finish_reason=None):
    return GenerationResult(
        text=text,
        token=token,
        logprobs=None,
        prompt_tokens=PROMPT_TOKENS,
        generation_tokens=generation_tokens,
        total_tokens=PROMPT_TOKENS + generation_tokens,
        prompt_tps=100.0,
        generation_tps=50.0,
        peak_memory=1.0,
        cached_tokens=0,
        finish_reason=finish_reason,
    )


CALIBRATION_SHAPE = [_chunk("The", 100, 1), _chunk("", 100, 1, "length")]


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.fixture
def calibration_generator(monkeypatch):
    rg = _bare_response_generator()

    def _gen(**kwargs):
        yield from CALIBRATION_SHAPE

    monkeypatch.setattr(sys.modules["mlx_vlm.generate"], "stream_generate", _gen)

    class CachedPathResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, *a, **kw):
            return None

        def _cpu_preprocess(self, *a, **kw):
            return {"input_ids": mx.zeros((1, PROMPT_TOKENS), dtype=mx.int32)}

        def generate(self, prompt=None, images=None, audio=None, args=None, **kw):
            rqueue: Queue = Queue()
            rg._process_cached_request(
                rqueue=rqueue,
                prompt=prompt or "hello",
                images=None,
                args=args or server.GenerationArguments(),
                prompt_tokens=PROMPT_TOKENS,
                prompt_cache_state=SimpleNamespace(),
            )
            ctx = rqueue.get_nowait()
            items = []
            while True:
                item = rqueue.get_nowait()
                if item is None:
                    break
                if isinstance(item, server_generation.KeepAlive):
                    continue
                items.append(item)
            return ctx, iter(items)

    monkeypatch.setattr(server.runtime, "response_generator", CachedPathResponseGenerator())
    with patch.object(
        server,
        "get_cached_model",
        return_value=(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(model_type="qwen2_vl")),
    ):
        yield


def _sse_objects(body):
    out = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):].strip()
        if payload in ("[DONE]", ""):
            continue
        out.append(json.loads(payload))
    return out


@pytest.mark.usefixtures("calibration_generator")
class TestEveryUsageConsumerReportsOneToken:
    def test_chat_completions_non_streaming(self, client):
        r = client.post("/v1/chat/completions", json={"model": "demo", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1})
        assert r.status_code == 200, r.text
        assert r.json()["usage"]["completion_tokens"] == 1
        assert r.json()["choices"][0]["finish_reason"] == "length"

    def test_chat_completions_streaming(self, client):
        r = client.post("/v1/chat/completions", json={"model": "demo", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1, "stream": True, "stream_options": {"include_usage": True}})
        assert r.status_code == 200, r.text
        usages = [o["usage"] for o in _sse_objects(r.read().decode()) if o.get("usage")]
        assert usages and usages[-1]["completion_tokens"] == 1

    def test_completions_non_streaming(self, client):
        r = client.post("/v1/completions", json={"model": "demo", "prompt": "Hi", "max_tokens": 1})
        assert r.status_code == 200, r.text
        assert r.json()["usage"]["completion_tokens"] == 1

    def test_responses_non_streaming(self, client):
        r = client.post("/v1/responses", json={"model": "demo", "input": "Hi", "max_output_tokens": 1})
        assert r.status_code == 200, r.text
        assert r.json()["usage"]["output_tokens"] == 1

    def test_responses_streaming(self, client):
        r = client.post("/v1/responses", json={"model": "demo", "input": "Hi", "max_output_tokens": 1, "stream": True})
        assert r.status_code == 200, r.text
        outs = [((o.get("response") or {}).get("usage") or {}).get("output_tokens") for o in _sse_objects(r.read().decode())]
        outs = [o for o in outs if o is not None]
        assert outs and outs[-1] == 1

    def test_messages_non_streaming(self, client):
        r = client.post("/v1/messages", json={"model": "demo", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1})
        assert r.status_code == 200, r.text
        assert r.json()["usage"]["output_tokens"] == 1

    def test_messages_streaming(self, client):
        r = client.post("/v1/messages", json={"model": "demo", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1, "stream": True})
        assert r.status_code == 200, r.text
        outs = [(o.get("usage") or {}).get("output_tokens") for o in _sse_objects(r.read().decode())]
        outs = [o for o in outs if o is not None]
        assert outs and outs[-1] == 1
