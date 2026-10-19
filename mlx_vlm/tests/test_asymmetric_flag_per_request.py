"""P369 round-6 F23: the asymmetric-rendering flag rides on the request.

Fork-only file. The chat handler used to store ``is_asymmetric_rendering`` on the
shared ``PromptCacheState`` before the GPU worker ran the request, so a second queued
request on the same chat could change the first one's publication policy. The flag is
now carried on the queued request and forwarded to ``stream_generate`` as a keyword;
the session object is never mutated by the handler. Round 7: an explicit per-request
flag (``True`` or ``False``) is authoritative; the session's legacy attribute is read
only when the request carries none.
"""

import importlib
import sys
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient
from test_cached_tokens_reporting import _bare_response_generator
from test_mtp_exactness import PROMPT, fake_model, hybrid_lm
from test_mtp_stop_state import turn
from test_zero_budget_logprobs import REAL

import mlx_vlm.server as server
import mlx_vlm.server.generation as server_generation
from mlx_vlm.generate.common import PromptCacheState
from mlx_vlm.server import session_manager

server_openai = importlib.import_module("mlx_vlm.server.openai")
_dispatch = importlib.import_module("mlx_vlm.generate.dispatch")


def test_f23_queued_request_carries_the_flag():
    assert (
        "is_asymmetric_rendering"
        in server_generation.QueuedGenerationRequest.__dataclass_fields__
    )


def test_f23_process_cached_request_forwards_the_flag_and_leaves_the_state_alone(
    monkeypatch,
):
    seen = []

    def fake_stream(**kwargs):
        seen.append(kwargs.get("is_asymmetric_rendering"))
        return iter(())

    monkeypatch.setattr(
        sys.modules["mlx_vlm.generate"],
        "stream_generate",
        lambda **kw: fake_stream(**kw),
    )
    rg = _bare_response_generator()
    state = SimpleNamespace(is_asymmetric_rendering=False)
    for flag in (True, False):
        rqueue = Queue()
        rg._process_cached_request(
            rqueue=rqueue,
            prompt="hi",
            images=None,
            args=server.GenerationArguments(),
            prompt_tokens=1,
            prompt_cache_state=state,
            is_asymmetric_rendering=flag,
        )
    assert seen == [True, False]
    assert state.is_asymmetric_rendering is False


def test_f23_handler_does_not_mutate_the_shared_session(monkeypatch):
    monkeypatch.setattr(session_manager, "_session_cache_max", 8)
    session_manager.clear_session_caches()
    calls = []

    class Fake:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, *a, **kw):
            return None

        def _cpu_preprocess(self, *a, **kw):
            return {"input_ids": mx.zeros((1, 4), dtype=mx.int32)}

        def generate(self, prompt=None, images=None, audio=None, args=None, **kw):
            calls.append(
                (
                    kw.get("is_asymmetric_rendering"),
                    getattr(
                        kw.get("prompt_cache_state"), "is_asymmetric_rendering", None
                    ),
                )
            )
            ctx = server_generation.GenerationContext(
                uid=1, prompt_tokens=4, backend=server_generation.BACKEND_CACHED_SESSION
            )
            return ctx, iter(list(REAL))

    monkeypatch.setattr(server.runtime, "response_generator", Fake())
    answers = iter([True, False])
    with (
        patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ),
        patch.object(
            server_openai,
            "_is_template_thinking_asymmetric",
            side_effect=lambda *_: next(answers),
        ),
        TestClient(server.app) as client,
    ):
        for _ in range(2):
            r = client.post(
                "/v1/chat/completions",
                headers={"X-MLX-VLM-Chat-Id": "p369-f23"},
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 2,
                },
            )
            assert r.status_code == 200, r.text
    session_manager.clear_session_caches()
    # Each request carries ITS OWN flag; the shared session never held either.
    assert [flag for flag, _ in calls] == [True, False]
    assert all(state_flag is False for _, state_flag in calls)


# ------------------------------------------------------------------ round 7 / F23
@pytest.mark.parametrize(
    ("explicit", "legacy", "asymmetric"),
    [
        (False, True, False),  # an explicit False wins over a stale session True
        (True, False, True),
        (None, True, True),  # no per-request flag: the session's legacy value
        (None, False, False),
    ],
    ids=["explicit-false-legacy-true", "explicit-true", "legacy-true", "legacy-false"],
)
def test_f23_an_explicit_request_flag_is_authoritative(
    monkeypatch, explicit, legacy, asymmetric
):
    """Round 7 F23: ``stream_generate`` read the request flag with a ``False`` default
    and fell back to the session whenever it was falsy, so the explicit ``False`` the
    server now forwards was overridden by a stale session ``True`` (asymmetric
    publication). A request with an explicit ``False`` publishes symmetrically."""
    anchors = []
    monkeypatch.setattr(
        _dispatch,
        "_compute_anchor_before_latest_user_offset",
        lambda prompt, tokenizer: anchors.append(prompt),  # None: no anchor found
    )
    state = PromptCacheState()
    state.is_asymmetric_rendering = legacy
    extra = {} if explicit is None else {"is_asymmetric_rendering": explicit}
    emitted, _ = turn(fake_model(hybrid_lm(logit_scale=4.0)), PROMPT, state, **extra)
    assert bool(anchors) is asymmetric
    if not asymmetric:
        # Symmetric publication: the full end-of-assistant history.
        assert list(state.token_ids) == PROMPT + list(emitted)
