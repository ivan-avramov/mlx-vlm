"""Fork-only: ``usage.prompt_tokens_details.cached_tokens`` must report the prefix
length the request actually reused, on every path that can reuse one.

Measured before the fix, against a real server (Qwen2.5-1.5B-Instruct-4bit, one
``X-MLX-VLM-Chat-Id``, each assistant turn fed back as the model's own reply)::

    turn  usage.prompt_tokens  cached_tokens  server telemetry
    1     37                   0              Skipped Context: 0    (cold, correct)
    2     56                   0   <- WRONG   Skipped Context: 39
    3     75                   0   <- WRONG   Skipped Context: 58

Three of the four links were already right: ``generate/dispatch.py`` sets
``GenerationResult.cached_tokens = reused_prefix_len``, ``GenerationMetrics.record_result``
reads ``cached_tokens`` off whatever it is handed, and ``StreamingToken`` declares the
field. The break was the producer->consumer handoff: ``record_result`` is duck-typed,
the streaming lane feeds it ``StreamingToken`` instances, and
``_process_cached_request`` -- the path every chat-id'd request takes -- built them
without ``cached_tokens``. The dataclass default of ``0`` is what made it silent: a
missing field would have raised, a defaulted one reads as a legitimate measurement of
"no reuse".

Why the existing tests missed it: every one of them hand-builds ``StreamingToken``
objects (or sets ``tok.cached_tokens`` on a fake) and feeds those to the endpoint, so
they exercise the consumer with a value the producer never actually supplied. The
tests here drive the REAL ``_process_cached_request`` and read the number off the
RESPONSE BODY, which is where clients read it.

Lives in its own file rather than in ``tests/test_server.py`` for the reason given at
the top of ``test_backend_label.py``: the subject is fork-only, and adding endpoint
boilerplate to a file upstream also has destabilizes ``git diff``'s alignment there.
"""

import ast
import json
import sys
from pathlib import Path
from queue import Queue
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server
import mlx_vlm.server.generation as server_generation
import mlx_vlm.server.session_manager as session_manager
from mlx_vlm.generate.common import GenerationResult

# The live measurement above: (prompt_tokens, reused_prefix_len).
MEASURED_TURNS = [(37, 0), (56, 39), (75, 58)]

PROMPT_TPS = 412.5
GENERATION_TPS = 178.25


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


def _bare_response_generator():
    """A ResponseGenerator with only the attributes ``_process_cached_request``
    touches. Mirrors ``test_backend_label.py``'s helper."""
    rg = server.ResponseGenerator.__new__(server.ResponseGenerator)
    rg.model = SimpleNamespace(language_model=SimpleNamespace())
    rg.processor = SimpleNamespace()
    rg.vision_cache = None
    rg.stop_tokens = set()
    rg.kv_bits = None
    rg.kv_group_size = None
    rg.kv_quant_scheme = None
    rg.quantized_kv_start = None
    rg.top_logprobs_k = 0
    rg.apc_manager = None
    rg.draft_model = None
    rg.draft_kind = None
    rg._cancelled = set()
    rg._cancel_lock = Lock()
    rg._stop = False
    rg._load_error = None
    rg._ready = Event()
    rg.requests = Queue()
    return rg


def _fake_stream_generate(*, cached_tokens, prompt_tokens, n_tokens=2):
    """Stand in for ``generate/dispatch.py::stream_generate`` on a cache hit.

    Yields real ``GenerationResult`` objects shaped exactly as dispatch yields
    them: ``cached_tokens=reused_prefix_len`` on every chunk, ``prompt_tps``
    measured over the whole logical prompt, and a terminal chunk carrying
    ``finish_reason``.
    """

    def _gen(**kwargs):
        for i in range(n_tokens):
            last = i == n_tokens - 1
            yield GenerationResult(
                text=("Red" if i == 0 else "."),
                token=100 + i,
                logprobs=None,
                prompt_tokens=prompt_tokens,
                generation_tokens=i + 1,
                total_tokens=prompt_tokens + i + 1,
                prompt_tps=PROMPT_TPS,
                generation_tps=GENERATION_TPS,
                peak_memory=1.5,
                cached_tokens=cached_tokens,
                finish_reason="stop" if last else None,
            )

    return _gen


def _drive_cached_path(monkeypatch, *, cached_tokens, prompt_tokens, n_tokens=2):
    """Run the REAL ``_process_cached_request`` and return (ctx, tokens)."""
    monkeypatch.setattr(
        sys.modules["mlx_vlm.generate"],
        "stream_generate",
        _fake_stream_generate(
            cached_tokens=cached_tokens,
            prompt_tokens=prompt_tokens,
            n_tokens=n_tokens,
        ),
    )
    rg = _bare_response_generator()
    rqueue: Queue = Queue()
    rg._process_cached_request(
        rqueue=rqueue,
        prompt="hello",
        images=None,
        args=server.GenerationArguments(),
        prompt_tokens=prompt_tokens,
        prompt_cache_state=SimpleNamespace(),
    )
    ctx = rqueue.get_nowait()
    tokens = []
    while True:
        item = rqueue.get_nowait()
        if item is None:
            break
        if isinstance(item, server_generation.KeepAlive):
            continue
        assert not isinstance(item, Exception), item
        tokens.append(item)
    return ctx, tokens


class TestTheProducerToConsumerHandoff:
    """``_process_cached_request`` converts ``GenerationResult`` ->
    ``StreamingToken``. Anything it does not copy across is silently defaulted."""

    def test_streaming_tokens_carry_the_reused_prefix_length(self, monkeypatch):
        """The bug, at its narrowest: the chunk has it, the token dropped it."""
        _, tokens = _drive_cached_path(
            monkeypatch, cached_tokens=39, prompt_tokens=56, n_tokens=3
        )
        assert tokens, "expected StreamingTokens on the queue"
        # EQUALS, not merely nonzero -- an off-by-a-lot value passes `> 0`.
        assert [t.cached_tokens for t in tokens] == [39, 39, 39]

    def test_streaming_tokens_carry_the_prefill_rate(self, monkeypatch):
        """``prefill=0.0 tok/s`` on every cached request came from here: the
        chunk's ``prompt_tps`` was dropped, and the envelope has no fallback for
        it, so ``float(None or 0.0)`` reached the log line."""
        _, tokens = _drive_cached_path(monkeypatch, cached_tokens=39, prompt_tokens=56)
        assert [t.prompt_tps for t in tokens] == [PROMPT_TPS, PROMPT_TPS]
        assert all(t.prompt_tps not in (None, 0.0) for t in tokens)

    def test_streaming_tokens_carry_the_decode_rate(self, monkeypatch):
        """``generation_tps`` is dispatch's decode-only rate (its ``tic`` is
        reset after prefill). Passing it makes this path report what the inline
        ``backend=generate`` path reports, from the same measurement."""
        _, tokens = _drive_cached_path(monkeypatch, cached_tokens=39, prompt_tokens=56)
        assert [t.generation_tps for t in tokens] == [GENERATION_TPS, GENERATION_TPS]

    def test_streaming_tokens_are_stamped_when_produced(self, monkeypatch):
        """``emitted_at`` is what ``GenerationMetrics.record_chunk`` uses for
        ``token_times``; without it the timestamps are taken when the async
        endpoint CONSUMES the token, so queue latency is charged to decode."""
        _, tokens = _drive_cached_path(monkeypatch, cached_tokens=39, prompt_tokens=56)
        stamps = [t.emitted_at for t in tokens]
        assert all(s is not None for s in stamps)
        assert stamps == sorted(stamps)

    def test_metrics_pick_the_value_up_off_the_streaming_tokens(self, monkeypatch):
        """The consumer half, with the real ``GenerationMetrics``: this is the
        object every usage schema and the metrics envelope read."""
        _, tokens = _drive_cached_path(monkeypatch, cached_tokens=58, prompt_tokens=75)
        metrics = server_generation.GenerationMetrics()
        for tok in tokens:
            metrics.record_chunk(tok)
        assert metrics.cached_tokens == 58
        assert metrics.prompt_tps == PROMPT_TPS

    @pytest.mark.parametrize("prompt_tokens,reused", MEASURED_TURNS)
    def test_each_measured_turn_reports_its_own_prefix_length(
        self, monkeypatch, prompt_tokens, reused
    ):
        """The full measured sequence, including the cold turn -- 0 on turn 1 is
        the one value that was already right, and it must stay right."""
        _, tokens = _drive_cached_path(
            monkeypatch, cached_tokens=reused, prompt_tokens=prompt_tokens
        )
        metrics = server_generation.GenerationMetrics()
        for tok in tokens:
            metrics.record_chunk(tok)
        assert metrics.cached_tokens == reused


class TestTheResponseBody:
    """Where clients read it, and where nothing guarded it.

    ``generate()`` here runs the REAL ``_process_cached_request``, so the number
    in the body has travelled the whole producer->consumer path.
    """

    @staticmethod
    def _generator(monkeypatch, *, cached_tokens, prompt_tokens):
        rg = _bare_response_generator()
        monkeypatch.setattr(
            sys.modules["mlx_vlm.generate"],
            "stream_generate",
            _fake_stream_generate(
                cached_tokens=cached_tokens, prompt_tokens=prompt_tokens
            ),
        )

        class CachedPathResponseGenerator:
            tokenizer = SimpleNamespace(decode=lambda tokens: "")

            def validate_context_budget(self, *a, **kw):
                return None

            def _cpu_preprocess(self, *a, **kw):
                return {"input_ids": mx.zeros((1, prompt_tokens), dtype=mx.int32)}

            def generate(self, prompt=None, images=None, audio=None, args=None, **kw):
                rqueue: Queue = Queue()
                rg._process_cached_request(
                    rqueue=rqueue,
                    prompt=prompt or "hello",
                    images=None,
                    args=args or server.GenerationArguments(),
                    prompt_tokens=prompt_tokens,
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

        return CachedPathResponseGenerator()

    @staticmethod
    def _model_patch():
        return patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        )

    def test_chat_completions_non_streaming(self, client, monkeypatch):
        monkeypatch.setattr(session_manager, "_session_cache_max", 8)
        monkeypatch.setattr(
            server.runtime,
            "response_generator",
            self._generator(monkeypatch, cached_tokens=39, prompt_tokens=56),
        )
        with self._model_patch():
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 8,
                },
                headers={session_manager._chat_id_header: "chat-cached"},
            )
        assert response.status_code == 200
        usage = response.json()["usage"]
        assert usage["prompt_tokens_details"]["cached_tokens"] == 39

    def test_chat_completions_streaming(self, client, monkeypatch):
        monkeypatch.setattr(session_manager, "_session_cache_max", 8)
        monkeypatch.setattr(
            server.runtime,
            "response_generator",
            self._generator(monkeypatch, cached_tokens=39, prompt_tokens=56),
        )
        with self._model_patch():
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 8,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                headers={session_manager._chat_id_header: "chat-cached-stream"},
            )
            assert response.status_code == 200
            body = response.read().decode()

        usages = []
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :].strip()
            if payload == "[DONE]":
                continue
            obj = json.loads(payload)
            if obj.get("usage"):
                usages.append(obj["usage"])
        assert usages, "expected a usage chunk with stream_options.include_usage"
        assert usages[-1]["prompt_tokens_details"]["cached_tokens"] == 39

    def test_responses_endpoint_input_tokens_details(self, client, monkeypatch):
        """The second consumer: ``OpenAIUsage.input_tokens_details``."""
        monkeypatch.setattr(session_manager, "_session_cache_max", 8)
        monkeypatch.setattr(
            server.runtime,
            "response_generator",
            self._generator(monkeypatch, cached_tokens=58, prompt_tokens=75),
        )
        with self._model_patch():
            response = client.post(
                "/v1/responses",
                json={"model": "demo", "input": "Hi", "max_output_tokens": 8},
                headers={session_manager._chat_id_header: "chat-responses"},
            )
        assert response.status_code == 200
        usage = response.json()["usage"]
        assert usage["input_tokens_details"]["cached_tokens"] == 58

    def test_metrics_envelope_reports_a_real_prefill_rate(self, client, monkeypatch):
        """``prefill=0.0 tok/s`` was reported for every cached request, including
        cold ones. The envelope's ``prefill_tok_s`` IS ``prompt_tps``, with no
        fallback, so a dropped value reads as zero throughput."""
        monkeypatch.setattr(session_manager, "_session_cache_max", 8)
        monkeypatch.setattr(
            server.runtime,
            "response_generator",
            self._generator(monkeypatch, cached_tokens=39, prompt_tokens=56),
        )
        with self._model_patch():
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 8,
                },
                headers={session_manager._chat_id_header: "chat-prefill"},
            )
        assert response.status_code == 200
        latest = client.get("/metrics").json()["latest"]
        assert latest["backend"] == server.BACKEND_CACHED_SESSION
        assert latest["cached_tokens"] == 39
        assert latest["prefill_tok_s"] == PROMPT_TPS
        # prompt_eval_time_s is derived from it; 0.0/None makes it disappear.
        assert latest["prompt_eval_time_s"] == pytest.approx(56 / PROMPT_TPS)


class TestEveryStreamingTokenSiteIsAccountedFor:
    """The structural guard. The bug was one construction site out of six
    omitting one field, and no test could see it because the omission is a
    default rather than an error. Any NEW site must either pass
    ``cached_tokens`` or be listed here with a reason.
    """

    # Sites that legitimately pass no ``cached_tokens``, by enclosing function.
    # ``_run_speculative`` builds a fresh ``make_speculative_prompt_cache`` per
    # batch and never consults ``request.prompt_cache_state``, so no prefix reuse
    # can occur on it -- its own "Prefill completed" line hardcodes
    # ``cached_tokens=0`` for the same reason. Reporting 0 there is truthful, not
    # a dropped field. (That the path cannot reuse a prefix at all is a separate,
    # pre-existing limitation; see the module docstring in ``generation.py``.)
    EXEMPT_FUNCTIONS = {"_run_speculative"}

    @staticmethod
    def _streaming_token_sites():
        source = Path(server_generation.__file__).read_text()
        tree = ast.parse(source)
        sites = []

        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.stack = []

            def _visit_scope(self, node):
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            visit_FunctionDef = _visit_scope
            visit_AsyncFunctionDef = _visit_scope

            def visit_Call(self, node):
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name == "StreamingToken":
                    sites.append(
                        SimpleNamespace(
                            lineno=node.lineno,
                            function=self.stack[-1] if self.stack else "<module>",
                            keywords={
                                kw.arg for kw in node.keywords if kw.arg is not None
                            },
                            starstar=any(kw.arg is None for kw in node.keywords),
                        )
                    )
                self.generic_visit(node)

        Visitor().visit(tree)
        return sites

    def test_the_sites_are_where_we_think_they_are(self):
        """Pin the count so a new construction site cannot be added without
        this file being updated."""
        sites = self._streaming_token_sites()
        assert len(sites) == 6, [(s.lineno, s.function) for s in sites]

    def test_every_reuse_capable_site_passes_cached_tokens(self):
        missing = [
            (s.lineno, s.function)
            for s in self._streaming_token_sites()
            if s.function not in self.EXEMPT_FUNCTIONS
            and not s.starstar
            and "cached_tokens" not in s.keywords
        ]
        assert not missing, (
            "StreamingToken built without cached_tokens on a path that can "
            f"reuse a prefix: {missing}. The dataclass default of 0 makes this "
            "read as a legitimate 'no reuse' measurement in the response body."
        )

    def test_every_reuse_capable_site_passes_the_prefill_rate(self):
        """Same shape, same silence: ``prompt_tps=None`` becomes
        ``prefill=0.0 tok/s``."""
        missing = [
            (s.lineno, s.function)
            for s in self._streaming_token_sites()
            if not s.starstar and "prompt_tps" not in s.keywords
        ]
        assert not missing, f"StreamingToken built without prompt_tps: {missing}"

    def test_the_exempt_path_is_still_exempt_for_the_stated_reason(self):
        """If ``_run_speculative`` ever gains prefix reuse, this fails and the
        exemption above has to be re-earned rather than silently inherited."""
        source = Path(server_generation.__file__).read_text()
        tree = ast.parse(source)
        func = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_run_speculative"
        )
        # AST, not text: a comment mentioning the name must not satisfy or trip
        # this. What matters is whether the loop READS the field.
        reads_cache_state = any(
            (isinstance(n, ast.Attribute) and n.attr == "prompt_cache_state")
            or (isinstance(n, ast.Name) and n.id == "prompt_cache_state")
            or (isinstance(n, ast.Constant) and n.value == "prompt_cache_state")
            for n in ast.walk(func)
        )
        assert not reads_cache_state, (
            "_run_speculative now looks at prompt_cache_state; it may be able to "
            "reuse a prefix, so its StreamingToken sites can no longer default "
            "cached_tokens to 0."
        )
        body = ast.get_source_segment(source, func) or ""
        assert "cached_tokens=0" in body, (
            "the hardcoded cached_tokens=0 in _run_speculative's prefill log is "
            "the in-tree statement that this path reuses nothing"
        )


class TestTheContinuousBatchingPathAlreadyCarriedIt:
    """``_step`` (the BatchGenerator path) reads ``cached_tokens`` off the
    prompt response into ``active[uid]`` and passes it. Pinned so the fix to the
    cached path cannot regress it, and because it is the reason the field looked
    wired end-to-end."""

    def test_step_copies_the_prompt_response_value_to_the_token(self, monkeypatch):
        rg = _bare_response_generator()
        rg.tokenizer = SimpleNamespace(decode=lambda t: "")
        rqueue: Queue = Queue()
        info = {
            "rqueue": rqueue,
            "streamer": SimpleNamespace(finalize=lambda: ""),
            "request_id": "req-1",
            "spec_snapshot": None,
        }
        active = {7: info}
        monkeypatch.setattr(rg, "_stream_text", lambda i, tok, finish: "hi")
        monkeypatch.setattr(rg, "_log_prefill_progress", lambda bg, a: None)
        monkeypatch.setattr(rg, "_log_decode_progress", lambda *a, **kw: 123.0)

        prompt_response = SimpleNamespace(uid=7, prompt_tps=99.0, cached_tokens=17)
        response = SimpleNamespace(
            uid=7, token=5, token_logprob=0.0, finish_reason=None, top_logprobs=None
        )

        class FakeBatchGen:
            def next(self, **kw):
                return [prompt_response], [response]

        rg._step(FakeBatchGen(), active)
        token = rqueue.get_nowait()
        assert token.cached_tokens == 17
        assert token.prompt_tps == 99.0
