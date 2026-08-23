"""Fork-only: the request-metrics ``backend=`` label must name the generation path
that RAN, not the one that was AVAILABLE.

Lives in its own file rather than in ``tests/test_server.py`` for two reasons.
The subject is fork-only — ``GenerationContext.backend``, ``resolve_backend_label``
and the serial cached-session path have no upstream counterpart — and adding ~550
lines of repetitive endpoint boilerplate to a file upstream also has destabilizes
``git diff``'s alignment there, which made ``check_fork_markers.py`` demand a
marker on a class that is byte-identical to upstream's. Restored upstream test
files are kept close to upstream so future merges apply cleanly.
"""

import logging
import sys
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


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


class TestBackendLabelReportsThePathThatRan:
    """``backend=`` in the request-metrics envelope must name the generation
    path that RAN, not the one that was AVAILABLE.

    Before this, the four ``openai.py`` envelope sites and the two in
    ``anthropic.py`` all computed::

        "continuous_batching" if runtime.response_generator is not None
        else "generate"

    That predicate is true for the whole process lifetime once a model is
    loaded, so it says nothing about the request. Meanwhile the GPU worker has
    two genuinely different AR paths: ``BatchGenerator`` (real continuous
    batching, concurrent rows) and ``_process_cached_request`` (B=1, multiple
    concurrent cached requests processed *serially* through the daemon's
    request queue, speculative decoding bypassed). Requests taking the serial
    path were labelled ``continuous_batching``.

    It matters because ``session_manager._resolve_session`` case 3 routes any
    anonymous request with >=1 hashable message to a fresh ``anon:`` session,
    so every one-shot API call (a UI's title/tags/follow-up generation, say)
    takes the serial path — and the logs actively confirmed the false premise
    that those calls would batch.

    The fix stamps the label on ``GenerationContext`` where the worker CHOOSES
    the path, so the value cannot drift from reality at a logging site again.
    """

    # -- the premise --

    def test_the_available_predicate_cannot_tell_the_two_paths_apart(self):
        """Pin the bug's mechanism: one loaded generator, two paths, one label."""
        batched = server.GenerationContext(
            uid=1, prompt_tokens=5, backend=server.BACKEND_CONTINUOUS_BATCHING
        )
        serial = server.GenerationContext(
            uid=2, prompt_tokens=5, backend=server.BACKEND_CACHED_SESSION
        )
        # The old expression, evaluated with a generator loaded, for both.
        response_generator_loaded = True
        old = [
            (
                "continuous_batching"
                if response_generator_loaded is not None
                else "generate"
            )
            for _ in (batched, serial)
        ]
        assert old[0] == old[1] == "continuous_batching"
        # The ctx-stamped label distinguishes them.
        assert server.resolve_backend_label(batched) == "continuous_batching"
        assert server.resolve_backend_label(serial) == "cached_session"

    def test_no_context_means_the_endpoint_generated_inline(self):
        assert server.resolve_backend_label(None) == server.BACKEND_DIRECT
        # An object without the field (an older fake, a partially-built ctx)
        # must degrade to the inline label rather than raise.
        assert server.resolve_backend_label(SimpleNamespace()) == "generate"

    # -- the worker half: the label is stamped where the path is chosen --

    @staticmethod
    def _bare_response_generator():
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

    def test_serial_cached_path_stamps_cached_session(self, monkeypatch):
        """The real ``_process_cached_request`` puts the serial label on ctx."""

        def fake_stream_generate(**kwargs):
            yield SimpleNamespace(
                token=7,
                text="hi",
                logprobs=None,
                finish_reason="stop",
                peak_memory=0.0,
            )

        monkeypatch.setattr(
            sys.modules["mlx_vlm.generate"], "stream_generate", fake_stream_generate
        )
        rg = self._bare_response_generator()
        rqueue: Queue = Queue()
        rg._process_cached_request(
            rqueue=rqueue,
            prompt="hello",
            images=None,
            args=server.GenerationArguments(),
            prompt_tokens=5,
            prompt_cache_state=SimpleNamespace(),
        )
        ctx = rqueue.get_nowait()
        assert isinstance(ctx, server.GenerationContext)
        assert ctx.backend == "cached_session"

    def _drive_run_once(self, monkeypatch, request, *, caplog=None):
        """Push one request through the real ``_run`` batch loop and return the
        ctx it handed back plus the captured prefill log records.

        Stubs only what needs a GPU: the model init, BatchGenerator, the vision
        embed, and the decode step. The dispatch and labelling logic under test
        is the real thing.
        """
        rg = self._bare_response_generator()
        # _run's finally-clause stream cleanup assumes it owns a dedicated GPU
        # worker thread; here _run() runs on pytest's main thread, and clearing
        # the main thread's MLX streams poisons every later test in the session.
        monkeypatch.setattr(server_generation, "clear_mlx_streams", lambda: None)
        monkeypatch.setattr(rg, "_initialize_model", lambda: None)
        monkeypatch.setattr(server_generation, "is_diffusion_model", lambda m: False)
        monkeypatch.setattr(
            rg,
            "_gpu_embed",
            lambda raw, images=None, apc_semantic_hash=None: (
                mx.zeros((1, 3), dtype=mx.int32),
                {},
            ),
        )
        monkeypatch.setattr(rg, "_make_sampler", lambda args: None)
        monkeypatch.setattr(rg, "_make_logits_processors", lambda args, ids: None)

        inserted = []

        class FakeBatchGenerator:
            has_work = True

            def __init__(self, *a, **kw):
                pass

            def insert(self, token_lists, **kw):
                inserted.append(token_lists)
                return (11,)

            def close(self):
                pass

        monkeypatch.setattr(server_generation, "BatchGenerator", FakeBatchGenerator)
        monkeypatch.setattr(
            rg, "_step", lambda batch_gen, active, gen_kwargs=None: None
        )

        # Hand the loop exactly one batch, then tell it to stop. `_run`'s inner
        # `except Exception` swallows errors and continues, so bounding the loop
        # here rather than from inside a stub is what keeps a broken stub a test
        # failure instead of a hang.
        served = []

        def _collect(**kwargs):
            if served:
                return [], True
            served.append(True)
            return [request], False

        monkeypatch.setattr(rg, "_collect_pending_requests", _collect)

        rg.requests.put(request)
        rg._run()
        return request.rqueue

    @staticmethod
    def _queued(rqueue, *, prompt_cache_state):
        return server_generation.QueuedGenerationRequest(
            rqueue=rqueue,
            raw_inputs={"input_ids": mx.zeros((1, 3), dtype=mx.int32)},
            prompt_tokens=3,
            args=server.GenerationArguments(),
            prompt_cache_state=prompt_cache_state,
            prompt="hello",
        )

    def test_the_two_paths_out_of_the_batch_loop_get_distinct_labels(self, monkeypatch):
        """Drive both paths through the real ``_run`` loop with identical stubs.

        The ONLY difference between the two requests is ``prompt_cache_state``,
        which is what ``_run`` dispatches on. Before the fix both came back
        labelled ``continuous_batching``.
        """

        def fake_stream_generate(**kwargs):
            yield SimpleNamespace(
                token=7,
                text="hi",
                logprobs=None,
                finish_reason="stop",
                peak_memory=0.0,
            )

        monkeypatch.setattr(
            sys.modules["mlx_vlm.generate"], "stream_generate", fake_stream_generate
        )

        batched_q: Queue = Queue()
        self._drive_run_once(
            monkeypatch, self._queued(batched_q, prompt_cache_state=None)
        )
        batched = batched_q.get_nowait()

        serial_q: Queue = Queue()
        self._drive_run_once(
            monkeypatch,
            self._queued(serial_q, prompt_cache_state=SimpleNamespace()),
        )
        serial = serial_q.get_nowait()

        assert isinstance(batched, server.GenerationContext)
        assert isinstance(serial, server.GenerationContext)
        assert batched.backend == "continuous_batching"
        assert serial.backend == "cached_session"
        assert batched.backend != serial.backend

    def test_prefill_log_names_the_serial_path(self, monkeypatch, caplog):
        """The ``Prefill started`` line was wrong for the same reason: it was
        emitted BEFORE the cached-session dispatch, with the batch literal."""

        def fake_stream_generate(**kwargs):
            yield SimpleNamespace(
                token=7,
                text="hi",
                logprobs=None,
                finish_reason="stop",
                peak_memory=0.0,
            )

        monkeypatch.setattr(
            sys.modules["mlx_vlm.generate"], "stream_generate", fake_stream_generate
        )
        caplog.set_level(logging.INFO, logger="mlx_vlm.server")
        rqueue: Queue = Queue()
        self._drive_run_once(
            monkeypatch, self._queued(rqueue, prompt_cache_state=SimpleNamespace())
        )
        prefill = [
            r.getMessage()
            for r in caplog.records
            if "Prefill started" in r.getMessage()
        ]
        assert prefill, "expected a Prefill started line"
        assert "backend=cached_session" in prefill[0]
        assert "backend=continuous_batching" not in prefill[0]

    # -- the endpoint half: the label reaches the metrics envelope --

    @staticmethod
    def _fake_generator(backend, *, prompt_tokens=6, cached_tokens=0):
        """A response_generator whose ctx carries the worker's real label."""
        tokens = [
            server.StreamingToken(
                text="Hello",
                token=1,
                logprobs=0.0,
                finish_reason=None,
                prompt_tps=20.0,
            ),
            server.StreamingToken(
                text=" there",
                token=2,
                logprobs=0.0,
                finish_reason="stop",
                prompt_tps=20.0,
            ),
        ]
        for tok in tokens:
            tok.cached_tokens = cached_tokens

        class FakeResponseGenerator:
            tokenizer = SimpleNamespace(decode=lambda tokens: "")

            def validate_context_budget(self, *a, **kw):
                return None

            def _cpu_preprocess(self, *a, **kw):
                return {"input_ids": mx.zeros((1, prompt_tokens), dtype=mx.int32)}

            def generate(self, prompt=None, images=None, audio=None, args=None, **kw):
                return (
                    server.GenerationContext(
                        uid=1, prompt_tokens=prompt_tokens, backend=backend
                    ),
                    iter(list(tokens)),
                )

        return FakeResponseGenerator()

    @staticmethod
    def _latest(client):
        payload = client.get("/metrics").json()
        return payload["latest"]

    @pytest.mark.parametrize(
        "backend",
        ["continuous_batching", "cached_session", "diffusion", "speculative_mtp"],
    )
    def test_chat_completions_non_streaming_reports_the_worker_label(
        self, client, monkeypatch, backend
    ):
        monkeypatch.setattr(
            server.runtime, "response_generator", self._fake_generator(backend)
        )
        with patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ):
            response = client.post(
                "/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 8,
                },
            )
        assert response.status_code == 200
        assert self._latest(client)["backend"] == backend

    @pytest.mark.parametrize("backend", ["continuous_batching", "cached_session"])
    def test_chat_completions_streaming_reports_the_worker_label(
        self, client, monkeypatch, backend
    ):
        monkeypatch.setattr(
            server.runtime, "response_generator", self._fake_generator(backend)
        )
        with patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ):
            response = client.post(
                "/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 8,
                    "stream": True,
                },
            )
            assert response.status_code == 200
            response.read()
        assert self._latest(client)["backend"] == backend

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("backend", ["continuous_batching", "cached_session"])
    def test_responses_endpoint_reports_the_worker_label(
        self, client, monkeypatch, backend, stream
    ):
        monkeypatch.setattr(
            server.runtime, "response_generator", self._fake_generator(backend)
        )
        with patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ):
            response = client.post(
                "/responses",
                json={
                    "model": "demo",
                    "input": "Hi",
                    "max_output_tokens": 8,
                    "stream": stream,
                },
            )
            assert response.status_code == 200
            if stream:
                response.read()
        assert self._latest(client)["backend"] == backend

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("backend", ["continuous_batching", "diffusion"])
    def test_completions_endpoint_reports_the_worker_label(
        self, client, monkeypatch, backend, stream
    ):
        monkeypatch.setattr(
            server.runtime, "response_generator", self._fake_generator(backend)
        )
        with patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ):
            response = client.post(
                "/v1/completions",
                json={"model": "demo", "prompt": "Continue: ", "stream": stream},
            )
            assert response.status_code == 200
            if stream:
                response.read()
        assert self._latest(client)["backend"] == backend

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("backend", ["continuous_batching", "speculative_mtp"])
    def test_anthropic_messages_reports_the_worker_label(
        self, client, monkeypatch, backend, stream
    ):
        monkeypatch.setattr(
            server.runtime, "response_generator", self._fake_generator(backend)
        )
        with patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ):
            response = client.post(
                "/v1/messages",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 8,
                    "stream": stream,
                },
            )
            assert response.status_code == 200
            if stream:
                response.read()
        assert self._latest(client)["backend"] == backend

    # -- the diagnostic that had to be reconstructed from timestamps --

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("cached_tokens", [0, 41])
    def test_envelope_carries_session_id_and_cached_tokens(
        self, client, monkeypatch, cached_tokens, stream
    ):
        """A serial request with cached_tokens=0 reused nothing and paid the
        serialization for it. That is the pair worth logging next to backend=,
        and it is what previously had to be reconstructed from timestamps.

        Both chat sites are covered: the streaming and non-streaming envelopes
        read ``chat_id`` from different scopes.
        """
        monkeypatch.setattr(session_manager, "_session_cache_max", 8)
        monkeypatch.setattr(
            server.runtime,
            "response_generator",
            self._fake_generator("cached_session", cached_tokens=cached_tokens),
        )
        with patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ):
            response = client.post(
                "/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 8,
                    "stream": stream,
                },
                headers={session_manager._chat_id_header: "chat-abc"},
            )
            assert response.status_code == 200
            if stream:
                response.read()
        latest = self._latest(client)
        assert latest["backend"] == "cached_session"
        assert latest["cached_tokens"] == cached_tokens
        assert latest["session_id"] == "chat-abc"

    def test_anonymous_one_shot_request_is_labelled_serial(self, client, monkeypatch):
        """The reported scenario: no chat id at all. ``_resolve_session`` case 3
        gives the request a fresh ``anon:`` session, so it takes the serial path
        — which is exactly what used to be reported as continuous_batching."""
        monkeypatch.setattr(session_manager, "_session_cache_max", 8)
        monkeypatch.setattr(
            server.runtime,
            "response_generator",
            self._fake_generator("cached_session"),
        )
        with patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ):
            response = client.post(
                "/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "Summarize: hi"}],
                    "max_tokens": 8,
                },
            )
        assert response.status_code == 200
        latest = self._latest(client)
        assert latest["backend"] == "cached_session"
        assert (latest["session_id"] or "").startswith("anon:")

    def test_completed_log_line_carries_the_three_fields(self, caplog):
        caplog.set_level(logging.INFO, logger="mlx_vlm.server")
        store = server.ServerMetricsStore()
        store.begin_request(endpoint="/chat/completions", model="demo", stream=False)
        store.record_success(
            server_generation._build_metrics_envelope(
                endpoint="/chat/completions",
                model="demo",
                stream=False,
                backend="cached_session",
                prompt_tokens=100,
                completion_tokens=4,
                generated_tokens=4,
                request_elapsed_s=1.0,
                request_started_s=0.0,
                session_id="anon:9f2c",
                cached_tokens=0,
            )
        )
        line = [
            r.getMessage()
            for r in caplog.records
            if "Request completed" in r.getMessage()
        ][0]
        assert "backend=cached_session" in line
        assert "session=anon:9f2c" in line
        assert "cached_tokens=0" in line
