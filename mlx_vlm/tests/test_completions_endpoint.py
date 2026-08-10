"""Tests for the fork-only legacy `/v1/completions` endpoint.

**Why this file exists: the endpoint had zero test coverage.** It is fork-only —
`completions_endpoint` (409 lines), `CompletionRequest`/`Choice`/`Response`/
`StreamChoice`/`StreamChunk`, `_normalize_stop_sequences`, `_truncate_at_stop`,
`_completion_final_chunk` and `_completion_usage_chunk` are all absent from
`upstream/main` — and both `/completions` and `/v1/completions` are live registered
routes. A reference sweep for fork-only definitions with no test reference put this
whole feature at the top of the list.

That gap is structural, not an oversight anyone could have caught: every one of the
seven audit scripts compares against `upstream/main`, so **fork-only code is invisible
to all of them by construction.** There is no upstream copy to diff, no upstream symbol
to miss, no upstream hunk to drop. `docs/upstream-gaps.md`'s machinery protects the
~800 commits of upstream content this fork carries; nothing protected the fork's own.

The contract being pinned, in the endpoint's own words: the prompt is fed to the model
**verbatim** — no `apply_chat_template`, no thinking-token injection, no tool parsing.
That is the endpoint's entire reason to exist, so `test_prompt_is_fed_verbatim` and
`test_thinking_is_forced_off` are the two tests to keep if any are ever pruned.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server
import mlx_vlm.server.openai as server_openai
from mlx_vlm.generate import GenerationResult


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _no_response_generator(monkeypatch):
    """Route every test through the `generate` fallback rather than the GPU worker.

    The endpoint has two backends and the `ResponseGenerator` one needs a live worker
    thread. Everything asserted here — verbatim prompting, stop truncation, echo,
    thinking suppression, the SSE envelope — is backend-independent, and the fallback
    is the path a test can drive deterministically.
    """
    monkeypatch.setattr(server.runtime, "response_generator", None)


def _fake_model():
    return SimpleNamespace(), SimpleNamespace(), SimpleNamespace(model_type="qwen2_vl")


def _result(text, prompt_tokens=7, generation_tokens=3):
    return GenerationResult(
        text=text,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        prompt_tps=20.0,
        generation_tps=8.0,
        peak_memory=0.1,
    )


def _post(client, body, *, text="world", captured=None):
    """POST to /v1/completions with `generate` returning `text`.

    When `captured` is a dict it collects the kwargs `generate` was called with, which
    is how the verbatim-prompt and thinking-suppression contracts get asserted.
    """

    def fake_generate(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return _result(text)

    with (
        patch.object(server_openai, "get_cached_model", return_value=_fake_model()),
        patch.object(server_openai, "generate", side_effect=fake_generate),
    ):
        return client.post("/v1/completions", json=body)


def _sse_chunks(response):
    """Parse an SSE body into the list of decoded `data:` payloads, minus [DONE]."""
    out = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


def _stream(client, body, tokens):
    def fake_stream_generate(**_kwargs):
        return iter(tokens)

    with (
        patch.object(server_openai, "get_cached_model", return_value=_fake_model()),
        patch.object(
            server_openai, "stream_generate", side_effect=fake_stream_generate
        ),
    ):
        return client.post("/v1/completions", json={**body, "stream": True})


def _token(text, finish_reason=None):
    return GenerationResult(
        text=text, prompt_tokens=7, generation_tokens=1, finish_reason=finish_reason
    )


class TestTheVerbatimContract:
    """The two tests to keep if any are ever pruned."""

    def test_prompt_is_fed_verbatim(self, client):
        """No chat template. This is the whole difference from /chat/completions.

        A regression here is silent and severe: the model would receive
        `<start_of_turn>user\\n…` wrapping for a raw-continuation request, so every
        completion would be a chat reply instead of a continuation.
        """
        captured = {}
        response = _post(
            client,
            {"model": "demo", "prompt": "Once upon a"},
            captured=captured,
        )

        assert response.status_code == 200
        assert captured["prompt"] == "Once upon a"

    def test_no_chat_template_is_applied(self, client):
        with patch.object(server_openai, "apply_chat_template") as template:
            response = _post(client, {"model": "demo", "prompt": "raw"})

        assert response.status_code == 200
        template.assert_not_called()

    def test_thinking_is_forced_off(self, client):
        """Otherwise the thinking budget engages and the output gets split.

        A raw continuation has no assistant turn to split on, so a thinking-enabled
        default would strip or misattribute real output text.

        Asserted on the `gen_args` object rather than on `generate`'s kwargs, because
        `to_generate_kwargs()` forwards `enable_thinking` but not `thinking_budget` —
        the budget is consumed server-side. Asserting only what reaches `generate`
        would have left the budget suppression untested.
        """
        captured = {}
        stash = []
        real_build = server_openai._build_gen_args

        def capture_build(*args, **kwargs):
            gen_args = real_build(*args, **kwargs)
            stash.append(gen_args)
            return gen_args

        with patch.object(server_openai, "_build_gen_args", side_effect=capture_build):
            _post(
                client,
                {"model": "demo", "prompt": "raw", "max_tokens": 5},
                captured=captured,
            )

        assert captured["enable_thinking"] is False
        assert stash and stash[0].enable_thinking is False
        assert stash[0].thinking_budget is None


class TestNonStreamingResponse:
    def test_shape_and_object_type(self, client):
        response = _post(client, {"model": "demo", "prompt": "hi"}, text=" world")

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "text_completion"
        assert body["id"].startswith("cmpl-")
        assert body["model"] == "demo"
        assert body["choices"][0]["text"] == " world"
        assert body["choices"][0]["index"] == 0
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_usage_is_reported(self, client):
        response = _post(client, {"model": "demo", "prompt": "hi"})

        usage = response.json()["usage"]
        assert usage["prompt_tokens"] == 7
        assert usage["completion_tokens"] == 3

    def test_logprobs_is_always_null(self, client):
        """Accepted for compatibility, documented as never populated."""
        response = _post(client, {"model": "demo", "prompt": "hi", "logprobs": 5})

        assert response.status_code == 200
        assert response.json()["choices"][0]["logprobs"] is None

    @pytest.mark.parametrize("field", ["suffix", "user"])
    def test_ignored_compatibility_fields_are_accepted(self, client, field):
        """They must not 400 — a client sending them is doing nothing wrong."""
        response = _post(client, {"model": "demo", "prompt": "hi", field: "x"})

        assert response.status_code == 200


class TestEcho:
    def test_echo_prepends_the_prompt(self, client):
        response = _post(
            client, {"model": "demo", "prompt": "Once", "echo": True}, text=" more"
        )

        assert response.json()["choices"][0]["text"] == "Once more"

    def test_echo_defaults_off(self, client):
        response = _post(client, {"model": "demo", "prompt": "Once"}, text=" more")

        assert response.json()["choices"][0]["text"] == " more"


class TestStopSequences:
    def test_a_string_stop_truncates(self, client):
        response = _post(
            client, {"model": "demo", "prompt": "p", "stop": "END"}, text="keepENDdrop"
        )

        body = response.json()
        assert body["choices"][0]["text"] == "keep"
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_a_list_stop_truncates(self, client):
        response = _post(
            client,
            {"model": "demo", "prompt": "p", "stop": ["X", "END"]},
            text="keepENDdrop",
        )

        assert response.json()["choices"][0]["text"] == "keep"

    def test_the_earliest_match_wins(self, client):
        """Not the first sequence listed — the earliest occurrence in the text."""
        response = _post(
            client,
            {"model": "demo", "prompt": "p", "stop": ["LATE", "EARLY"]},
            text="aEARLYbLATEc",
        )

        assert response.json()["choices"][0]["text"] == "a"

    def test_a_non_matching_stop_leaves_text_intact(self, client):
        response = _post(
            client, {"model": "demo", "prompt": "p", "stop": "ZZZ"}, text="keep all"
        )

        assert response.json()["choices"][0]["text"] == "keep all"

    def test_echo_and_stop_together(self, client):
        """The prompt is prepended AFTER truncation, so a stop sequence occurring in
        the PROMPT must not truncate the echoed prefix."""
        response = _post(
            client,
            {"model": "demo", "prompt": "pENDq", "stop": "END", "echo": True},
            text="rENDs",
        )

        assert response.json()["choices"][0]["text"] == "pENDqr"


class TestRequestValidation:
    @pytest.mark.parametrize("n", [2, 5])
    def test_n_other_than_one_is_rejected(self, client, n):
        response = _post(client, {"model": "demo", "prompt": "p", "n": n})

        assert response.status_code == 400
        assert "n=" in response.json()["detail"]

    def test_n_equal_to_one_is_accepted(self, client):
        response = _post(client, {"model": "demo", "prompt": "p", "n": 1})

        assert response.status_code == 200

    def test_a_missing_model_is_a_400_not_a_500(self, client):
        """The endpoint catches its own validation error on purpose — a malformed body
        must produce an OpenAI-style error, never a traceback."""
        response = _post(client, {"prompt": "p"})

        assert response.status_code == 400
        assert "Invalid request body" in response.json()["detail"]

    def test_a_list_prompt_uses_the_first_element(self, client):
        captured = {}
        _post(
            client, {"model": "demo", "prompt": ["first", "second"]}, captured=captured
        )

        assert captured["prompt"] == "first"

    def test_an_empty_list_prompt_becomes_empty_string(self, client):
        captured = {}
        _post(client, {"model": "demo", "prompt": []}, captured=captured)

        assert captured["prompt"] == ""

    def test_an_omitted_prompt_becomes_empty_string(self, client):
        captured = {}
        _post(client, {"model": "demo"}, captured=captured)

        assert captured["prompt"] == ""


class TestBothRoutesAreRegistered:
    """`/completions` and `/v1/completions` must both resolve to the same handler.

    A registry-shaped concern: the route table is a sequence of `app.post(...)` calls,
    which `check_upstream_registries.py` cannot see because these routes are fork-only.
    """

    @pytest.mark.parametrize("path", ["/completions", "/v1/completions"])
    def test_route_exists(self, client, path):
        def fake_generate(**_kwargs):
            return _result("ok")

        with (
            patch.object(server_openai, "get_cached_model", return_value=_fake_model()),
            patch.object(server_openai, "generate", side_effect=fake_generate),
        ):
            response = client.post(path, json={"model": "demo", "prompt": "p"})

        assert response.status_code == 200
        assert response.json()["choices"][0]["text"] == "ok"


class TestStreaming:
    def test_it_streams_deltas_and_terminates(self, client):
        response = _stream(
            client,
            {"model": "demo", "prompt": "p"},
            [_token("Hello"), _token(" world", finish_reason="stop")],
        )

        assert response.status_code == 200
        assert response.text.endswith("data: [DONE]\n\n")
        texts = [c["choices"][0]["text"] for c in _sse_chunks(response) if c["choices"]]
        assert "".join(texts) == "Hello world"

    def test_every_chunk_is_a_text_completion(self, client):
        response = _stream(
            client, {"model": "demo", "prompt": "p"}, [_token("a", "stop")]
        )

        chunks = _sse_chunks(response)
        assert chunks
        assert all(c["object"] == "text_completion" for c in chunks)

    def test_a_final_chunk_carries_the_finish_reason(self, client):
        response = _stream(
            client, {"model": "demo", "prompt": "p"}, [_token("a", "length")]
        )

        finals = [
            c
            for c in _sse_chunks(response)
            if c["choices"] and c["choices"][0]["finish_reason"]
        ]
        assert [c["choices"][0]["finish_reason"] for c in finals] == ["length"]
        assert finals[0]["choices"][0]["text"] == ""

    def test_a_usage_chunk_follows_the_final_chunk(self, client):
        response = _stream(
            client, {"model": "demo", "prompt": "p"}, [_token("a", "stop")]
        )

        usage_chunks = [c for c in _sse_chunks(response) if c.get("usage")]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["choices"] == []

    def test_echo_emits_the_prompt_as_the_first_chunk(self, client):
        response = _stream(
            client,
            {"model": "demo", "prompt": "PROMPT", "echo": True},
            [_token("tail", "stop")],
        )

        first = _sse_chunks(response)[0]
        assert first["choices"][0]["text"] == "PROMPT"

    def test_streaming_withholds_the_stop_sequence_and_its_tail(self, client):
        """The stop sequence and everything after it must never reach the client.

        This is the streaming half of the stop contract and the easiest half to get
        wrong: deltas are computed against what has already been emitted, so a naive
        implementation streams the matched sequence before noticing it.
        """
        response = _stream(
            client,
            {"model": "demo", "prompt": "p", "stop": "END"},
            [_token("keep"), _token("EN"), _token("Ddrop"), _token("more", "stop")],
        )

        texts = [c["choices"][0]["text"] for c in _sse_chunks(response) if c["choices"]]
        joined = "".join(texts)
        assert "END" not in joined
        assert "drop" not in joined
        assert joined == "keep"

    def test_an_incomplete_partial_stop_is_flushed_not_dropped(self, client):
        """The other half of the hold-back, and the worse bug if missed.

        Generation ending in "EN" with `stop="END"` never completes the sequence, so
        those bytes are real output being held. Holding them back without a final flush
        silently truncates the completion — a fix that trades a visible leak for
        invisible data loss.
        """
        response = _stream(
            client,
            {"model": "demo", "prompt": "p", "stop": "END"},
            [_token("keep"), _token("EN", "stop")],
        )

        texts = [c["choices"][0]["text"] for c in _sse_chunks(response) if c["choices"]]
        assert "".join(texts) == "keepEN"

    def test_a_stop_that_never_matches_streams_everything(self, client):
        """The hold-back must not eat output when the stop simply does not occur."""
        response = _stream(
            client,
            {"model": "demo", "prompt": "p", "stop": "ZZZ"},
            [_token("all"), _token(" of it", "stop")],
        )

        texts = [c["choices"][0]["text"] for c in _sse_chunks(response) if c["choices"]]
        assert "".join(texts) == "all of it"

    def test_a_stop_hit_reports_finish_reason_stop(self, client):
        response = _stream(
            client,
            {"model": "demo", "prompt": "p", "stop": "END"},
            [_token("aENDb"), _token("never", "stop")],
        )

        finals = [
            c
            for c in _sse_chunks(response)
            if c["choices"] and c["choices"][0]["finish_reason"]
        ]
        assert finals[0]["choices"][0]["finish_reason"] == "stop"


class TestHelpers:
    """The four fork-only helpers, unit-level."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, []),
            ("", []),
            ("END", ["END"]),
            (["a", "b"], ["a", "b"]),
            ([], []),
            (["a", "", "b"], ["a", "b"]),
            (["a", None, 3, "b"], ["a", "b"]),
        ],
    )
    def test_normalize_stop_sequences(self, value, expected):
        """Non-string and empty entries are dropped rather than raising: `stop` comes
        straight off a client request, so a malformed element must not 500."""
        assert server_openai._normalize_stop_sequences(value) == expected

    @pytest.mark.parametrize(
        ("text", "stops", "expected"),
        [
            ("abc", [], ("abc", None)),
            ("", ["x"], ("", None)),
            ("abc", ["x"], ("abc", None)),
            ("aXbc", ["X"], ("a", "X")),
            ("aXbYc", ["Y", "X"], ("a", "X")),
            ("Xabc", ["X"], ("", "X")),
            ("abc", [""], ("abc", None)),
        ],
    )
    def test_truncate_at_stop(self, text, stops, expected):
        assert server_openai._truncate_at_stop(text, stops) == expected

    def test_truncate_at_stop_matches_the_anthropic_handler(self):
        """`_truncate_at_stop`'s docstring claims it mirrors anthropic's
        `_apply_stop_sequences` "so behaviour stays consistent". Two endpoints
        disagreeing about where a stop sequence cuts is a real bug, and a claim in a
        docstring is not a check — so this is the check.
        """
        from mlx_vlm.server.anthropic import _apply_stop_sequences

        cases = [
            ("aXbYc", ["Y", "X"]),
            ("no match", ["Q"]),
            ("", ["X"]),
            ("Xlead", ["X"]),
            ("multi", []),
        ]
        for text, stops in cases:
            assert server_openai._truncate_at_stop(
                text, stops
            ) == _apply_stop_sequences(text, stops), (text, stops)


class TestStreamChunkSerialization:
    def test_to_sse_json_omits_unset_timings(self, cur=None):
        """`timings` is a llama.cpp-style extension; emitting `null` for it on every
        delta chunk would triple the SSE payload for no benefit."""
        from mlx_vlm.server.schemas import CompletionStreamChoice, CompletionStreamChunk

        chunk = CompletionStreamChunk(
            id="cmpl-1",
            created=1,
            model="demo",
            choices=[CompletionStreamChoice(text="hi", index=0)],
        )

        payload = json.loads(chunk.to_sse_json())

        assert "timings" not in payload
        assert payload["choices"][0]["text"] == "hi"


class _FakeCtx:
    def __init__(self, prompt_tokens=11):
        self.prompt_tokens = prompt_tokens


class _FakeResponseGenerator:
    """Minimal stand-in for the continuous-batching worker.

    `generate` is called POSITIONALLY on the streaming path and by KEYWORD on the
    non-streaming one, so it has to accept both — which is itself worth pinning, since a
    signature change on one call site only would break exactly one of the two.
    """

    def __init__(self, tokens, prompt_tokens=11):
        self._tokens = list(tokens)
        self._ctx = _FakeCtx(prompt_tokens)
        self.closed = False
        self.calls = []

    def generate(self, prompt=None, images=None, audio=None, args=None, **kwargs):
        self.calls.append({"prompt": prompt, "images": images, "audio": audio})
        return self._ctx, self._iter()

    def validate_context_budget(self, prompt, **kwargs):
        """The streaming path preflights the context budget through the generator.

        A no-op here, but it has to EXIST: without it the endpoint's outer handler
        turns the AttributeError into a 500 and the test sees an empty stream rather
        than a failure at the point of the mistake. That is how the first version of
        this fake failed, and it is worth the comment because the symptom points
        nowhere near the cause.
        """
        self.validated = prompt
        return None

    def _iter(self):
        outer = self

        class _It:
            def __init__(self):
                self._i = iter(outer._tokens)

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._i)

            def close(self):
                outer.closed = True

        return _It()


def _with_generator(client, body, generator, *, stream=False):
    with patch.object(server_openai, "get_cached_model", return_value=_fake_model()):
        with patch.object(server.runtime, "response_generator", generator):
            return client.post("/v1/completions", json={**body, "stream": stream})


class TestResponseGeneratorBackend:
    """The continuous-batching backend — the endpoint's PRIMARY path in production.

    Added after a real coverage run: measured against this file alone, the
    `runtime.response_generator is not None` branch was entirely unexecuted. The
    full-suite run showed it covered, which was an artifact of another module leaving
    `runtime.response_generator` set — executed by accident, asserted by nobody. That is
    the sharper form of `fork_coverage_report.py`'s own caveat that "executed is not
    asserted", and the reason a whole-suite number can flatter a single file.
    """

    def test_non_streaming_uses_the_generator_and_its_prompt_tokens(self, client):
        gen = _FakeResponseGenerator([_token("from worker", "stop")], prompt_tokens=11)

        response = _with_generator(client, {"model": "demo", "prompt": "raw"}, gen)

        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["text"] == "from worker"
        # prompt_tokens comes from the worker's ctx, not from a GenerationResult
        assert body["usage"]["prompt_tokens"] == 11
        assert gen.calls[0]["prompt"] == "raw"

    def test_the_prompt_is_still_verbatim_on_this_backend(self, client):
        """The verbatim contract is the endpoint's reason to exist, and it has to hold
        on BOTH backends — the earlier test only covered the fallback."""
        gen = _FakeResponseGenerator([_token("x", "stop")])

        _with_generator(client, {"model": "demo", "prompt": "Once upon a"}, gen)

        assert gen.calls[0]["prompt"] == "Once upon a"
        assert gen.calls[0]["images"] is None
        assert gen.calls[0]["audio"] is None

    def test_stop_truncation_applies_on_this_backend_too(self, client):
        gen = _FakeResponseGenerator([_token("keepENDdrop", "stop")])

        response = _with_generator(
            client, {"model": "demo", "prompt": "p", "stop": "END"}, gen
        )

        assert response.json()["choices"][0]["text"] == "keep"
        assert response.json()["choices"][0]["finish_reason"] == "stop"

    def test_the_token_iterator_is_closed(self, client):
        """It holds a slot on the GPU worker; leaking it keeps the daemon decoding an
        orphaned request."""
        gen = _FakeResponseGenerator([_token("x", "stop")])

        _with_generator(client, {"model": "demo", "prompt": "p"}, gen)

        assert gen.closed is True

    def test_streaming_uses_the_generator(self, client):
        gen = _FakeResponseGenerator([_token("a"), _token("b", "stop")])

        response = _with_generator(
            client, {"model": "demo", "prompt": "p"}, gen, stream=True
        )

        assert response.status_code == 200
        texts = [c["choices"][0]["text"] for c in _sse_chunks(response) if c["choices"]]
        assert "".join(texts) == "ab"
        assert response.text.endswith("data: [DONE]\n\n")

    def test_streaming_reports_the_continuous_batching_backend(self, client):
        """The metrics envelope's `backend` field distinguishes the two paths; getting
        it wrong makes production telemetry attribute work to the wrong engine."""
        gen = _FakeResponseGenerator([_token("a", "stop")])
        recorded = {}

        with patch.object(
            server.runtime.metrics,
            "record_success",
            side_effect=lambda env: recorded.update(env),
        ):
            _with_generator(client, {"model": "demo", "prompt": "p"}, gen, stream=True)

        assert recorded.get("backend") == "continuous_batching"

    def test_streaming_withholds_a_partial_stop_on_this_backend(self, client):
        """b75c18b7's fix has to hold on both backends — the loop is shared, but the
        token source is not, and only the fallback was covered."""
        gen = _FakeResponseGenerator(
            [_token("keep"), _token("EN"), _token("Ddrop"), _token("z", "stop")]
        )

        response = _with_generator(
            client, {"model": "demo", "prompt": "p", "stop": "END"}, gen, stream=True
        )

        texts = [c["choices"][0]["text"] for c in _sse_chunks(response) if c["choices"]]
        assert "".join(texts) == "keep"


class TestErrorContracts:
    """Every one of these was an unexecuted `except` block. They are API contracts: the
    status code a client sees is the whole observable behaviour."""

    def test_a_too_long_prompt_is_a_400_not_a_500(self, client):
        def boom(**_kwargs):
            raise server.PromptTooLongError("MAX_KV_SIZE is 8")

        with (
            patch.object(server_openai, "get_cached_model", return_value=_fake_model()),
            patch.object(server_openai, "generate", side_effect=boom),
        ):
            response = client.post(
                "/v1/completions", json={"model": "demo", "prompt": "p"}
            )

        assert response.status_code == 400
        assert "MAX_KV_SIZE is 8" in response.json()["detail"]

    def test_an_unexpected_generation_failure_is_a_500(self, client):
        def boom(**_kwargs):
            raise RuntimeError("metal exploded")

        with (
            patch.object(server_openai, "get_cached_model", return_value=_fake_model()),
            patch.object(server_openai, "generate", side_effect=boom),
        ):
            response = client.post(
                "/v1/completions", json={"model": "demo", "prompt": "p"}
            )

        assert response.status_code == 500
        assert "Generation failed" in response.json()["detail"]

    def test_a_bad_sampling_parameter_is_a_400(self, client):
        """`_build_gen_args` validates sampling params; its failure is the client's
        fault and must not read as a server error."""

        def boom(*_args, **_kwargs):
            raise ValueError("top_p must be in [0, 1]")

        with (
            patch.object(server_openai, "get_cached_model", return_value=_fake_model()),
            patch.object(server_openai, "_build_gen_args", side_effect=boom),
        ):
            response = client.post(
                "/v1/completions", json={"model": "demo", "prompt": "p"}
            )

        assert response.status_code == 400
        assert "top_p" in response.json()["detail"]

    def test_a_streaming_failure_is_reported_in_band(self, client):
        """SSE headers are already sent, so a mid-stream failure cannot change the
        status code — it has to arrive as a data frame or the client hangs."""

        def boom(**_kwargs):
            raise RuntimeError("decode blew up")

        with (
            patch.object(server_openai, "get_cached_model", return_value=_fake_model()),
            patch.object(server_openai, "stream_generate", side_effect=boom),
        ):
            response = client.post(
                "/v1/completions",
                json={"model": "demo", "prompt": "p", "stream": True},
            )

        assert response.status_code == 200
        errors = [c for c in _sse_chunks(response) if "error" in c]
        assert errors and "decode blew up" in errors[0]["error"]


class TestTokenAccounting:
    """The two token shapes the stream loop must count differently.

    `openai.py`'s own comment: *"GenerationResult.generation_tokens is cumulative;
    StreamingToken lacks it and is counted per-token."* Getting that backwards
    double-counts or under-counts `usage.completion_tokens` on every streamed request —
    silently, since neither shape raises.

    Both branches were unexecuted by this file until now; the `GenerationResult` one is
    what every other test here happens to use.
    """

    def test_a_cumulative_generation_tokens_is_taken_as_is(self, client):
        """`GenerationResult.generation_tokens` is a running total, so the last value
        wins — summing them would multiply the count."""
        tokens = [
            GenerationResult(text="a", prompt_tokens=7, generation_tokens=1),
            GenerationResult(
                text="b", prompt_tokens=7, generation_tokens=2, finish_reason="stop"
            ),
        ]

        response = _stream(client, {"model": "demo", "prompt": "p"}, tokens)

        usage = [c for c in _sse_chunks(response) if c.get("usage")][0]["usage"]
        assert usage["completion_tokens"] == 2

    def test_a_streaming_token_is_accumulated_per_token(self, client):
        """`StreamingToken` has `token_count` but no `generation_tokens`, so the loop
        must ADD rather than assign. Two tokens of 3 is 6, not 3."""
        tokens = [
            server.StreamingToken(text="a", token=1, logprobs=0.0, finish_reason=None),
            server.StreamingToken(
                text="b", token=2, logprobs=0.0, finish_reason="stop"
            ),
        ]
        for tok in tokens:
            tok.token_count = 3

        response = _stream(client, {"model": "demo", "prompt": "p"}, tokens)

        usage = [c for c in _sse_chunks(response) if c.get("usage")][0]["usage"]
        assert usage["completion_tokens"] == 6

    def test_an_exhausted_iterator_without_a_finish_reason_terminates(self, client):
        """The `token is None` break — normal exhaustion. Every other streaming test
        here ends on an explicit `finish_reason`, so this exit was never taken, and a
        stream that cannot end on exhaustion hangs the client."""
        tokens = [_token("a"), _token("b")]  # no finish_reason anywhere

        response = _stream(client, {"model": "demo", "prompt": "p"}, tokens)

        assert response.text.endswith("data: [DONE]\n\n")
        texts = [c["choices"][0]["text"] for c in _sse_chunks(response) if c["choices"]]
        assert "".join(texts) == "ab"
        finals = [
            c
            for c in _sse_chunks(response)
            if c["choices"] and c["choices"][0]["finish_reason"]
        ]
        assert finals[0]["choices"][0]["finish_reason"] == "stop"
