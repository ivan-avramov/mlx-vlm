"""P426 B1: per-request drafter counters reach the metrics envelope, `/metrics` and the `Request completed` line.

The worker already accumulates ``draft_kind / draft_rounds / draft_n / draft_n_accepted`` in ``GenerationMetrics``; the
re-certification needs them per request on real router traffic (opencode), where only the server log is observable.
Absent when no drafter ran (the line stays byte-identical for MTP-OFF requests).
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server
import mlx_vlm.server.generation as server_generation

SERVER_DIR = Path(server_generation.__file__).resolve().parent
ON = dict(draft_kind="mtp", draft_rounds=3, draft_n=6, draft_n_accepted=5)
ON_TEXT = "draft_kind=mtp draft_n=6 draft_n_accepted=5 draft_rounds=3"
BASE = dict(
    endpoint="/chat/completions",
    model="demo",
    stream=False,
    backend="b",
    prompt_tokens=1,
    completion_tokens=1,
    generated_tokens=1,
    request_elapsed_s=1.0,
    request_started_s=0.0,
)


def _completed_lines(caplog):
    return [
        r.getMessage() for r in caplog.records if "Request completed" in r.getMessage()
    ]


# ------------------------------------------------------------------ envelope + log line
def test_envelope_and_log_line_carry_draft_counters_only_when_a_drafter_ran(caplog):
    plain = server_generation._build_metrics_envelope(**BASE)
    assert not any(k.startswith("draft_") for k in plain)
    env = server_generation._build_metrics_envelope(**BASE, draft_counters=dict(ON))
    assert {k: env[k] for k in ON} == ON
    store = server_generation.ServerMetricsStore()
    with caplog.at_level(logging.INFO):
        store.record_success(env)
        store.record_success(plain)
    lines = _completed_lines(caplog)
    assert ON_TEXT in lines[0]
    assert "draft_" not in lines[1]


def test_draft_counters_of_reads_generation_metrics():
    metrics = server_generation.GenerationMetrics()
    assert server_generation.draft_counters_of(metrics) is None
    for key, value in ON.items():
        setattr(metrics, key, value)
    assert server_generation.draft_counters_of(metrics) == ON


def test_draft_counters_of_keeps_a_zero_acceptance_request():
    metrics = server_generation.GenerationMetrics()
    for key, value in dict(ON, draft_n_accepted=0).items():
        setattr(metrics, key, value)
    assert server_generation.draft_counters_of(metrics)["draft_n_accepted"] == 0


def test_every_envelope_call_site_with_verify_counters_also_passes_draft_counters():
    """The M58 counters mark the generation endpoints; B1 must reach every one of them."""
    sites = []
    for path in SERVER_DIR.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None))
                == "_build_metrics_envelope"
            ):
                kws = {k.arg for k in node.keywords}
                if "verify_counters" in kws:
                    sites.append((path.name, node.lineno, "draft_counters" in kws))
    assert len(sites) >= 8, sites
    assert all(has for _, _, has in sites), [s for s in sites if not s[2]]


# ------------------------------------------------------------------ end to end
def _fake_generator(counters):
    tokens = [
        server.StreamingToken(
            text="Hello", token=1, logprobs=0.0, finish_reason=None, prompt_tps=20.0
        ),
        server.StreamingToken(
            text=" there",
            token=2,
            logprobs=0.0,
            finish_reason="stop",
            prompt_tps=20.0,
            **counters,
        ),
    ]

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, *a, **kw):
            return None

        def _cpu_preprocess(self, *a, **kw):
            return {"input_ids": mx.zeros((1, 6), dtype=mx.int32)}

        def generate(self, prompt=None, images=None, audio=None, args=None, **kw):
            return (
                server.GenerationContext(
                    uid=1, prompt_tokens=6, backend="cached_session"
                ),
                iter(list(tokens)),
            )

    return FakeResponseGenerator()


REQUESTS = {
    "/chat/completions": lambda stream: dict(
        model="demo",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        stream=stream,
    ),
    "/responses": lambda stream: dict(
        model="demo", input="hi", max_output_tokens=8, stream=stream
    ),
    "/completions": lambda stream: dict(
        model="demo", prompt="hi", max_tokens=8, stream=stream
    ),
    "/v1/messages": lambda stream: dict(
        model="demo",
        max_tokens=8,
        messages=[{"role": "user", "content": "hi"}],
        stream=stream,
    ),
}


@pytest.mark.parametrize("arm", ["on", "off"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("endpoint", sorted(REQUESTS))
def test_a_drafted_request_reports_its_counters_end_to_end(
    monkeypatch, caplog, endpoint, stream, arm
):
    counters = dict(ON) if arm == "on" else {}
    caplog.set_level(logging.INFO, logger="mlx_vlm.server")
    with TestClient(server.app) as client:
        monkeypatch.setattr(
            server.runtime, "response_generator", _fake_generator(counters)
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
            response = client.post(endpoint, json=REQUESTS[endpoint](stream))
            assert response.status_code == 200, response.text
            response.read()
        latest = client.get("/metrics").json()["latest"]
    lines = _completed_lines(caplog)
    assert lines, "no Request completed line"
    if arm == "on":
        assert {k: latest.get(k) for k in ON} == ON
        assert ON_TEXT in lines[-1], lines[-1]
    else:
        assert not any(k.startswith("draft_") for k in latest)
        assert "draft_" not in lines[-1]


# ------------------------------------------------------------------ code round 1 D2: sequences, zero, errors, OFF bytes
def _post(client, monkeypatch, counters, endpoint="/chat/completions", stream=False):
    monkeypatch.setattr(server.runtime, "response_generator", _fake_generator(counters))
    with patch.object(
        server,
        "get_cached_model",
        return_value=(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(model_type="qwen2_vl"),
        ),
    ):
        response = client.post(endpoint, json=REQUESTS[endpoint](stream))
        response.read()
    return response


def test_a_plain_request_after_a_drafted_one_carries_no_stale_counters(
    monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger="mlx_vlm.server")
    with TestClient(server.app) as client:
        assert _post(client, monkeypatch, dict(ON)).status_code == 200
        assert _post(client, monkeypatch, {}).status_code == 200
        latest = client.get("/metrics").json()["latest"]
    first, second = _completed_lines(caplog)[-2:]
    assert ON_TEXT in first
    assert "draft_" not in second
    assert not any(k.startswith("draft_") for k in latest)


def test_a_zero_acceptance_request_reports_zero_end_to_end(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="mlx_vlm.server")
    with TestClient(server.app) as client:
        assert (
            _post(client, monkeypatch, dict(ON, draft_n_accepted=0)).status_code == 200
        )
        latest = client.get("/metrics").json()["latest"]
    assert latest["draft_n_accepted"] == 0 and latest["draft_n"] == 6
    assert "draft_n_accepted=0" in _completed_lines(caplog)[-1]


def test_a_failed_drafted_request_logs_a_failure_and_no_completion(monkeypatch, caplog):
    gen = _fake_generator(dict(ON))

    def broken(*a, **k):
        ctx, _ = _fake_generator(dict(ON)).generate()

        def tokens():
            raise RuntimeError("worker failed mid-request")
            yield  # pragma: no cover

        return ctx, tokens()

    gen.generate = broken
    caplog.set_level(logging.INFO, logger="mlx_vlm.server")
    with TestClient(server.app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(server.runtime, "response_generator", gen)
        with patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ):
            client.post("/chat/completions", json=REQUESTS["/chat/completions"](False))
    messages = [r.getMessage() for r in caplog.records]
    assert not _completed_lines(caplog)
    failed = [m for m in messages if "Request failed" in m]
    assert failed and all("draft_" not in m for m in failed)


def test_the_mtp_off_line_is_byte_identical_to_the_pre_p426_format(caplog):
    store = server_generation.ServerMetricsStore()
    store.begin_request(endpoint="/chat/completions", model="demo", stream=False)
    with caplog.at_level(logging.INFO):
        store.record_success(server_generation._build_metrics_envelope(**BASE))
    assert _completed_lines(caplog)[-1] == (
        "Request completed: endpoint=/chat/completions model=demo stream=False backend=b "
        "session=None cached_tokens=0 prompt_tokens=1 generated_tokens=1 elapsed=1.000s "
        "prefill=0.0 tok/s decode=0.0 tok/s finish_reason=None in_flight=0"
    )


def test_started_lines_carry_in_flight_for_the_overlap_rule(monkeypatch, caplog):
    """Counters are exact only without overlap (drafter-global diffs, code round 1 D1); P426 B6 refuses any ON-leg
    ``Request started`` line with in_flight > 1 — so the field must be on every started line. A fresh store: the
    process-wide one carries other tests' leaked counts (test_completions_endpoint patches record_success).
    """
    caplog.set_level(logging.INFO, logger="mlx_vlm.server")
    with TestClient(server.app) as client:
        monkeypatch.setattr(
            server.runtime, "metrics", server_generation.ServerMetricsStore()
        )
        _post(client, monkeypatch, dict(ON))
    started = [
        r.getMessage() for r in caplog.records if "Request started" in r.getMessage()
    ]
    assert started and all(m.endswith("in_flight=1") for m in started), started


# ------------------------------------------------------------------ code round 2 D2: the REAL cached-path worker seam
def _cached_path_generator(drafter, rounds_per_request):
    """A response generator whose ``generate`` runs the real ``ResponseGenerator._process_cached_request`` (the path every
    P426 lean-router request takes, ``backend=cached_session``) over a stream_generate that bumps the drafter's lifetime
    counters as the inline MTP loop does — so the endpoint sees exactly what the worker emits.
    """
    import sys as _sys
    from queue import Queue

    from mlx_vlm.tests.test_server import TestCachedPathDraftCounters

    rg = TestCachedPathDraftCounters._rg(drafter)
    script = list(rounds_per_request)

    def fake_stream_generate(**kwargs):
        rounds, drafted, accepted = script.pop(0)
        drafter.speculative_total_rounds += rounds
        drafter.speculative_total_drafted += drafted
        drafter.speculative_total_accepted += float(accepted)
        yield SimpleNamespace(
            token=1, text="hi", logprobs=None, finish_reason="stop", peak_memory=0.0
        )

    class Bridge:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, *a, **kw):
            return None

        def _cpu_preprocess(self, *a, **kw):
            return {"input_ids": mx.zeros((1, 6), dtype=mx.int32)}

        def generate(self, prompt=None, images=None, audio=None, args=None, **kw):
            gen_module = _sys.modules["mlx_vlm.generate"]
            real = gen_module.stream_generate
            gen_module.stream_generate = fake_stream_generate
            try:
                q = Queue()
                rg._process_cached_request(
                    rqueue=q,
                    prompt="x",
                    images=None,
                    args=server.GenerationArguments(),
                    prompt_tokens=6,
                    prompt_cache_state=SimpleNamespace(),
                )
            finally:
                gen_module.stream_generate = real
            items = []
            while not q.empty():
                items.append(q.get_nowait())
            tokens = [it for it in items if isinstance(it, server.StreamingToken)]
            return (
                server.GenerationContext(
                    uid=1, prompt_tokens=6, backend="cached_session"
                ),
                iter(tokens),
            )

    return Bridge()


@pytest.mark.parametrize("stream", [False, True])
def test_the_real_cached_path_counters_reach_the_log_then_a_no_round_request_has_none(
    monkeypatch, caplog, stream
):
    drafter = SimpleNamespace(
        speculative_total_rounds=10,
        speculative_total_accepted=20.0,
        speculative_total_drafted=30,
    )
    gen = _cached_path_generator(drafter, [(3, 7, 5), (0, 0, 0)])
    caplog.set_level(logging.INFO, logger="mlx_vlm.server")
    with TestClient(server.app) as client:
        monkeypatch.setattr(server.runtime, "response_generator", gen)
        with patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ):
            for _ in range(2):
                r = client.post(
                    "/chat/completions", json=REQUESTS["/chat/completions"](stream)
                )
                assert r.status_code == 200, r.text
                r.read()
    first, second = _completed_lines(caplog)[-2:]
    assert "draft_kind=mtp draft_n=7 draft_n_accepted=5 draft_rounds=3" in first
    assert "backend=cached_session" in first
    assert "draft_" not in second
