"""M57 review round 1 (Amendment 1): rules 5/6, GPU refusal, F1-F4. CPU only."""

import asyncio
import json
import logging
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient
from test_attention_policy import (  # noqa: F401  (fixtures + helpers)
    NativeCache,
    Recorder,
    _attn_modules,
    _cpu_device,
    _k,
    _q,
    _tiny_lm,
    passthrough,
    sdpa,
)
from test_cached_tokens_reporting import _bare_response_generator

import mlx_vlm.server as server
import mlx_vlm.server.generation as generation_module
import mlx_vlm.server.session_manager as session_manager
from mlx_vlm import attention_policy as ap
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_vlm.server.schemas import GenerationTimings, StreamingTimings


@pytest.fixture
def gpu(monkeypatch):
    """Pretend the default device is the GPU (forced calls are mocked)."""
    monkeypatch.setattr(ap, "_force_calls_enabled", lambda: True)


# ------------------------------------------------------------ rules 5 and 6
class TestRules5And6:
    def test_rule5_batch_larger_than_one_is_never_forced(self):
        policy = ap.resolve_policy("fused_v1")
        q = SimpleNamespace(shape=(2, 24, 512, 256), dtype=mx.bfloat16)
        assert not policy.should_force(q, 4096, None, None, NativeCache())
        assert policy.should_force(_q(512), 4096, None, None, NativeCache())

    def test_rule5_cache_carrying_left_padding_is_never_forced(self):
        policy = ap.resolve_policy("fused_v1")
        padded = SimpleNamespace(left_padding=mx.array([0], dtype=mx.int32))
        assert not policy.should_force(_q(512), 4096, None, None, padded)

    def test_rule6_causal_query_longer_than_keys_is_never_forced(self):
        policy = ap.resolve_policy("fused_v1")
        assert not policy.should_force(_q(512), 256, None, "causal", NativeCache())
        assert policy.should_force(_q(512), 512, None, "causal", NativeCache())
        assert policy.should_force(_q(512), 256, None, None, NativeCache())

    def test_rule6_reaches_the_native_branch(self, sdpa):
        policy = ap.resolve_policy("fused_v1")
        from mlx_vlm.models import base

        base.scaled_dot_product_attention(
            _q(512), _k(256), _k(256), NativeCache(), 0.5, "causal", policy=policy
        )
        assert "force_fused" not in sdpa.kwargs[0]

    def test_suspended_policy_never_forces(self):
        policy = ap.resolve_policy("fused_v1")
        with policy.suspended():
            assert not policy.should_force(_q(512), 4096, None, None, NativeCache())
        assert policy.should_force(_q(512), 4096, None, None, NativeCache())


def _batch_cache(left_padding):
    arrays = ArraysCache(size=2)
    arrays.left_padding = mx.array(left_padding, dtype=mx.int32)
    return [arrays, BatchKVCache(list(left_padding))]


class TestF2RealPaddedBatch:
    def _stamped(self):
        lm = _tiny_lm(layers=2)
        policy = ap.resolve_policy("fused_v1")
        ap.apply_to_model(lm, policy)
        return lm, policy

    def test_f2_positive_control_single_sequence_prefill_is_forced(self, passthrough):
        lm, policy = self._stamped()
        ids = mx.array([list(range(1, 131))], dtype=mx.int32) % 60
        lm.model(ids, cache=[ArraysCache(size=2), KVCache()])
        assert [k.get("force_fused") for k in passthrough.kwargs] == [True]

    def test_f2_padded_batch_prefill_and_row_recursion_never_forced(self, passthrough):
        lm, policy = self._stamped()
        ids = (
            mx.array([[0] + list(range(1, 130)), list(range(1, 131))], dtype=mx.int32)
            % 60
        )
        out = lm.model(ids, cache=_batch_cache([1, 0]))
        mx.eval(out)
        assert passthrough.calls, "attention never ran"
        assert all("force_fused" not in k for k in passthrough.kwargs)
        assert policy.counters()[0] == 0

    def test_f2_unpadded_batch_of_two_is_not_forced(self, passthrough):
        lm, policy = self._stamped()
        ids = mx.array([list(range(1, 131))] * 2, dtype=mx.int32) % 60
        mx.eval(lm.model(ids, cache=_batch_cache([0, 0])))
        assert all("force_fused" not in k for k in passthrough.kwargs)

    def test_f2_row_recursion_alone_is_suspended(self, passthrough):
        """Rows re-enter the same stamped modules with a plain B=1 cache."""
        lm, policy = self._stamped()
        ids = (
            mx.array([[0] + list(range(1, 130)), list(range(1, 131))], dtype=mx.int32)
            % 60
        )
        mx.eval(lm.model(ids, cache=_batch_cache([1, 0])))
        shapes = {c[0].shape[0] for c in passthrough.calls}
        assert 1 in shapes  # the recursion really ran single-row attention
        assert all("force_fused" not in k for k in passthrough.kwargs)
        assert policy._suspended == 0  # restored


# ------------------------------------------------------------ non-GPU refusal
class TestNonGpuRefusal:
    def test_non_gpu_worker_refuses_at_load(self, monkeypatch, capsys):
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
        lm = _tiny_lm()
        model = SimpleNamespace(language_model=lm, config=lm.config)
        assert mx.default_device() == mx.cpu
        with pytest.raises(ap.AttentionPolicyError, match="GPU"):
            generation_module._apply_attention_policy_from_env(model)
        assert not hasattr(lm, "attention_policy") or lm.attention_policy is None
        assert "SKIPPED" not in capsys.readouterr().err

    def test_self_test_refuses_instead_of_skipping_off_gpu(self):
        with pytest.raises(ap.AttentionPolicyError, match="GPU"):
            ap.self_test(
                ap.resolve_policy("fused_v1"),
                heads=24,
                kv_heads=4,
                head_dim=256,
                dtype=mx.bfloat16,
                max_kv=262144,
                sdpa=Recorder(),
            )

    def test_auto_on_cpu_still_loads(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "auto")
        assert (
            generation_module._apply_attention_policy_from_env(
                SimpleNamespace(language_model=None)
            )
            is None
        )


# ------------------------------------------------------------ F3 self-test
class TestF3SelfTest:
    DIMS = dict(heads=24, kv_heads=4, head_dim=256, dtype=mx.bfloat16)

    def test_f3_decision_at_max_kv_covers_the_short_forced_region(self):
        policy = ap.resolve_policy("fused_v1")
        rec = Recorder()
        res = ap.self_test(
            policy, max_kv=262144, sdpa=rec, force_calls=True, **self.DIMS
        )
        assert res.ran == len(rec.calls) == 12
        # qL 22 is the smallest rule 4 forces at 24 heads x 262144 keys (qL 9 is
        # not forced there: 113 MB < 2**28); 127 is forced only by the decision
        # at max_kv, yet the calls run at 4096 keys.
        assert {c[0].shape[-2] for c in rec.calls} == {22, 127, 128, 512}
        assert all(c[1].shape[-2] == 4096 for c in rec.calls)
        assert all(c[3]["force_fused"] is True for c in rec.calls)
        assert policy.counters() == (0, 0)

    def test_f3_zero_forced_calls_on_gpu_is_a_failure(self):
        class Never(ap.FusedV1Policy):
            def should_force(self, *a, **k):
                return False

        with pytest.raises(ap.AttentionPolicyError, match="zero"):
            ap.self_test(
                Never(), max_kv=262144, sdpa=Recorder(), force_calls=True, **self.DIMS
            )

    def test_f3_query_dtype_comes_from_the_models_attention_computation(self):
        lm = _tiny_lm()  # bf16 weights and activations
        rec = Recorder()
        results = ap.self_test_model(
            lm,
            ap.resolve_policy("fused_v1"),
            max_kv=262144,
            sdpa=rec,
            force_calls=True,
        )
        assert results and rec.calls
        assert all(c[0].dtype == mx.bfloat16 for c in rec.calls)

    def test_f3_fp32_model_fails_loudly_not_silently_zero_calls(self):
        lm = _tiny_lm(dtype=mx.float32)
        with pytest.raises(ap.AttentionPolicyError, match="zero"):
            ap.self_test_model(
                lm,
                ap.resolve_policy("fused_v1"),
                max_kv=262144,
                sdpa=Recorder(),
                force_calls=True,
            )

    def test_f3_load_logs_calls_run_and_elapsed(
        self, monkeypatch, capsys, gpu, passthrough
    ):
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
        lm = _tiny_lm()
        model = SimpleNamespace(language_model=lm, config=lm.config)
        generation_module._apply_attention_policy_from_env(model)
        err = capsys.readouterr().err
        assert "self-test" in err and "forced calls" in err and "s" in err

    def test_f3_docstring_states_readiness_timeout_bounds_a_hang(self):
        assert "readiness timeout" in ap.self_test.__doc__


# ------------------------------------------------------------ F1 counters
def _fake_stream(policy, forced=3, auto=2):
    from mlx_vlm.generate.common import GenerationResult

    def _gen(**kwargs):
        for _ in range(forced if policy else 0):
            policy.decide(_q(512), 4096, None)
        for _ in range(auto if policy else 0):
            policy.decide(_q(1), 4096, None)
        for i in range(2):
            yield GenerationResult(
                text="x",
                token=100 + i,
                logprobs=None,
                prompt_tokens=56,
                generation_tokens=i + 1,
                total_tokens=57 + i,
                prompt_tps=400.0,
                generation_tps=170.0,
                peak_memory=1.5,
                cached_tokens=39,
                finish_reason="stop" if i else None,
            )

    return _gen


def _endpoint(monkeypatch, policy, stream):
    from queue import Queue

    import mlx_vlm.tests.test_cached_tokens_reporting  # noqa: F401

    monkeypatch.setattr(session_manager, "_session_cache_max", 8)
    monkeypatch.setattr(
        sys.modules["mlx_vlm.generate"], "stream_generate", _fake_stream(policy)
    )
    rg = _bare_response_generator()
    rg.attention_policy = policy

    class Gen:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, *a, **kw):
            return None

        def _cpu_preprocess(self, *a, **kw):
            return {"input_ids": mx.zeros((1, 56), dtype=mx.int32)}

        def generate(self, prompt=None, images=None, audio=None, args=None, **kw):
            rqueue = Queue()
            rg._process_cached_request(
                rqueue=rqueue,
                prompt=prompt or "hi",
                images=None,
                args=args or server.GenerationArguments(),
                prompt_tokens=56,
                prompt_cache_state=SimpleNamespace(),
            )
            ctx = rqueue.get_nowait()
            items = []
            while True:
                item = rqueue.get_nowait()
                if item is None:
                    break
                if isinstance(item, generation_module.KeepAlive):
                    continue
                items.append(item)
            return ctx, iter(items)

    monkeypatch.setattr(server.runtime, "response_generator", Gen())


def _post(client, stream=False):
    from unittest.mock import patch

    body = {
        "model": "demo",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8,
    }
    if stream:
        body["stream"] = True
    with patch.object(
        server,
        "get_cached_model",
        return_value=(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(model_type="qwen2_vl"),
        ),
    ):
        return client.post(
            "/v1/chat/completions",
            json=body,
            headers={session_manager._chat_id_header: "chat-r1" + str(stream)},
        )


class TestF1HttpTimings:
    @pytest.fixture
    def client(self):
        with TestClient(server.app) as c:
            yield c

    def test_f1_non_streaming_timings_carry_counters_under_fused_v1(
        self, client, monkeypatch
    ):
        _endpoint(monkeypatch, ap.resolve_policy("fused_v1"), None)
        r = _post(client)
        assert r.status_code == 200
        t = r.json()["timings"]
        assert (t["sdpa_forced"], t["sdpa_auto"]) == (3, 2)

    def test_f1_non_streaming_response_has_no_sdpa_keys_under_auto(
        self, client, monkeypatch
    ):
        _endpoint(monkeypatch, None, None)
        # under auto the generator has no policy; counters stay absent
        r = _post(client)
        t = r.json()["timings"]
        assert "sdpa_forced" not in t and "sdpa_auto" not in t

    def test_f1_streaming_final_chunk_timings_carry_counters(self, client, monkeypatch):
        _endpoint(monkeypatch, ap.resolve_policy("fused_v1"), None)
        r = _post(client, stream=True)
        seen = []
        for line in r.read().decode().splitlines():
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                t = json.loads(line[6:]).get("timings") or {}
                if "sdpa_forced" in t:
                    seen.append((t["sdpa_forced"], t["sdpa_auto"]))
        assert seen == [(3, 2)]

    def test_f1_timing_models_omit_none_counters(self):
        assert (
            "sdpa_forced"
            not in StreamingTimings(predicted_per_second=1.0).model_dump_json()
        )
        m = SimpleNamespace(
            rate=10.0,
            generation_tps=10.0,
            token_times=[],
            cached_tokens=0,
            prompt_tps=100.0,
            peak_memory=0.0,
            sdpa_forced=None,
            sdpa_auto=None,
        )
        dumped = GenerationTimings.from_metrics(m, 10, 5).model_dump_json()
        assert "sdpa" not in dumped
        m.sdpa_forced, m.sdpa_auto = 1, 2
        dumped = json.loads(GenerationTimings.from_metrics(m, 10, 5).model_dump_json())
        assert (dumped["sdpa_forced"], dumped["sdpa_auto"]) == (1, 2)

    def test_f1_request_completed_log_line_carries_counters_only_when_set(self, caplog):
        env = generation_module._build_metrics_envelope(
            endpoint="e",
            model="m",
            stream=False,
            backend="b",
            prompt_tokens=1,
            completion_tokens=1,
            generated_tokens=1,
            request_elapsed_s=1.0,
            request_started_s=0.0,
            sdpa_forced=3,
            sdpa_auto=2,
        )
        plain = generation_module._build_metrics_envelope(
            endpoint="e",
            model="m",
            stream=False,
            backend="b",
            prompt_tokens=1,
            completion_tokens=1,
            generated_tokens=1,
            request_elapsed_s=1.0,
            request_started_s=0.0,
        )
        rec = generation_module.ServerMetricsStore()
        with caplog.at_level(logging.INFO):
            rec.record_success(env)
            rec.record_success(plain)
        lines = [
            r.getMessage()
            for r in caplog.records
            if "Request completed" in r.getMessage()
        ]
        assert "sdpa_forced=3 sdpa_auto=2" in lines[0]
        assert "sdpa" not in lines[1]


# ------------------------------------------------------------ F4 readiness
class TestF4ReadinessPath:
    def _failing_env(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
        model = SimpleNamespace(
            language_model=SimpleNamespace(named_modules=lambda: []),
            config=SimpleNamespace(model_type="llama"),
        )
        monkeypatch.setattr(
            generation_module,
            "load_model_resources",
            lambda *a, **k: (model, SimpleNamespace(), model.config),
        )

    def test_f4_response_generator_wait_until_ready_raises(self, monkeypatch):
        self._failing_env(monkeypatch)
        rg = generation_module.ResponseGenerator("x", None)
        try:
            with pytest.raises(ap.AttentionPolicyError):
                rg.wait_until_ready(timeout=30)
        finally:
            rg.stop_and_join(timeout=2)

    def test_f4_lifespan_preload_propagates_so_startup_fails(self, monkeypatch, gpu):
        self._failing_env(monkeypatch)
        monkeypatch.setenv("MLX_VLM_PRELOAD_MODEL", "x")

        async def go():
            async with server.app.router.lifespan_context(server.app):
                pass

        with pytest.raises(Exception) as exc:
            asyncio.run(go())
        assert exc.type is ap.AttentionPolicyError
        assert str(exc.value).startswith("attention-policy fused_v1:")
        assert "no qualified attention call sites" in str(exc.value)
        server.runtime.response_generator = None
