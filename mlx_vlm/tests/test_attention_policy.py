"""M57 `fused_v1` attention dispatch policy (fork-only, CPU, no real model).

`mx.fast.scaled_dot_product_attention(force_fused=True)` has no CPU kernel, so
every test MOCKS the MLX call and asserts on its keyword arguments; the model
tests forward the call to the real implementation with `force_fused` stripped.
"""

import ast
import inspect
import itertools
import pathlib
import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.base as base
from mlx_vlm import attention_policy as ap
from mlx_vlm.models.qwen3_5 import language as qwen_language
from mlx_vlm.models.qwen3_5 import speculative_verifier as qwen_verifier
from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel, Qwen3_5Attention
from mlx_vlm.server import cli as cli_module
from mlx_vlm.server import generation as generation_module

REAL_SDPA = mx.fast.scaled_dot_product_attention


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture
def gpu(monkeypatch):
    """Pretend the default device is the GPU (forced calls are mocked)."""
    monkeypatch.setattr(ap, "_force_calls_enabled", lambda: True)


class Recorder:
    """Stand-in for mx.fast.scaled_dot_product_attention."""

    def __init__(self, passthrough=False, raise_on_forced=None):
        self.calls = []
        self.passthrough = passthrough
        self.raise_on_forced = raise_on_forced

    def __call__(self, q, k, v, **kwargs):
        self.calls.append((q, k, v, kwargs))
        forced = kwargs.get("force_fused", False)
        if forced and self.raise_on_forced is not None:
            raise self.raise_on_forced
        if self.passthrough:
            kwargs = {a: b for a, b in kwargs.items() if a != "force_fused"}
            return REAL_SDPA(q, k, v, **kwargs)
        return mx.zeros((1,))

    @property
    def kwargs(self):
        return [c[3] for c in self.calls]


@pytest.fixture
def sdpa(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", rec)
    return rec


class NativeCache:
    """A cache with no `bits` attribute: the native branch."""


def _q(q_len, dtype=mx.bfloat16, heads=24):
    return SimpleNamespace(shape=(1, heads, q_len, 256), dtype=dtype)


def _k(key_len, heads=4):
    return SimpleNamespace(shape=(1, heads, key_len, 256), dtype=mx.bfloat16)


def _oracle(q_len, key_len, dtype, sinks, heads=24):
    """Rules 1-4 of the spec, written independently of the implementation."""
    if sinks is not None or dtype == mx.float32:
        return False
    if q_len <= 8:
        return False
    return q_len >= 128 or heads * q_len * key_len * dtype.size >= 2**28


# --------------------------------------------------------------------- AC1
class TestAC1DefaultPreservation:
    @pytest.mark.parametrize("q_len", [1, 2, 8, 9, 127, 128, 512, 1024])
    @pytest.mark.parametrize("policy_name", [None, "auto", ""])
    def test_ac1_absent_or_auto_policy_calls_mlx_with_todays_keywords(
        self, sdpa, q_len, policy_name
    ):
        policy = ap.resolve_policy(policy_name)
        base.scaled_dot_product_attention(
            _q(q_len), _k(4096), _k(4096), NativeCache(), 0.5, "causal", policy=policy
        )
        assert [set(k) for k in sdpa.kwargs] == [{"scale", "mask", "sinks"}]

    def test_ac1_default_keyword_is_none_and_signature_gains_only_policy(self):
        params = inspect.signature(base.scaled_dot_product_attention).parameters
        assert list(params) == [
            "queries",
            "keys",
            "values",
            "cache",
            "scale",
            "mask",
            "sinks",
            "policy",
        ]
        assert params["policy"].default is None

    def test_ac1_auto_resolves_to_no_policy_object(self):
        assert ap.resolve_policy("auto") is None
        assert ap.resolve_policy(None) is None
        assert ap.resolve_policy("") is None

    def test_ac1_attention_function_never_reads_the_environment(
        self, sdpa, monkeypatch
    ):
        class Boom(dict):
            def __getitem__(self, key):
                raise AssertionError("environment read in attention")

            get = __getitem__

        import os

        policy = ap.resolve_policy("fused_v1")
        real_environ = os.environ
        monkeypatch.setattr(os, "environ", Boom())
        try:
            base.scaled_dot_product_attention(
                _q(512), _k(4096), _k(4096), NativeCache(), 0.5, "causal",
                policy=policy,
            )
            base.scaled_dot_product_attention(
                _q(512), _k(4096), _k(4096), NativeCache(), 0.5, "causal"
            )
        finally:
            monkeypatch.setattr(os, "environ", real_environ)


# --------------------------------------------------------------------- AC2
Q_LENS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 64, 127, 128, 511, 512, 1024]
KEY_LENS = [1024, 16384, 131072, 262144]


class TestAC2DecisionTable:
    @pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
    @pytest.mark.parametrize("sinks", [None, "present"])
    def test_ac2_forced_set_equals_rules_1_to_4(self, sdpa, dtype, sinks):
        policy = ap.resolve_policy("fused_v1")
        sink_arg = None if sinks is None else mx.zeros((24,))
        forced_cells = set()
        for q_len, key_len in itertools.product(Q_LENS, KEY_LENS):
            sdpa.calls.clear()
            base.scaled_dot_product_attention(
                _q(q_len, dtype),
                _k(key_len),
                _k(key_len),
                NativeCache(),
                0.5,
                None,
                sinks=sink_arg,
                policy=policy,
            )
            (kwargs,) = sdpa.kwargs
            expected = _oracle(q_len, key_len, dtype, sink_arg)
            assert kwargs.get("force_fused", False) is expected, (q_len, key_len)
            if expected:
                assert kwargs["force_fused"] is True
                forced_cells.add((q_len, key_len))
            else:
                assert "force_fused" not in kwargs
        if dtype != mx.float32 and sinks is None:
            assert (64, 131072) in forced_cells  # score tensor >= 2**28
            assert (9, 262144) not in forced_cells  # 113 MB < 2**28
            assert (9, 1024) not in forced_cells
            assert (127, 131072) in forced_cells
            assert (127, 16384) not in forced_cells
            assert (128, 1024) in forced_cells

    def test_ac2_never_forces_query_lengths_1_to_8(self):
        policy = ap.resolve_policy("fused_v1")
        for q_len in range(1, 9):
            for key_len in (262144, 10**9):  # 10**9 makes rule 4 true at qL=8
                assert not policy.should_force(_q(q_len), key_len, None)

    def test_ac2_threshold_boundary_is_inclusive(self):
        policy = ap.resolve_policy("fused_v1")
        # heads * q_len * key_len * 2 == 2**28 exactly.
        key_len = 2**28 // (24 * 100 * 2)
        assert 24 * 100 * key_len * 2 <= 2**28
        assert not policy.should_force(_q(100), key_len, None)
        assert policy.should_force(_q(100), key_len + 4096, None)
        exact = 2**28 // (16 * 2 * 64)
        assert policy.should_force(_q(64, heads=16), exact, None)


# --------------------------------------------------------------------- tiny model
def _tiny_lm(dtype=mx.bfloat16, layers=3):
    mx.random.seed(0)
    cfg = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=layers,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=512,
        full_attention_interval=2,
        head_dim=16,
    )
    vision = VisionConfig(
        model_type="qwen3_5",
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        out_hidden_size=32,
        num_heads=1,
        in_channels=3,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=1,
        num_position_embeddings=4,
    )
    lm = LanguageModel(
        cfg,
        config=ModelConfig(text_config=cfg, vision_config=vision, model_type="qwen3_5"),
    )
    lm.set_dtype(dtype)
    mx.eval(lm.parameters())
    return lm


def _attn_modules(lm):
    return [m for _, m in lm.named_modules() if type(m) is Qwen3_5Attention]


def _forward(lm, n_tokens, cache=None):
    cache = cache if cache is not None else lm.make_cache()
    ids = mx.array([list(range(1, n_tokens + 1))], dtype=mx.int32) % 60
    out = lm(ids, cache=cache)
    mx.eval(out.logits)
    return out.logits, cache


@pytest.fixture
def passthrough(monkeypatch):
    rec = Recorder(passthrough=True)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", rec)
    return rec


# --------------------------------------------------------------------- AC3
class TestAC3Propagation:
    def test_ac3_cli_flag_reaches_the_worker_environment(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            cli_module.uvicorn, "run", lambda *a, **k: seen.setdefault("ran", True)
        )
        monkeypatch.setattr(
            cli_module, "_apply_mlx_memory_limits", lambda *a, **k: None
        )
        monkeypatch.setattr(cli_module, "_configure_session_manager", lambda **k: None)
        monkeypatch.delenv("MLX_VLM_ATTENTION_POLICY", raising=False)
        monkeypatch.setattr(
            sys, "argv", ["mlx_vlm.server", "--attention-policy", "fused_v1"]
        )
        cli_module.main()
        assert seen["ran"]
        import os

        assert os.environ["MLX_VLM_ATTENTION_POLICY"] == "fused_v1"
        monkeypatch.setattr(sys, "argv", ["mlx_vlm.server"])
        cli_module.main()
        assert os.environ["MLX_VLM_ATTENTION_POLICY"] == "auto"  # no stale value

    def test_ac3_env_to_instance_to_call_site_to_keyword_and_counters(
        self, passthrough, monkeypatch, gpu
    ):
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
        lm = _tiny_lm()
        model = SimpleNamespace(language_model=lm, config=lm.config)
        policy = generation_module._apply_attention_policy_from_env(model)
        assert isinstance(policy, ap.FusedV1Policy)
        assert lm.attention_policy is policy
        assert model.attention_policy is policy
        attn = _attn_modules(lm)
        assert attn and all(m.attention_policy is policy for m in attn)

        passthrough.calls.clear()
        _forward(lm, 130)  # one chunk of 130 query tokens: forced (>= 128)
        forced = [k.get("force_fused", False) for k in passthrough.kwargs]
        assert forced == [True] * len(attn)
        assert policy.counters() == (len(attn), 0)

        passthrough.calls.clear()
        _forward(lm, 20)  # 9 <= qL < 128 and a small score tensor: auto
        assert all("force_fused" not in k for k in passthrough.kwargs)
        assert policy.counters() == (len(attn), len(attn))

        passthrough.calls.clear()
        _forward(lm, 1)  # decode-like qL = 1: auto
        assert all("force_fused" not in k for k in passthrough.kwargs)
        assert policy.counters() == (len(attn), 2 * len(attn))

    def test_ac3_counters_reach_the_timings_envelope(self):
        base_kwargs = dict(
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
        plain = generation_module._build_metrics_envelope(**base_kwargs)
        assert "sdpa_forced" not in plain and "sdpa_auto" not in plain
        env = generation_module._build_metrics_envelope(
            **base_kwargs, sdpa_forced=3, sdpa_auto=5
        )
        assert env["sdpa_forced"] == 3 and env["sdpa_auto"] == 5

    def test_ac3_counters_flow_through_metrics_and_streaming_token(self):
        tok = generation_module.StreamingToken(
            text="", token=1, logprobs=0.0, finish_reason="stop",
            sdpa_forced=4, sdpa_auto=6,
        )
        metrics = generation_module.GenerationMetrics()
        metrics.record_result(tok)
        assert (metrics.sdpa_forced, metrics.sdpa_auto) == (4, 6)

    def test_ac3_request_delta_is_taken_from_the_policy(self):
        policy = ap.resolve_policy("fused_v1")
        snap = policy.snapshot()
        policy.decide(_q(512), 4096, None)
        policy.decide(_q(1), 4096, None)
        policy.decide(_q(1), 4096, None)
        assert policy.since(snap) == (1, 2)


# --------------------------------------------------------------------- AC4
class _TQ(base.TurboQuantKVCache):
    def __init__(self):  # no real quantizer state
        self.calls = []

    def prefill_attention(self, *a, **k):
        self.calls.append(("prefill", sorted(k)))
        return mx.zeros((1,))

    def decode_attention(self, *a, **k):
        self.calls.append(("decode", sorted(k)))
        return mx.zeros((1,))

    def quantized_attention(self, *a, **k):
        self.calls.append(("tiled", sorted(k)))
        return mx.zeros((1,))

    def dequantize(self, keys, values):
        self.calls.append(("dequantize", []))
        return keys, values


class TestAC4QuantizedDispatchUnchanged:
    @pytest.mark.parametrize("q_len", [1, 9, 512])
    def test_ac4_turboquant_branch_args_identical_with_and_without_policy(
        self, sdpa, q_len
    ):
        runs = []
        for policy in (None, ap.resolve_policy("fused_v1")):
            cache = _TQ()
            sdpa.calls.clear()
            base.scaled_dot_product_attention(
                _q(q_len), _k(4096), _k(4096), cache, 0.5, "causal", policy=policy
            )
            runs.append((cache.calls, list(sdpa.kwargs)))
            if policy is not None:
                assert policy.counters() == (0, 0)
        assert runs[0] == runs[1]

    def test_ac4_turboquant_dequantize_branch_never_forced(self, sdpa):
        policy = ap.resolve_policy("fused_v1")
        keys = mx.zeros((1, 1, 4, 4))
        queries = mx.zeros((1, 1, 512, 4))
        base.scaled_dot_product_attention(
            queries, keys, keys, _TQ(), 0.5, "causal",
            sinks=mx.zeros((1,)), policy=policy,
        )
        (kwargs,) = sdpa.kwargs
        assert "force_fused" not in kwargs
        assert policy.counters() == (0, 0)

    @pytest.mark.parametrize("q_len", [1, 9, 512])
    def test_ac4_bit_quantized_branch_args_identical(self, monkeypatch, sdpa, q_len):
        got = []
        monkeypatch.setattr(
            base,
            "quantized_scaled_dot_product_attention",
            lambda *a, **k: got.append((len(a), sorted(k.items()))) or mx.zeros((1,)),
        )
        cache = SimpleNamespace(bits=4, group_size=64)
        outs = []
        for policy in (None, ap.resolve_policy("fused_v1")):
            got.clear()
            base.scaled_dot_product_attention(
                _q(q_len), _k(4096), _k(4096), cache, 0.5, None, policy=policy
            )
            outs.append(list(got))
            if policy is not None:
                assert policy.counters() == (0, 0)
        assert outs[0] == outs[1] and len(outs[0]) == 1
        assert sdpa.calls == []


# --------------------------------------------------------------------- AC5
MODELS_DIR = pathlib.Path(base.__file__).parent


class TestAC5Scope:
    def test_ac5_exactly_one_call_site_passes_the_policy(self):
        sites = []
        for path in MODELS_DIR.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(fn):
                    if isinstance(node, ast.Call) and any(
                        kw.arg == "policy" for kw in node.keywords
                    ):
                        name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                        if name == "scaled_dot_product_attention":
                            sites.append((path.name, fn.name))
        # ast.walk visits nested functions again; de-duplicate.
        assert sorted(set(sites)) == [("language.py", "__call__")]
        assert len(sites) == 1

    def test_ac5_policy_call_site_is_in_qwen3_5_attention_only(self):
        src = inspect.getsource(Qwen3_5Attention.__call__)
        assert "policy=self.attention_policy" in src
        assert "attention_policy" not in inspect.getsource(
            qwen_language._qwen3_5_left_padded_attention
        )
        assert "policy" not in inspect.signature(
            qwen_language._qwen3_5_left_padded_attention
        ).parameters

    def test_ac5_foreign_family_modules_are_not_stamped(self):
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttention

        assert Qwen4ExpAttention is not ap.QUALIFIED_ATTENTION
        lm = _tiny_lm()
        foreign = nn.Linear(2, 2)
        holder = SimpleNamespace(language_model=lm, foreign=foreign)
        n = ap.apply_to_model(lm, ap.resolve_policy("fused_v1"))
        assert n == len(_attn_modules(lm))
        assert getattr(foreign, "attention_policy", None) is None
        assert getattr(holder, "attention_policy", None) is None


# --------------------------------------------------------------------- AC6
class TestAC6LoudFailure:
    def test_ac6_unknown_value_is_rejected(self):
        with pytest.raises(ValueError, match="bogus"):
            ap.resolve_policy("bogus")

    def test_ac6_worker_argparse_rejects_unknown_value(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["mlx_vlm.server", "--attention-policy", "bogus"]
        )
        with pytest.raises(SystemExit) as exc:
            cli_module.main()
        assert exc.value.code != 0

    def test_ac6_unqualified_family_refuses_with_one_stderr_line(
        self, monkeypatch, capsys, gpu
    ):
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
        model = SimpleNamespace(
            language_model=nn.Sequential(nn.Linear(2, 2)),
            config=SimpleNamespace(model_type="llama"),
        )
        with pytest.raises(ap.AttentionPolicyError):
            generation_module._apply_attention_policy_from_env(model)
        err = [l for l in capsys.readouterr().err.splitlines() if l.strip()]
        assert len(err) == 1 and "fused_v1" in err[0]

    def test_ac6_model_with_attention_sinks_refuses(self, monkeypatch, capsys, gpu):
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
        lm = _tiny_lm()
        _attn_modules(lm)[0].sinks = mx.zeros((2,))
        model = SimpleNamespace(language_model=lm, config=lm.config)
        with pytest.raises(ap.AttentionPolicyError, match="sinks"):
            generation_module._apply_attention_policy_from_env(model)
        assert len([l for l in capsys.readouterr().err.splitlines() if l]) == 1

    def test_ac6_auto_never_refuses_any_family(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "auto")
        model = SimpleNamespace(
            language_model=nn.Sequential(nn.Linear(2, 2)),
            config=SimpleNamespace(model_type="llama"),
        )
        assert generation_module._apply_attention_policy_from_env(model) is None
        assert not hasattr(model, "attention_policy")

    def test_ac6_self_test_raise_fails_the_load(self, monkeypatch, gpu):
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
        rec = Recorder(raise_on_forced=ValueError("no fused kernel"))
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", rec)
        lm = _tiny_lm()
        model = SimpleNamespace(language_model=lm, config=lm.config)
        with pytest.raises(ap.AttentionPolicyError, match="self-test"):
            generation_module._apply_attention_policy_from_env(model)

    def test_ac6_initialize_model_raises_so_the_worker_never_reports_ready(
        self, monkeypatch, gpu
    ):
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
        model = SimpleNamespace(
            language_model=nn.Sequential(nn.Linear(2, 2)),
            config=SimpleNamespace(model_type="llama"),
        )
        monkeypatch.setattr(
            generation_module,
            "load_model_resources",
            lambda *a, **k: (model, SimpleNamespace(), model.config),
        )
        monkeypatch.delenv("MLX_VLM_MOE_EXPAND", raising=False)
        fake = SimpleNamespace(
            model_path="x", adapter_path=None, draft_kind_override=None,
            draft_model_path=None, apc_manager=None,
        )
        with pytest.raises(ap.AttentionPolicyError, match="qualified"):
            generation_module.ResponseGenerator._initialize_model(fake)
        # The readiness / lifespan propagation is tested for real in
        # test_attention_policy_r1.py::TestF4ReadinessPath.


class TestSelfTest:
    DIMS = dict(heads=24, kv_heads=4, head_dim=256, dtype=mx.bfloat16)

    def test_self_test_fails_when_a_policy_forces_a_short_query(self):
        class Bad(ap.FusedV1Policy):
            def should_force(self, queries, key_length, sinks, mask=None, cache=None):
                return True

        with pytest.raises(ap.AttentionPolicyError, match="1..8"):
            ap.self_test(
                Bad(), sdpa=Recorder(), force_calls=True, max_kv=262144, **self.DIMS
            )

    def test_self_test_raise_from_a_forced_call_is_an_error(self):
        rec = Recorder(raise_on_forced=ValueError("boom"))
        with pytest.raises(ap.AttentionPolicyError, match="self-test.*boom"):
            ap.self_test(
                ap.resolve_policy("fused_v1"), sdpa=rec, force_calls=True,
                max_kv=262144, **self.DIMS,
            )


# --------------------------------------------------------------------- AC7
class TestAC7Isolation:
    def test_ac7_two_instances_dispatch_per_instance(self, passthrough):
        fused = _tiny_lm()
        plain = _tiny_lm()
        policy = ap.resolve_policy("fused_v1")
        ap.apply_to_model(fused, policy)
        passthrough.calls.clear()
        _forward(plain, 130)
        assert all("force_fused" not in k for k in passthrough.kwargs)
        passthrough.calls.clear()
        _forward(fused, 130)
        assert all(k.get("force_fused") is True for k in passthrough.kwargs)
        passthrough.calls.clear()
        _forward(plain, 130)
        assert all("force_fused" not in k for k in passthrough.kwargs)
        assert policy.counters()[0] == len(_attn_modules(fused))

    def test_ac7_two_instances_with_separate_policy_objects_count_separately(
        self, passthrough
    ):
        a, b = _tiny_lm(), _tiny_lm()
        pa, pb = ap.resolve_policy("fused_v1"), ap.resolve_policy("fused_v1")
        ap.apply_to_model(a, pa)
        ap.apply_to_model(b, pb)
        _forward(a, 130)
        assert pa.counters()[0] > 0 and pb.counters() == (0, 0)

    def test_ac7_environment_change_after_load_has_no_effect(
        self, passthrough, monkeypatch, gpu
    ):
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
        lm = _tiny_lm()
        model = SimpleNamespace(language_model=lm, config=lm.config)
        generation_module._apply_attention_policy_from_env(model)
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "auto")
        passthrough.calls.clear()
        _forward(lm, 130)
        assert all(k.get("force_fused") is True for k in passthrough.kwargs)
        plain = _tiny_lm()
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
        passthrough.calls.clear()
        _forward(plain, 130)
        assert all("force_fused" not in k for k in passthrough.kwargs)


# --------------------------------------------------------------------- AC8
class TestAC8Verifier:
    @pytest.mark.parametrize("block", [2, 3])
    def test_ac8_verifier_never_receives_the_policy_and_is_bit_identical(
        self, monkeypatch, block
    ):
        rec = Recorder(passthrough=True)
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", rec)
        seen = []
        real = qwen_verifier.scaled_dot_product_attention

        def spy(*args, **kwargs):
            seen.append(kwargs.get("policy"))
            assert "policy" not in kwargs
            return real(*args, **kwargs)

        monkeypatch.setattr(qwen_verifier, "scaled_dot_product_attention", spy)

        def run(lm):
            cache = lm.make_cache()
            _forward(lm, 12, cache)  # warm native cache so keys exist
            ids = mx.array([[5, 6, 7][:block]], dtype=mx.int32)
            rec.calls.clear()
            out = lm(ids, cache=cache, speculative_verify=True)
            mx.eval(out.logits)
            return out.logits

        main_lm = _tiny_lm()
        policy_lm = _tiny_lm()
        policy = ap.resolve_policy("fused_v1")
        ap.apply_to_model(policy_lm, policy)
        ref = run(main_lm)
        got = run(policy_lm)
        assert seen and all(p is None for p in seen)
        assert all("force_fused" not in k for k in rec.kwargs)
        assert mx.array_equal(ref, got).item()


# --------------------------------------------------------------------- AC9
class TestAC9PoolLimit:
    def _heads(self, monkeypatch, heads=24):
        monkeypatch.setattr(cli_module, "_model_num_attention_heads", lambda p: heads)

    def test_ac9_auto_equals_todays_value_for_the_first_pick_shape(self, monkeypatch):
        self._heads(monkeypatch)
        assert cli_module._derive_cache_limit_gb("m", 262144, 512) == 9.0
        assert cli_module._derive_cache_limit_gb("m", 262144, 512, policy=None) == 9.0

    def test_ac9_fused_v1_derivation_equals_auto_amendment_2_g3(self, monkeypatch):
        self._heads(monkeypatch)
        policy = ap.resolve_policy("fused_v1")
        got = cli_module._derive_cache_limit_gb("m", 262144, 512, policy=policy)
        assert got == 9.0  # unchanged: the batched path still runs unfused
        got = cli_module._derive_cache_limit_gb("m", 262144, 1024, policy=policy)
        assert got == cli_module._derive_cache_limit_gb("m", 262144, 1024)

    def test_ac9_explicit_cache_limit_still_overrides(self, monkeypatch):
        import mlx.core as real_mx

        calls = []
        monkeypatch.setattr(real_mx, "set_cache_limit", lambda n: calls.append(n))
        self._heads(monkeypatch)
        policy = ap.resolve_policy("fused_v1")
        cli_module._apply_mlx_memory_limits(
            9, 0, model_path="m", max_kv_size=262144, prefill_step=512, policy=policy
        )
        assert calls == [9 * 1024**3]
        calls.clear()
        cli_module._apply_mlx_memory_limits(
            0, 0, model_path="m", max_kv_size=262144, prefill_step=512, policy=policy
        )
        assert calls == [9 * 1024**3]
        calls.clear()
        cli_module._apply_mlx_memory_limits(
            0, 0, model_path="m", max_kv_size=262144, prefill_step=512
        )
        assert calls == [9 * 1024**3]
