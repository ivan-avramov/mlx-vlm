"""M58 `joint_v1` MTP verification scan policy (fork-only, CPU, no real model).

The joint call, the per-query decomposition and the plan mirror are exercised
with mocked or CPU-fallback SDPA; bit identity is a GPU property established by
STEP 1, the load-time self-test and gate 1, never by these tests.
"""

import itertools
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.base as base
from mlx_vlm import mtp_verify_scan as mv
from mlx_vlm.models.qwen3_5 import language as qwen_language
from mlx_vlm.models.qwen3_5 import speculative_verifier as qwen_verifier
from mlx_vlm.server import cli as cli_module
from mlx_vlm.server import generation as generation_module
from test_attention_policy import (  # noqa: F401  (fixtures + helpers)
    NativeCache,
    _TQ,
    _attn_modules,
    _cpu_device,
    _tiny_lm,
    passthrough,
)

REAL_SDPA = mx.fast.scaled_dot_product_attention
REPO = Path(__file__).resolve().parents[2]

D = 256  # qualified head dim
V = 32  # the fused vector kernel's query bound on this device class


# ------------------------------------------------------------------ helpers
class ShapeRecorder:
    """Stand-in for mx.fast.scaled_dot_product_attention: records, returns lazy
    zeros of the query's shape (so concatenation works)."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, **kwargs):
        self.calls.append((q.shape, k.shape, kwargs))
        return mx.zeros(q.shape, q.dtype)

    @property
    def masks(self):
        return [c[2].get("mask") for c in self.calls]


@pytest.fixture
def rec(monkeypatch):
    r = ShapeRecorder()
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", r)
    return r


def stub_plan(n_keys, q_heads, kv_heads):
    """Per-device-class plan stand-in: thresholds 1024 / 8192 / 32768 / 65536."""
    if n_keys < 1024:
        return ("one_pass", 0)
    for limit, blocks in ((8192, 128), (32768, 256), (65536, 512)):
        if n_keys <= limit:
            return ("two_pass", blocks)
    return ("two_pass", 1024)


@pytest.fixture
def qualified(monkeypatch):
    """Pretend: GPU class applegpu_g17s with the STEP 1 plan mirror."""
    monkeypatch.setattr(mv, "_device_class", lambda: "applegpu_g17s")
    monkeypatch.setattr(qwen_language, "_qwen3_5_sdpa_vector_plan", stub_plan)
    return stub_plan


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(mv, "_gpu_enabled", lambda: True)


def make_qkv(length, key_length, *, gqa=6, batch=1, dtype=mx.bfloat16, head_dim=D):
    q = mx.zeros((batch, gqa, length, head_dim), dtype)
    k = mx.zeros((batch, 1, key_length, head_dim), dtype)
    return q, k, mx.zeros((batch, 1, key_length, head_dim), dtype)


def attend(policy, q, k, v, cache=None, mask="causal", scale=0.0625):
    return policy.attend(
        queries=q, keys=k, values=v,
        cache=cache if cache is not None else NativeCache(), scale=scale, mask=mask,
    )


# ------------------------------------------------------------------ AC1
class TestAC1DefaultPreservation:
    def test_ac1_per_query_names_resolve_to_no_policy_object(self):
        for name in (None, "", "per_query"):
            assert mv.resolve_policy(name) is None

    @pytest.mark.parametrize("length", [3, 4, 5])
    def test_ac1_verifier_branch_calls_sdpa_exactly_as_today(
        self, monkeypatch, length
    ):
        calls = []

        def spy(queries, keys, values, **kwargs):
            calls.append((queries.shape, keys.shape, kwargs))
            return mx.zeros(queries.shape, queries.dtype)

        monkeypatch.setattr(qwen_verifier, "scaled_dot_product_attention", spy)
        key_length = 40
        q, k, v = make_qkv(length, key_length)
        mask = mx.ones((1, 1, length, key_length), dtype=mx.bool_)
        run_verifier_attention(q, k, v, mask)
        assert len(calls) == length
        prefix = key_length - length
        for index, (qs, ks, kw) in enumerate(calls):
            assert qs == (1, 6, 1, D)
            assert ks == (1, 1, prefix + index + 1, D)
            assert kw["mask"].shape == (1, 1, 1, prefix + index + 1)
            assert set(kw) == {"cache", "scale", "mask"}
        calls.clear()
        run_verifier_attention(q, k, v, "causal")
        assert [kw["mask"] for _, _, kw in calls] == [None] * length

    def test_ac1_length2_branch_is_the_shipped_joint_call(self, monkeypatch):
        calls = []

        def spy(queries, keys, values, **kwargs):
            calls.append(kwargs)
            return mx.zeros(queries.shape, queries.dtype)

        monkeypatch.setattr(qwen_verifier, "scaled_dot_product_attention", spy)
        q, k, v = make_qkv(2, 40)
        run_verifier_attention(q, k, v, "causal")
        assert len(calls) == 1 and calls[0]["mask"].shape == (1, 1, 2, 40)
        assert calls[0]["mask"].dtype == mx.bool_

    def test_ac1_default_policy_attribute_is_absent_on_attention(self):
        assert getattr(Qwen3_5Attention_cls(), "mtp_verify_policy", None) is None


def Qwen3_5Attention_cls():
    return qwen_language.Qwen3_5Attention.__new__(qwen_language.Qwen3_5Attention)


def run_verifier_attention(q, k, v, mask, policy=None, cache=None):
    """Drive Qwen3_5BatchInvariantForward._attention with the projections stubbed
    out, so the verifier hunk is exercised in isolation."""
    batch, _, length, _ = q.shape
    verifier = qwen_verifier.Qwen3_5BatchInvariantForward()
    verifier._linears = lambda linears, x: (x, x, x)
    verifier._linear = lambda linear, x: x
    gate = mx.zeros((batch, length, q.shape[1] * q.shape[3]), q.dtype)
    attention = SimpleNamespace(
        q_proj=None, k_proj=None, v_proj=None, o_proj=None, scale=0.0625,
        _prepare_projected_qkv=lambda *a: (q, k, v, gate, mask),
    )
    if policy is not None:
        attention.mtp_verify_policy = policy
    x = mx.zeros((batch, length, 8), q.dtype)
    return verifier._attention(
        attention, x, mask, cache if cache is not None else NativeCache(), None, None
    )


# ------------------------------------------------------------------ AC2
def _mask(kind, length, key_length, dtype):
    prefix = key_length - length
    if kind == "none":
        return None
    if kind == "causal":
        return "causal"
    if kind == "bool4":
        return (
            mx.arange(key_length)[None, None, None, :]
            < (prefix + mx.arange(length) + 1)[None, None, :, None]
        )
    if kind == "float4":
        return mx.zeros((1, 1, length, key_length), dtype)
    if kind == "bool2":
        return mx.ones((length, key_length), dtype=mx.bool_)
    if kind == "other":
        return "left_padded_decode"
    raise AssertionError(kind)


class _Quant:
    bits = 4
    group_size = 64


def oracle(length, cache, mask, gqa, batch, dtype, key_length):
    """Rules 1-7, written independently of the implementation."""
    if length < 2:
        return ("len1", None)
    if cache != "native" or batch != 1:
        return ("per_query", "cache" if cache != "native" else "batch")
    if mask not in ("none", "causal", "bool4"):
        return ("per_query", "mask_form")
    if key_length - length < 0:
        return ("per_query", "prefix")
    if length * gqa > V:
        return ("per_query", "gqa_bound")
    if dtype != mx.bfloat16 or gqa != 6:
        return ("per_query", "domain")
    if stub_plan(key_length - length + 1, gqa, 1) != stub_plan(key_length, gqa, 1):
        return ("straddle", None)
    return ("joint", None)


class TestAC2DecisionTable:
    @pytest.mark.parametrize("gqa", [6, 8])
    def test_ac2_joint_exactly_on_rules_1_to_7(self, qualified, rec, gqa):
        v_over_gqa = V // gqa
        lengths = sorted({1, 2, 3, 4, 5, v_over_gqa, v_over_gqa + 1, 9})
        caches = {"native": NativeCache, "quant": _Quant, "turbo": _TQ}
        masks = ["none", "causal", "bool4", "float4", "bool2", "other"]
        keys_ = [1024, 1025, 1026, 8193, 8194, 65537, 70000]
        cells = itertools.product(
            lengths, caches, masks, [1, 2], [mx.bfloat16, mx.float16, mx.float32], keys_
        )
        seen = {"joint": 0, "straddle": 0, "per_query": 0, "len1": 0}
        for length, cache_name, mask_kind, batch, dtype, key_length in cells:
            policy = mv.JointV1Policy()
            rec.calls.clear()
            q, k, v = make_qkv(length, key_length, gqa=gqa, batch=batch, dtype=dtype)
            if cache_name == "quant":
                k = v = (k, k, k)
            mask = _mask(mask_kind, length, key_length, dtype)
            out = attend(policy, q, k, v, caches[cache_name](), mask)
            route, reason = oracle(
                length, cache_name, mask_kind, gqa, batch, dtype, key_length
            )
            cell = (length, cache_name, mask_kind, gqa, batch, dtype, key_length)
            counters = policy.counters()
            seen[route] += 1
            if route == "joint":
                assert out is not None and len(rec.calls) == 1, cell
                want = "causal" if mask_kind in ("none", "causal") else None
                if want:
                    assert rec.masks[0] == "causal", cell
                else:
                    assert rec.masks[0].dtype == mx.bool_, cell
                assert counters["verify_blocks_joint_v1"] == 1, cell
            elif route == "straddle":
                assert out is not None and len(rec.calls) == length, cell
                assert counters["verify_blocks_straddle"] == 1, cell
                assert counters["verify_blocks_joint_v1"] == 0, cell
            else:
                assert out is None and rec.calls == [], cell  # today's code runs
                if route == "len1":
                    assert counters["verify_blocks_len1"] == 1, cell
                else:
                    assert counters["verify_blocks_per_query"] == 1, cell
                    assert counters["verify_fallback_reasons"] == {reason: 1}, cell
        if gqa == 6:
            assert all(seen.values()), seen  # every route was exercised
        else:
            assert seen["joint"] == seen["straddle"] == 0, seen  # outside the domain

    def test_ac2_rule1_admits_length_two(self, qualified, rec):
        policy = mv.JointV1Policy()
        q, k, v = make_qkv(2, 4096)
        assert attend(policy, q, k, v) is not None
        assert policy.counters()["verify_blocks_joint_v1"] == 1

    def test_ac2_rule4_prefix_must_be_non_negative(self, qualified, rec):
        policy = mv.JointV1Policy()
        q, k, v = make_qkv(4, 3)
        assert attend(policy, q, k, v) is None
        assert policy.counters()["verify_fallback_reasons"] == {"prefix": 1}

    def test_ac2_rule5_bound_is_length_times_gqa_at_most_32(self, qualified, rec):
        policy = mv.JointV1Policy()
        q, k, v = make_qkv(5, 4096)
        assert attend(policy, q, k, v) is not None  # 5 * 6 = 30
        q, k, v = make_qkv(6, 4096)
        assert attend(policy, q, k, v) is None  # 6 * 6 = 36
        assert policy.counters()["verify_fallback_reasons"] == {"gqa_bound": 1}

    def test_ac2_unqualified_device_class_falls_back_with_domain_reason(
        self, monkeypatch, rec
    ):
        monkeypatch.setattr(mv, "_device_class", lambda: "applegpu_g16g")
        monkeypatch.setattr(qwen_language, "_qwen3_5_sdpa_vector_plan", stub_plan)
        policy = mv.JointV1Policy()
        q, k, v = make_qkv(3, 4096)
        assert attend(policy, q, k, v) is None
        assert policy.counters()["verify_fallback_reasons"] == {"domain": 1}

    def test_ac2_float_mask_and_two_d_mask_never_go_joint(self, qualified, rec):
        policy = mv.JointV1Policy()
        q, k, v = make_qkv(3, 4096)
        for kind in ("float4", "bool2", "other"):
            assert attend(policy, q, k, v, mask=_mask(kind, 3, 4096, mx.bfloat16)) is None
        assert policy.counters()["verify_fallback_reasons"] == {"mask_form": 3}

    def test_ac2_bool_mask_too_small_falls_back(self, qualified, rec):
        policy = mv.JointV1Policy()
        q, k, v = make_qkv(3, 4096)
        small = mx.ones((1, 1, 2, 4096), dtype=mx.bool_)
        assert attend(policy, q, k, v, mask=small) is None
        short = mx.ones((1, 1, 3, 100), dtype=mx.bool_)
        assert attend(policy, q, k, v, mask=short) is None
        assert policy.counters()["verify_fallback_reasons"] == {"mask_form": 2}

    def test_ac2_straddle_counter_and_per_query_serving(self, qualified, rec):
        policy = mv.JointV1Policy()
        q, k, v = make_qkv(3, 1025)  # keys 1023..1025 cross the 1024 threshold
        out = attend(policy, q, k, v)
        assert out is not None and len(rec.calls) == 3
        assert [c[1][2] for c in rec.calls] == [1023, 1024, 1025]
        assert policy.counters()["verify_blocks_straddle"] == 1
        assert policy.counters()["verify_fallback_reasons"] == {}

    def test_ac2_each_threshold_is_protected(self, qualified, rec):
        # first key length that the stub plan sends to the next block count
        for first_new in (1024, 8193, 32769, 65537):
            assert stub_plan(first_new - 1, 6, 1) != stub_plan(first_new, 6, 1)
            policy = mv.JointV1Policy()
            q, k, v = make_qkv(3, first_new + 1)  # per-query keys: new-1, new, new+1
            attend(policy, q, k, v)
            assert policy.counters()["verify_blocks_straddle"] == 1, first_new
            policy = mv.JointV1Policy()
            q, k, v = make_qkv(3, first_new - 8)  # entirely below
            attend(policy, q, k, v)
            assert policy.counters()["verify_blocks_joint_v1"] == 1, first_new
            policy = mv.JointV1Policy()
            q, k, v = make_qkv(3, first_new + 2)  # entirely at/above
            attend(policy, q, k, v)
            assert policy.counters()["verify_blocks_joint_v1"] == 1, first_new


# ------------------------------------------------------------------ AC3
class TestAC3aMaskExactness:
    @pytest.mark.parametrize("length", [3, 4, 5])
    @pytest.mark.parametrize("key_length", [64, 1024])
    def test_ac3a_step_mask_true_set_equals_per_query_prefixes(
        self, length, key_length
    ):
        step = mv.step_mask(length, key_length)
        prefix = key_length - length
        assert step.shape == (1, 1, length, key_length)
        for index in range(length):
            row = [i for i in range(key_length) if step[0, 0, index, i].item()]
            assert row == list(range(prefix + index + 1))

    @pytest.mark.parametrize("length", [3, 4, 5])
    @pytest.mark.parametrize("key_length", [64, 1024])
    @pytest.mark.parametrize("kind", ["none", "causal", "bool4"])
    def test_ac3a_joint_mask_forms(self, qualified, rec, length, key_length, kind):
        # classify() exposes the mask the joint call would receive, straddling or not
        policy = mv.JointV1Policy()
        q, k, v = make_qkv(length, key_length)
        mask = _mask(kind, length, key_length, mx.bfloat16)
        served = policy.classify(q, k, v, NativeCache(), mask).joint_mask
        if kind in ("none", "causal"):
            assert served == "causal"  # STEP 1 amendment: string form served
        else:
            expect = mv.step_mask(length, key_length) & mask[..., :length, :key_length]
            assert mx.array_equal(served, expect).item()
        if key_length == 64:  # not straddling: the served call carries exactly it
            attend(policy, q, k, v, mask=mask)
            if isinstance(served, str):
                assert rec.masks == [served]
            else:
                assert mx.array_equal(rec.masks[0], served).item()

    def test_ac3a_bool_mask_is_anded_with_the_step_mask(self, qualified, rec):
        length, key_length = 3, 64
        policy = mv.JointV1Policy()
        q, k, v = make_qkv(length, key_length)
        everything = mx.ones((1, 1, length, key_length), dtype=mx.bool_)
        got = policy.classify(q, k, v, NativeCache(), everything).joint_mask
        assert mx.array_equal(got, mv.step_mask(length, key_length)).item()
        padded = everything & (mx.arange(key_length)[None, None, None, :] >= 4)
        got = policy.classify(q, k, v, NativeCache(), padded).joint_mask
        assert mx.array_equal(got, mv.step_mask(length, key_length) & padded).item()
        assert not got[0, 0, 0, :4].any().item()

    @pytest.mark.parametrize("kind", ["empty_row", "holes"])
    def test_ac3a_all_masked_row_and_non_contiguous_masks_fall_back(
        self, qualified, rec, kind
    ):
        length, key_length = 3, 64
        m = mv.step_mask(length, key_length)
        if kind == "empty_row":
            m = m & (mx.arange(length)[None, None, :, None] != 1)
        else:
            m = m & (mx.arange(key_length)[None, None, None, :] != 10)
        policy = mv.JointV1Policy()
        q, k, v = make_qkv(length, key_length)
        assert attend(policy, q, k, v, mask=m) is None
        assert policy.counters()["verify_fallback_reasons"] == {"mask_rows": 1}

    def test_ac3a_causal_leakage_future_keys_never_change_earlier_queries(self):
        mx.random.seed(3)
        length, key_length = 4, 40
        q = mx.random.normal((1, 6, length, 16)).astype(mx.float32)
        k = mx.random.normal((1, 1, key_length, 16)).astype(mx.float32)
        v = mx.random.normal((1, 1, key_length, 16)).astype(mx.float32)
        prefix = key_length - length
        for mask in ("causal", mv.step_mask(length, key_length)):
            ref = REAL_SDPA(q, k, v, scale=0.25, mask=mask)
            for index in range(length):
                pos = prefix + index + 1  # first key the query must not see
                if pos >= key_length:
                    continue
                k2 = k.at[:, :, pos:, :].add(5.0)
                v2 = v.at[:, :, pos:, :].add(5.0)
                got = REAL_SDPA(q, k2, v2, scale=0.25, mask=mask)
                assert mx.allclose(got[:, :, : index + 1], ref[:, :, : index + 1]).item()


class TestAC3bNumerics:
    @pytest.mark.parametrize("length", [2, 3, 4, 5])
    @pytest.mark.parametrize("key_length", [64, 1024])
    def test_ac3b_joint_and_per_query_match_an_fp32_reference(
        self, length, key_length
    ):
        mx.random.seed(length)
        q = mx.random.normal((1, 6, length, D)).astype(mx.bfloat16)
        k = mx.random.normal((1, 1, key_length, D)).astype(mx.bfloat16)
        v = mx.random.normal((1, 1, key_length, D)).astype(mx.bfloat16)
        cache = NativeCache()
        joint = mv.joint_attention(q, k, v, cache=cache, scale=D**-0.5, mask="causal")
        per_q = mv.per_query_attention(q, k, v, cache=cache, scale=D**-0.5, mask=None)
        q32, k32, v32 = (a.astype(mx.float32) for a in (q, k, v))
        prefix = key_length - length
        ref = mx.concatenate(
            [
                REAL_SDPA(
                    q32[:, :, i : i + 1], k32[:, :, : prefix + i + 1],
                    v32[:, :, : prefix + i + 1], scale=D**-0.5, mask=None,
                )
                for i in range(length)
            ],
            axis=2,
        )
        # bf16 inputs: tolerance stated at 3 bf16 ulps of the reference scale.
        for got in (joint, per_q):
            assert mx.allclose(got.astype(mx.float32), ref, atol=3e-2, rtol=3e-2).item()
        # NOT asserted: bit identity (the CPU fallback is not bitwise; GPU only).


# ------------------------------------------------------------------ AC4
class TestAC4Propagation:
    def test_ac4_cli_flags_reach_the_worker_environment(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            cli_module.uvicorn, "run", lambda *a, **k: seen.setdefault("ran", True)
        )
        monkeypatch.setattr(cli_module, "_apply_mlx_memory_limits", lambda *a, **k: None)
        monkeypatch.setattr(cli_module, "_configure_session_manager", lambda **k: None)
        monkeypatch.delenv("MLX_VLM_MTP_VERIFY_SCAN", raising=False)
        monkeypatch.delenv("MLX_VLM_MTP_VERIFY_AB", raising=False)
        import os

        monkeypatch.setattr(
            sys, "argv",
            ["mlx_vlm.server", "--mtp-verify-scan", "joint_v1", "--mtp-verify-ab"],
        )
        cli_module.main()
        assert os.environ["MLX_VLM_MTP_VERIFY_SCAN"] == "joint_v1"
        assert os.environ["MLX_VLM_MTP_VERIFY_AB"] == "1"
        monkeypatch.setattr(sys, "argv", ["mlx_vlm.server"])
        cli_module.main()
        assert os.environ["MLX_VLM_MTP_VERIFY_SCAN"] == "per_query"  # no stale value
        assert os.environ["MLX_VLM_MTP_VERIFY_AB"] == "0"

    def _apply(self, monkeypatch, lm, *, scan="joint_v1", ab="0", kind="mtp", kv=None):
        monkeypatch.setenv("MLX_VLM_MTP_VERIFY_SCAN", scan)
        monkeypatch.setenv("MLX_VLM_MTP_VERIFY_AB", ab)
        monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
        model = SimpleNamespace(language_model=lm, config=lm.config)
        return model, generation_module._apply_mtp_verify_from_env(
            model, draft_kind=kind, kv_bits=kv
        )

    @pytest.fixture
    def tiny_domain(self, monkeypatch, qualified, gpu):
        """The tiny model's shape (GQA 2, head dim 16) stands in for the domain."""
        monkeypatch.setattr(
            mv, "QUALIFIED_DOMAIN",
            mv.Domain(dtypes=(mx.bfloat16,), head_dim=16, gqa=(2,),
                      device_classes=("applegpu_g17s",)),
        )
        monkeypatch.setattr(mv, "self_test_model", lambda *a, **k: [mv.SelfTestResult()])

    def test_ac4_env_to_instance_to_modules_to_branch_to_counters(
        self, monkeypatch, tiny_domain, rec
    ):
        lm = _tiny_lm()
        model, policy = self._apply(monkeypatch, lm)
        assert isinstance(policy, mv.JointV1Policy)
        assert lm.mtp_verify_policy is policy and model.mtp_verify_policy is policy
        modules = _attn_modules(lm)
        assert modules and all(m.mtp_verify_policy is policy for m in modules)
        q, k, v = make_qkv(3, 64, gqa=2, head_dim=16)
        before = policy.snapshot()
        run_verifier_attention(q, k, v, "causal", policy=modules[0].mtp_verify_policy)
        assert len(rec.calls) == 1  # joint, one call
        assert policy.since(before)["verify_blocks_joint_v1"] == 1

    def test_ac4_two_instances_dispatch_per_instance(self, qualified, rec, monkeypatch):
        monkeypatch.setattr(
            mv, "QUALIFIED_DOMAIN",
            mv.Domain(dtypes=(mx.bfloat16,), head_dim=16, gqa=(2,),
                      device_classes=("applegpu_g17s",)),
        )
        joint = mv.JointV1Policy()
        q, k, v = make_qkv(3, 64, gqa=2, head_dim=16)
        run_verifier_attention(q, k, v, "causal", policy=joint)
        assert len(rec.calls) == 1
        rec.calls.clear()
        run_verifier_attention(q, k, v, "causal", policy=None)  # a plain instance
        assert len(rec.calls) == 3  # today's per-query calls
        rec.calls.clear()
        other = mv.JointV1Policy()
        run_verifier_attention(q, k, v, "causal", policy=other)
        assert joint.counters()["verify_blocks_joint_v1"] == 1
        assert other.counters()["verify_blocks_joint_v1"] == 1

    def test_ac4_environment_change_after_load_has_no_effect(
        self, monkeypatch, tiny_domain, rec
    ):
        lm = _tiny_lm()
        _, policy = self._apply(monkeypatch, lm)
        monkeypatch.setenv("MLX_VLM_MTP_VERIFY_SCAN", "per_query")
        q, k, v = make_qkv(3, 64, gqa=2, head_dim=16)
        run_verifier_attention(q, k, v, "causal", policy=_attn_modules(lm)[0].mtp_verify_policy)
        assert len(rec.calls) == 1

    def test_ac4_the_verifier_singleton_carries_no_state(self):
        before = dict(vars(qwen_verifier.Qwen3_5BatchInvariantForward()))
        assert before == {}
        assert not hasattr(qwen_verifier.Qwen3_5BatchInvariantForward, "mtp_verify_policy")
        assert "mtp_verify_policy" not in vars(qwen_language.Qwen3_5Attention) or (
            vars(qwen_language.Qwen3_5Attention)["mtp_verify_policy"] is None
        )

    def test_ac4_per_query_is_never_stamped(self, monkeypatch, tiny_domain):
        lm = _tiny_lm()
        model, policy = self._apply(monkeypatch, lm, scan="per_query")
        assert policy is None
        assert not hasattr(model, "mtp_verify_policy")
        assert all(getattr(m, "mtp_verify_policy", None) is None for m in _attn_modules(lm))


# ------------------------------------------------------------------ AC5
ALLOWED = {
    "mlx_vlm/mtp_verify_scan.py",
    "mlx_vlm/models/qwen3_5/speculative_verifier.py",
    "mlx_vlm/server/cli.py",
    "mlx_vlm/server/generation.py",
    "mlx_vlm/server/schemas.py",
    "mlx_vlm/server/openai.py",
    "mlx_vlm/server/anthropic.py",
}
UNTOUCHED = [
    "mlx_vlm/models/nemotron_h",
    "mlx_vlm/models/gemma4",
    "mlx_vlm/models/lfm2",
    "mlx_vlm/models/glm_moe_dsa",
    "mlx_vlm/models/base.py",
    "mlx_vlm/models/quantized_verifier.py",
    "mlx_vlm/models/qwen3_5/language.py",
]


def _git(*args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


class TestAC5Scope:
    def test_ac5_diff_touches_only_the_listed_files(self):
        changed = set(_git("diff", "--name-only", "main").split())
        extra = {
            f for f in changed
            if f not in ALLOWED and not f.startswith("mlx_vlm/tests/")
        }
        assert not extra, extra

    def test_ac5_other_verifiers_helper_and_quantized_branches_are_untouched(self):
        assert _git("diff", "main", "--", *UNTOUCHED) == ""

    def test_ac5_every_hunk_in_upstream_owned_files_is_marked(self):
        owned = sorted(ALLOWED - {"mlx_vlm/mtp_verify_scan.py"})
        diff = _git("diff", "-U0", "main", "--", *owned)
        hunks, current = [], None
        for line in diff.splitlines():
            if line.startswith("@@"):
                current = []
                hunks.append(current)
            elif current is not None and line.startswith("+") and not line.startswith("+++"):
                current.append(line)
        # a hunk may be a pure deletion; every added-line hunk needs the marker
        unmarked = [h for h in hunks if h and not any("Fork (M58)" in l for l in h)]
        assert not unmarked, unmarked[:3]

    def test_ac5_verifier_keeps_its_sdpa_calls_and_never_passes_policy_keywords(self):
        path = "mlx_vlm/models/qwen3_5/speculative_verifier.py"
        now = (REPO / path).read_text()
        main = _git("show", f"main:{path}")
        assert now.count("scaled_dot_product_attention(") == main.count(
            "scaled_dot_product_attention("
        )
        assert "force_fused" not in now and "policy=" not in now


# ------------------------------------------------------------------ AC6
class TestAC6LoudFailure:
    def test_ac6_unknown_value_is_rejected(self):
        with pytest.raises(ValueError, match="bogus"):
            mv.resolve_policy("bogus")

    def test_ac6_worker_argparse_rejects_unknown_value(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["mlx_vlm.server", "--mtp-verify-scan", "bogus"])
        with pytest.raises(SystemExit) as exc:
            cli_module.main()
        assert exc.value.code != 0

    def test_ac6_ab_without_joint_v1_is_refused_at_the_cli_and_at_resolve(
        self, monkeypatch
    ):
        monkeypatch.setattr(sys, "argv", ["mlx_vlm.server", "--mtp-verify-ab"])
        with pytest.raises(SystemExit) as exc:
            cli_module.main()
        assert exc.value.code != 0
        with pytest.raises(ValueError, match="joint_v1"):
            mv.resolve_policy("per_query", ab=True)

    def _env(self, monkeypatch, **kw):
        monkeypatch.setenv("MLX_VLM_MTP_VERIFY_SCAN", kw.get("scan", "joint_v1"))
        monkeypatch.setenv("MLX_VLM_MTP_VERIFY_AB", kw.get("ab", "0"))
        monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)

    def _call(self, model, kind="mtp", kv=None):
        return generation_module._apply_mtp_verify_from_env(
            model, draft_kind=kind, kv_bits=kv
        )

    def test_ac6_ab_env_without_joint_v1_refuses(self, monkeypatch, gpu):
        self._env(monkeypatch, scan="per_query", ab="1")
        lm = _tiny_lm()
        with pytest.raises(mv.VerifyScanError, match="joint_v1"):
            self._call(SimpleNamespace(language_model=lm, config=lm.config))

    def test_ac6_unsupported_family_refuses_with_one_stderr_line(
        self, monkeypatch, capsys, gpu
    ):
        self._env(monkeypatch)
        model = SimpleNamespace(
            language_model=nn.Sequential(nn.Linear(2, 2)),
            config=SimpleNamespace(model_type="llama"),
        )
        with pytest.raises(mv.VerifyScanError):
            self._call(model)
        err = [l for l in capsys.readouterr().err.splitlines() if l.strip()]
        assert len(err) == 1 and "joint_v1" in err[0]

    def test_ac6_quantized_kv_refuses(self, monkeypatch, gpu):
        self._env(monkeypatch)
        lm = _tiny_lm()
        with pytest.raises(mv.VerifyScanError, match="quantized"):
            self._call(SimpleNamespace(language_model=lm, config=lm.config), kv=4.0)

    @pytest.mark.parametrize("kind", [None, "suffix", "dflash"])
    def test_ac6_without_draft_kind_mtp_refuses(self, monkeypatch, gpu, kind):
        self._env(monkeypatch)
        lm = _tiny_lm()
        with pytest.raises(mv.VerifyScanError, match="mtp"):
            self._call(SimpleNamespace(language_model=lm, config=lm.config), kind=kind)

    def test_ac6_non_gpu_default_device_refuses(self, monkeypatch):
        self._env(monkeypatch)
        lm = _tiny_lm()  # CPU default device in these tests
        with pytest.raises(mv.VerifyScanError, match="GPU"):
            self._call(SimpleNamespace(language_model=lm, config=lm.config))

    def test_ac6_mlx_sdpa_blocks_env_refuses(self, monkeypatch, gpu):
        self._env(monkeypatch)
        monkeypatch.setenv("MLX_SDPA_BLOCKS", "64")
        lm = _tiny_lm()
        with pytest.raises(mv.VerifyScanError, match="MLX_SDPA_BLOCKS"):
            self._call(SimpleNamespace(language_model=lm, config=lm.config))

    def test_ac6_per_query_never_refuses_any_family(self, monkeypatch):
        self._env(monkeypatch, scan="per_query")
        model = SimpleNamespace(
            language_model=nn.Sequential(nn.Linear(2, 2)),
            config=SimpleNamespace(model_type="llama"),
        )
        assert self._call(model, kind=None, kv=4.0) is None

    def test_ac6_self_test_failure_fails_the_load(self, monkeypatch, gpu, qualified):
        self._env(monkeypatch)
        monkeypatch.setattr(
            mv, "QUALIFIED_DOMAIN",
            mv.Domain(dtypes=(mx.bfloat16,), head_dim=16, gqa=(2,),
                      device_classes=("applegpu_g17s",)),
        )

        def boom(*a, **k):
            raise mv.VerifyScanError("joint_v1 self-test: boom")

        monkeypatch.setattr(mv, "self_test_model", boom)
        lm = _tiny_lm()
        with pytest.raises(mv.VerifyScanError, match="self-test"):
            self._call(SimpleNamespace(language_model=lm, config=lm.config))

    def test_ac6_initialize_model_raises_so_the_worker_never_reports_ready(
        self, monkeypatch, gpu
    ):
        self._env(monkeypatch)
        model = SimpleNamespace(
            language_model=nn.Sequential(nn.Linear(2, 2)),
            config=SimpleNamespace(model_type="llama"),
        )
        monkeypatch.setattr(
            generation_module, "load_model_resources",
            lambda *a, **k: (model, SimpleNamespace(), model.config),
        )
        monkeypatch.delenv("MLX_VLM_MOE_EXPAND", raising=False)
        monkeypatch.delenv("MLX_VLM_ATTENTION_POLICY", raising=False)
        monkeypatch.setenv("MLX_VLM_DRAFT_KIND", "mtp")
        monkeypatch.delenv("MLX_VLM_DRAFT_MODEL", raising=False)
        fake = SimpleNamespace(
            model_path="x", adapter_path=None, draft_kind_override=None,
            draft_model_path=None, apc_manager=None, kv_bits=None,
        )
        with pytest.raises(mv.VerifyScanError, match="qualified"):
            generation_module.ResponseGenerator._initialize_model(fake)
        # (the readiness / lifespan propagation is the M57 F4 test's, unchanged)


# ------------------------------------------------------------------ AC7
def _two_pass_cells():
    return make_qkv(3, 4096)


class TestAC7ABInstrument:
    def test_ac7_both_paths_run_and_the_joint_output_is_served(
        self, qualified, monkeypatch
    ):
        order = []
        joint_out = mx.ones((1, 6, 3, D), dtype=mx.bfloat16)
        pq_out = mx.zeros((1, 6, 3, D), dtype=mx.bfloat16)
        monkeypatch.setattr(
            mv, "joint_attention", lambda *a, **k: order.append("joint") or joint_out
        )
        monkeypatch.setattr(
            mv, "per_query_attention", lambda *a, **k: order.append("pq") or pq_out
        )
        policy = mv.JointV1Policy(ab=True)
        q, k, v = _two_pass_cells()
        out = attend(policy, q, k, v)
        assert out is joint_out
        assert sorted(order) == ["joint", "pq"]
        policy.end_block()
        c = policy.counters()
        assert c["verify_ab_blocks"] == 1 and c["verify_ab_mismatch"] == 1

    def test_ac7_the_joint_call_is_the_production_joint_function(
        self, qualified, monkeypatch
    ):
        calls = []
        real = mv.joint_attention
        monkeypatch.setattr(
            mv, "joint_attention",
            lambda *a, **k: calls.append(1) or real(*a, **k),
        )
        monkeypatch.setattr(
            mv, "per_query_attention",
            lambda q, *a, **k: mx.zeros(q.shape, q.dtype),
        )
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", ShapeRecorder())
        for ab in (False, True):
            calls.clear()
            attend(mv.JointV1Policy(ab=ab), *_two_pass_cells())
            assert calls == [1]  # one joint call, same function object either way

    def test_ac7_identical_outputs_count_no_mismatch(self, qualified, monkeypatch, rec):
        policy = mv.JointV1Policy(ab=True)
        attend(policy, *_two_pass_cells())
        policy.end_block()
        c = policy.counters()
        assert (c["verify_ab_blocks"], c["verify_ab_mismatch"]) == (1, 0)

    def test_ac7_injected_mismatch_is_counted_and_logged_once_per_shape(
        self, qualified, monkeypatch, caplog, capsys
    ):
        real = mv.per_query_attention
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", ShapeRecorder())
        monkeypatch.setattr(
            mv, "per_query_attention",
            lambda q, *a, **k: mx.ones(q.shape, q.dtype),  # differs from zeros
        )
        policy = mv.JointV1Policy(ab=True)
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                attend(policy, *_two_pass_cells())
            policy.end_block()
            attend(policy, *make_qkv(4, 4096))  # a new shape
            policy.end_block()
        c = policy.counters()
        assert c["verify_ab_blocks"] == 4 and c["verify_ab_mismatch"] == 4
        lines = [r.getMessage() for r in caplog.records if "mismatch" in r.getMessage()]
        assert len(lines) == 2  # once per distinct (length, key_length, dtype)
        assert "max_abs" in lines[0] and "4096" in lines[0]
        assert real is not mv.per_query_attention

    def test_ac7_shape_dtype_and_non_finite_count_as_mismatches(
        self, qualified, monkeypatch
    ):
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", ShapeRecorder())
        policy = mv.JointV1Policy(ab=True)
        base_out = mx.zeros((1, 6, 3, D), dtype=mx.bfloat16)
        variants = [
            mx.zeros((1, 6, 3, D), dtype=mx.float32),  # dtype
            mx.zeros((1, 6, 2, D), dtype=mx.bfloat16),  # shape
            mx.full((1, 6, 3, D), float("nan"), dtype=mx.bfloat16),  # non-finite
        ]
        for other in variants:
            monkeypatch.setattr(mv, "joint_attention", lambda *a, **k: base_out)
            monkeypatch.setattr(mv, "per_query_attention", lambda *a, _o=other, **k: _o)
            attend(policy, *_two_pass_cells())
        policy.end_block()
        assert policy.counters()["verify_ab_mismatch"] == 3

    def test_ac7_comparison_is_bitwise_not_value_equal(self, qualified, monkeypatch):
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", ShapeRecorder())
        plus = mx.zeros((1, 6, 3, D), dtype=mx.bfloat16)
        minus = -plus  # -0 == +0 as values, different bits
        assert bool(mx.array_equal(plus, minus).item())
        monkeypatch.setattr(mv, "joint_attention", lambda *a, **k: plus)
        monkeypatch.setattr(mv, "per_query_attention", lambda *a, **k: minus)
        policy = mv.JointV1Policy(ab=True)
        attend(policy, *_two_pass_cells())
        policy.end_block()
        assert policy.counters()["verify_ab_mismatch"] == 1

    def test_ac7_accumulation_is_lazy_and_materialised_before_counters_advance(
        self, qualified, monkeypatch
    ):
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", ShapeRecorder())
        policy = mv.JointV1Policy(ab=True)
        evals = []
        real_eval = mx.eval
        monkeypatch.setattr(mv.mx, "eval", lambda *a: evals.append(1) or real_eval(*a))
        for _ in range(5):
            attend(policy, *_two_pass_cells())
        assert evals == []  # nothing materialised per layer
        assert policy._counts["verify_ab_mismatch"] == 0
        policy.end_block()
        assert len(evals) == 1  # one materialisation per round
        assert policy.counters()["verify_ab_blocks"] == 5

    def test_ac7_straddle_blocks_are_shadow_computed_into_their_own_counters(
        self, qualified, monkeypatch
    ):
        pq_out = mx.ones((1, 6, 3, D), dtype=mx.bfloat16)
        joint_out = mx.zeros((1, 6, 3, D), dtype=mx.bfloat16)
        monkeypatch.setattr(mv, "joint_attention", lambda *a, **k: joint_out)
        monkeypatch.setattr(mv, "per_query_attention", lambda *a, **k: pq_out)
        policy = mv.JointV1Policy(ab=True)
        q, k, v = make_qkv(3, 1025)
        out = attend(policy, q, k, v)
        assert out is pq_out  # straddle: today's per-query numerics are served
        policy.end_block()
        c = policy.counters()
        assert c["verify_ab_straddle_blocks"] == 1
        assert c["verify_ab_straddle_mismatch"] == 1
        assert c["verify_ab_blocks"] == 0 and c["verify_ab_mismatch"] == 0

    def test_ac7_ab_keys_are_absent_without_ab(self, qualified):
        c = mv.JointV1Policy(ab=False).counters()
        assert not any(k.startswith("verify_ab_") for k in c)
        assert {"verify_blocks_joint_v1", "verify_blocks_per_query",
                "verify_blocks_straddle", "verify_blocks_len1",
                "verify_fallback_reasons"} <= set(c)


# ------------------------------------------------------------------ self-test
class TestSelfTest:
    DIMS = dict(heads=6, kv_heads=1, head_dim=D, dtype=mx.bfloat16)

    class FakeSdpa:
        """A kernel stand-in: the result depends on the pass plan of the key
        length each call sees, like MLX's block count (a straddle mismatches)."""

        def __call__(self, q, k, v, scale=None, mask=None, **kw):
            plan = stub_plan(k.shape[2], q.shape[1], k.shape[1])
            return mx.full(q.shape, float(plan[1]), q.dtype)

    def test_self_test_passes_with_a_consistent_known_positive(
        self, qualified, gpu, monkeypatch
    ):
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", self.FakeSdpa())
        result = mv.self_test(mv.JointV1Policy(), **self.DIMS)
        assert result.ran >= 2

    def test_self_test_fails_when_eligible_cell_is_not_identical(
        self, qualified, gpu, monkeypatch
    ):
        class Noisy(self.FakeSdpa):
            def __call__(self, q, k, v, scale=None, mask=None, **kw):
                return mx.full(q.shape, float(q.shape[2]), q.dtype)  # depends on qL

        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", Noisy())
        with pytest.raises(mv.VerifyScanError, match="self-test"):
            mv.self_test(mv.JointV1Policy(), **self.DIMS)

    def test_self_test_fails_when_the_straddle_is_predicted_but_does_not_occur(
        self, qualified, gpu, monkeypatch
    ):
        class Flat:
            def __call__(self, q, k, v, scale=None, mask=None, **kw):
                return mx.zeros(q.shape, q.dtype)  # never mismatches

        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", Flat())
        with pytest.raises(mv.VerifyScanError, match="straddle"):
            mv.self_test(mv.JointV1Policy(), **self.DIMS)

    def test_self_test_fails_on_an_unpredicted_mismatch(self, gpu, monkeypatch):
        monkeypatch.setattr(mv, "_device_class", lambda: "applegpu_g17s")
        # a mirror that predicts nothing, against a kernel that mismatches
        monkeypatch.setattr(
            qwen_language, "_qwen3_5_sdpa_vector_plan", lambda n, q, kv: ("one_pass", 0)
        )
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", self.FakeSdpa())
        with pytest.raises(mv.VerifyScanError, match="self-test"):
            mv.self_test(mv.JointV1Policy(), **self.DIMS)

    def test_self_test_refuses_a_non_gpu_device_and_a_raising_call(
        self, qualified, monkeypatch
    ):
        with pytest.raises(mv.VerifyScanError, match="GPU"):
            mv.self_test(mv.JointV1Policy(), **self.DIMS)

        def raising(*a, **k):
            raise ValueError("kernel exploded")

        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", raising)
        with pytest.raises(mv.VerifyScanError, match="self-test"):
            mv.self_test(mv.JointV1Policy(), force_calls=True, **self.DIMS)

    def test_self_test_requires_an_eligible_cell(self, qualified, gpu):
        with pytest.raises(mv.VerifyScanError, match="eligible"):
            mv.self_test(
                mv.JointV1Policy(), heads=8, kv_heads=1, head_dim=D, dtype=mx.bfloat16
            )

    def test_self_test_docstring_states_the_readiness_timeout_bounds_a_hang(self):
        assert "readiness timeout" in mv.self_test.__doc__

    def test_self_test_leaves_the_counters_untouched(self, qualified, gpu, monkeypatch):
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", self.FakeSdpa())
        policy = mv.JointV1Policy()
        mv.self_test(policy, **self.DIMS)
        assert policy.counters()["verify_blocks_joint_v1"] == 0
        assert policy.counters()["verify_blocks_straddle"] == 0


# ------------------------------------------------------------------ AC9
class TestAC9Counters:
    BASE = dict(
        endpoint="e", model="m", stream=False, backend="b", prompt_tokens=1,
        completion_tokens=1, generated_tokens=1, request_elapsed_s=1.0,
        request_started_s=0.0,
    )

    def test_ac9_request_scoping_snapshot_since(self, qualified, rec):
        policy = mv.JointV1Policy()
        attend(policy, *make_qkv(3, 4096))
        snap = policy.snapshot()
        attend(policy, *make_qkv(3, 4096))
        attend(policy, *make_qkv(3, 1025))
        attend(policy, *make_qkv(6, 4096))
        delta = policy.since(snap)
        assert delta["verify_blocks_joint_v1"] == 1
        assert delta["verify_blocks_straddle"] == 1
        assert delta["verify_blocks_per_query"] == 1
        assert delta["verify_fallback_reasons"] == {"gqa_bound": 1}
        assert policy.counters()["verify_blocks_joint_v1"] == 2  # lifetime intact

    def test_ac9_envelope_and_log_line_only_under_a_policy(self, caplog):
        plain = generation_module._build_metrics_envelope(**self.BASE)
        assert not any(k.startswith("verify_") for k in plain)
        counters = {
            "verify_blocks_joint_v1": 7, "verify_blocks_per_query": 1,
            "verify_blocks_straddle": 2, "verify_blocks_len1": 5,
            "verify_fallback_reasons": {"domain": 1},
        }
        env = generation_module._build_metrics_envelope(
            **self.BASE, verify_counters=counters
        )
        assert env["verify_blocks_joint_v1"] == 7
        assert env["verify_fallback_reasons"] == {"domain": 1}
        store = generation_module.ServerMetricsStore()
        with caplog.at_level(logging.INFO):
            store.record_success(env)
            store.record_success(plain)
        lines = [r.getMessage() for r in caplog.records
                 if "Request completed" in r.getMessage()]
        assert "verify_blocks_joint_v1=7" in lines[0]
        assert 'verify_fallback_reasons={"domain":1}' in lines[0]
        assert "verify_" not in lines[1]

    def test_ac9_streaming_token_metrics_and_timings(self):
        from mlx_vlm.server.schemas import GenerationTimings, StreamingTimings

        counters = {"verify_blocks_joint_v1": 4, "verify_fallback_reasons": {}}
        tok = generation_module.StreamingToken(
            text="", token=1, logprobs=0.0, finish_reason="stop",
            verify_counters=counters,
        )
        metrics = generation_module.GenerationMetrics()
        metrics.record_result(tok)
        assert metrics.verify_counters == counters
        dumped = json.loads(GenerationTimings.from_metrics(metrics, 10, 5).model_dump_json())
        assert dumped["verify_blocks_joint_v1"] == 4
        assert dumped["verify_fallback_reasons"] == {}
        assert "verify_counters" not in dumped
        bare = generation_module.GenerationMetrics()
        dumped = json.loads(GenerationTimings.from_metrics(bare, 10, 5).model_dump_json())
        assert not any(k.startswith("verify_") for k in dumped)
        assert "verify" not in StreamingTimings(predicted_per_second=1.0).model_dump_json()

    def test_ac9_streaming_timings_helper_carries_counters(self):
        import mlx_vlm.server.openai as openai_module

        metrics = SimpleNamespace(
            rate=3.0, sdpa_forced=None, sdpa_auto=None,
            verify_counters={"verify_blocks_joint_v1": 2},
        )
        dumped = json.loads(openai_module._streaming_timings(3.0, metrics).model_dump_json())
        assert dumped == {"predicted_per_second": 3.0, "verify_blocks_joint_v1": 2}
        bare = openai_module._streaming_timings(
            3.0, SimpleNamespace(sdpa_forced=None, sdpa_auto=None)
        )
        assert json.loads(bare.model_dump_json()) == {"predicted_per_second": 3.0}
        chunk = openai_module._completion_final_chunk("id", "m", 1, "stop", metrics)
        assert json.loads(chunk.to_sse_json())["timings"]["verify_blocks_joint_v1"] == 2


from fastapi.testclient import TestClient  # noqa: E402

import mlx_vlm.server as server  # noqa: E402
import mlx_vlm.server.session_manager as session_manager  # noqa: E402
from test_cached_tokens_reporting import _bare_response_generator  # noqa: E402


class _CountingPolicy:
    """Policy double: counters move during the stream; since() is the delta."""

    def __init__(self):
        self.n = 0

    def snapshot(self):
        return self.n

    def since(self, snap):
        return {"verify_blocks_joint_v1": self.n - snap, "verify_fallback_reasons": {}}


def _endpoint(monkeypatch, policy):
    from queue import Queue

    from mlx_vlm.generate.common import GenerationResult

    monkeypatch.setattr(session_manager, "_session_cache_max", 8)

    def _gen(**kwargs):
        for i in range(2):
            if policy is not None:
                policy.n += 6
            yield GenerationResult(
                text="x", token=100 + i, logprobs=None, prompt_tokens=56,
                generation_tokens=i + 1, total_tokens=57 + i, prompt_tps=400.0,
                generation_tps=170.0, peak_memory=1.5, cached_tokens=39,
                finish_reason="stop" if i else None,
            )

    monkeypatch.setattr(sys.modules["mlx_vlm.generate"], "stream_generate", _gen)
    rg = _bare_response_generator()
    rg.mtp_verify_policy = policy

    class Gen:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, *a, **kw):
            return None

        def _cpu_preprocess(self, *a, **kw):
            return {"input_ids": mx.zeros((1, 56), dtype=mx.int32)}

        def generate(self, prompt=None, images=None, audio=None, args=None, **kw):
            rqueue = Queue()
            rg._process_cached_request(
                rqueue=rqueue, prompt=prompt or "hi", images=None,
                args=args or server.GenerationArguments(), prompt_tokens=56,
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

    body = {"model": "demo", "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8}
    if stream:
        body["stream"] = True
    with patch.object(
        server, "get_cached_model",
        return_value=(SimpleNamespace(), SimpleNamespace(),
                      SimpleNamespace(model_type="qwen2_vl")),
    ):
        return client.post(
            "/v1/chat/completions", json=body,
            headers={session_manager._chat_id_header: "chat-m58" + str(stream)},
        )


class TestAC9Endpoints:
    @pytest.fixture
    def client(self):
        with TestClient(server.app) as c:
            yield c

    def test_ac9_non_streaming_timings_carry_counters_per_request(
        self, client, monkeypatch
    ):
        policy = _CountingPolicy()
        _endpoint(monkeypatch, policy)
        first = _post(client).json()["timings"]
        assert first["verify_blocks_joint_v1"] == 12
        second = _post(client).json()["timings"]
        assert second["verify_blocks_joint_v1"] == 12  # per-request reset, not lifetime

    def test_ac9_streaming_final_chunk_timings_carry_counters(self, client, monkeypatch):
        _endpoint(monkeypatch, _CountingPolicy())
        r = _post(client, stream=True)
        seen = []
        for line in r.read().decode().splitlines():
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                t = json.loads(line[6:]).get("timings") or {}
                if "verify_blocks_joint_v1" in t:
                    seen.append(t["verify_blocks_joint_v1"])
        assert seen == [12]

    def test_ac9_response_bytes_unchanged_under_per_query(self, client, monkeypatch):
        _endpoint(monkeypatch, None)
        t = _post(client).json()["timings"]
        assert not any(k.startswith("verify_") for k in t)
        r = _post(client, stream=True)
        assert "verify_" not in r.read().decode()
