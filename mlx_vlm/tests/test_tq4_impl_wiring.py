"""M72 1.3 A8 (fork side) and A3's refusal half: ``--tq4-attention`` -> ``MLX_VLM_TQ4_ATTENTION`` -> the server load
order (resolve -> tq4 require_environment -> attention policy -> MTP verify (kv_bits 4 accepted only with the mode) ->
check_fp16_range -> apply_to_model -> self_test_model), refusals at load (never a silent fallback), ``/health`` details
(mode + route counters, absent when the mode is off)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm import mtp_verify_scan as mv  # noqa: E402
from mlx_vlm import tq4_attention as t4  # noqa: E402

FULL = dict(
    kv_bits=4,
    kv_quant_scheme="turboquant",
    attention_policy="fused_v1",
    mtp_verify_scan="joint_v1",
)


# ================================================================== resolve / require_environment
@pytest.mark.parametrize("value", [None, "", "off"])
def test_resolve_mode_absent(value):
    assert t4.resolve_mode(value) is None


@pytest.mark.parametrize("value", t4.MODES)
def test_resolve_mode_values(value):
    assert t4.resolve_mode(value) == value


@pytest.mark.parametrize("value", ["m72", "on", "joint_v1", "M72_V1"])
def test_resolve_mode_unknown(value):
    with pytest.raises(ValueError):
        t4.resolve_mode(value)


@pytest.mark.parametrize("mode", t4.MODES)
def test_require_environment_accepts_the_full_set(mode, monkeypatch):
    monkeypatch.delenv("TQ_DECODE_2PASS_LEGACY", raising=False)
    t4.require_environment(mode, **FULL)
    t4.require_environment(mode, **dict(FULL, kv_bits=4.0), kv_quant_mode="mse")


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"kv_bits": 0}, "kv_bits 4"),
        ({"kv_bits": None}, "kv_bits 4"),
        ({"kv_bits": 8}, "kv_bits 4"),
        ({"kv_bits": 3.5}, "kv_bits 4"),
        ({"kv_quant_scheme": "uniform"}, "turboquant"),
        ({"attention_policy": "auto"}, "fused_v1"),
        ({"mtp_verify_scan": "per_query"}, "joint_v1"),
    ],
)
@pytest.mark.parametrize("mode", t4.MODES)
def test_require_environment_refuses(mode, over, needle, monkeypatch):
    monkeypatch.delenv("TQ_DECODE_2PASS_LEGACY", raising=False)
    with pytest.raises(t4.Tq4AttentionError, match=needle):
        t4.require_environment(mode, **dict(FULL, **over))


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
def test_require_environment_refuses_the_legacy_decode_flag(value, monkeypatch):
    monkeypatch.setenv("TQ_DECODE_2PASS_LEGACY", value)
    with pytest.raises(t4.Tq4AttentionError, match="TQ_DECODE_2PASS_LEGACY"):
        t4.require_environment("m72_v1", **FULL)


@pytest.mark.parametrize("value", ["0", "false", "no", "", "off", "2"])
def test_require_environment_accepts_a_legacy_flag_production_reads_as_off(
    value, monkeypatch
):
    monkeypatch.setenv("TQ_DECODE_2PASS_LEGACY", value)
    from mlx_vlm import turboquant as tq

    assert tq.TurboQuantKVCache(bits=4)._decode_2pass_use_legacy is False
    t4.require_environment("m72_v1", **FULL)


@pytest.mark.parametrize("value", ["1", "true", "yes", "0", "no", ""])
def test_legacy_parse_matches_the_cache_constructor(value, monkeypatch):
    from mlx_vlm import turboquant as tq

    monkeypatch.setenv("TQ_DECODE_2PASS_LEGACY", value)
    assert (
        t4.legacy_decode_enabled()
        is tq.TurboQuantKVCache(bits=4)._decode_2pass_use_legacy
    )


def test_require_environment_refuses_the_prod_key_codec(monkeypatch):
    monkeypatch.delenv("TQ_DECODE_2PASS_LEGACY", raising=False)
    with pytest.raises(t4.Tq4AttentionError, match="MSE"):
        t4.require_environment("m72_v1", **FULL, kv_quant_mode="prod")
    with pytest.raises(t4.Tq4AttentionError, match="MSE"):
        t4.require_environment("m72_v1", **FULL, kv_quant_mode="PROD")


@pytest.mark.parametrize("value,refused", [("prod", True), ("mse", False), ("", False)])
def test_require_environment_reads_the_codec_the_cache_is_built_with(
    value, refused, monkeypatch
):
    # kv_quant_mode omitted: the effective value is the cache constructor's own resolution of TQ_KV_QUANT_MODE
    from mlx_vlm import turboquant as tq

    monkeypatch.delenv("TQ_DECODE_2PASS_LEGACY", raising=False)
    monkeypatch.setenv("TQ_KV_QUANT_MODE", value)
    assert t4.effective_kv_quant_mode() == tq.TurboQuantKVCache(bits=4)._kv_quant_mode
    if refused:
        with pytest.raises(t4.Tq4AttentionError, match="MSE"):
            t4.require_environment("m72_v1", **FULL)
    else:
        t4.require_environment("m72_v1", **FULL)


@pytest.fixture
def restore_environ():
    import os

    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_cli_kv_quant_mode_prod_is_refused_at_load(monkeypatch, restore_environ):
    # the CLI path: --kv-quant-mode writes TQ_KV_QUANT_MODE; the load-time check must see it
    from mlx_vlm.server import cli as cli_module
    from mlx_vlm.server import generation as g

    monkeypatch.setattr(cli_module.uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_apply_mlx_memory_limits", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_configure_session_manager", lambda **k: None)
    for k in ("TQ_DECODE_2PASS_LEGACY", "TQ_KV_QUANT_MODE", "MLX_VLM_TQ4_ATTENTION"):
        monkeypatch.delenv(k, raising=False)
    argv = ["mlx_vlm.server", "--tq4-attention", "m72_v1"]
    argv += ["--attention-policy", "fused_v1", "--mtp-verify-scan", "joint_v1"]
    monkeypatch.setattr(
        sys,
        "argv",
        [*argv, "--kv-quant-mode", "prod"],
    )
    cli_module.main()
    with pytest.raises(t4.Tq4AttentionError, match="MSE"):
        g._resolve_tq4_mode_from_env(kv_bits=4, kv_quant_scheme="turboquant")
    monkeypatch.setattr(
        sys,
        "argv",
        [*argv, "--kv-quant-mode", "mse"],
    )
    cli_module.main()
    assert (
        g._resolve_tq4_mode_from_env(kv_bits=4, kv_quant_scheme="turboquant")
        == "m72_v1"
    )


def test_require_environment_absent_mode_is_a_noop():
    t4.require_environment(
        None, kv_bits=0, kv_quant_scheme="", attention_policy="auto", mtp_verify_scan=""
    )


# ================================================================== mtp_verify_scan: kv_bits 4 only with the mode
def test_mtp_verify_still_refuses_quantized_kv_without_the_mode(monkeypatch):
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    for kv_bits, scheme in ((4, "turboquant"), (8, "uniform"), (4, "uniform")):
        with pytest.raises(mv.VerifyScanError, match="quantized KV"):
            mv.require_environment(
                draft_kind="mtp", kv_bits=kv_bits, kv_quant_scheme=scheme
            )
    mv.require_environment(draft_kind="mtp", kv_bits=0, kv_quant_scheme="turboquant")
    mv.require_environment(draft_kind="mtp", kv_bits=None)


@pytest.mark.parametrize("mode", t4.MODES)
def test_mtp_verify_accepts_tq4_kv_bits_4_with_the_mode(mode, monkeypatch):
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    mv.require_environment(
        draft_kind="mtp", kv_bits=4, kv_quant_scheme="turboquant", tq4_mode=mode
    )
    mv.require_environment(
        draft_kind="mtp", kv_bits=4.0, kv_quant_scheme="turboquant", tq4_mode=mode
    )


@pytest.mark.parametrize(
    "kv_bits,scheme", [(8, "turboquant"), (3, "turboquant"), (4, "uniform")]
)
def test_mtp_verify_with_the_mode_still_refuses_other_quantized_kv(
    kv_bits, scheme, monkeypatch
):
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    with pytest.raises(mv.VerifyScanError, match="quantized KV"):
        mv.require_environment(
            draft_kind="mtp", kv_bits=kv_bits, kv_quant_scheme=scheme, tq4_mode="m72_v1"
        )


def test_mtp_verify_with_the_mode_still_requires_mtp():
    with pytest.raises(mv.VerifyScanError, match="draft-kind"):
        mv.require_environment(
            draft_kind=None, kv_bits=4, kv_quant_scheme="turboquant", tq4_mode="m72_v1"
        )


# ================================================================== CLI -> environment
def test_cli_flag_reaches_the_worker_environment(monkeypatch):
    import os

    from mlx_vlm.server import cli as cli_module

    monkeypatch.setattr(cli_module.uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_apply_mlx_memory_limits", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_configure_session_manager", lambda **k: None)
    for mode in t4.MODES:
        monkeypatch.setattr(sys, "argv", ["mlx_vlm.server", "--tq4-attention", mode])
        cli_module.main()
        assert os.environ["MLX_VLM_TQ4_ATTENTION"] == mode
    monkeypatch.setattr(sys, "argv", ["mlx_vlm.server"])
    cli_module.main()
    assert (
        os.environ["MLX_VLM_TQ4_ATTENTION"] == "off"
    )  # always written: no stale value
    monkeypatch.setattr(sys, "argv", ["mlx_vlm.server", "--tq4-attention", "m72"])
    with pytest.raises(SystemExit):
        cli_module.main()


# ================================================================== server load order
@pytest.fixture
def generation():
    from mlx_vlm.server import generation as g

    return g


def _fake_self(**over):
    base = dict(
        model_path="x",
        adapter_path=None,
        draft_kind_override="mtp",
        draft_model_path=None,
        apc_manager=None,
        kv_bits=4,
        kv_quant_scheme="turboquant",
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Stop(Exception):
    pass


def _record_load_order(generation, monkeypatch, mode_env):
    order = []
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))
    monkeypatch.setattr(
        generation,
        "load_model_resources",
        lambda *a, **k: (model, SimpleNamespace(), model.config),
    )
    monkeypatch.delenv("MLX_VLM_MOE_EXPAND", raising=False)
    if mode_env is None:
        monkeypatch.delenv("MLX_VLM_TQ4_ATTENTION", raising=False)
    else:
        monkeypatch.setenv("MLX_VLM_TQ4_ATTENTION", mode_env)
    real_resolve = generation._resolve_tq4_mode_from_env

    def resolve(**kw):
        order.append(("resolve", kw))
        return real_resolve(**kw)

    monkeypatch.setattr(generation, "_resolve_tq4_mode_from_env", resolve)
    monkeypatch.setattr(
        generation,
        "_apply_attention_policy_from_env",
        lambda m: order.append(("policy",)),
    )

    def verify(m, **kw):
        order.append(("verify", kw))

    monkeypatch.setattr(generation, "_apply_mtp_verify_from_env", verify)

    def tq4(m, mode):
        order.append(("tq4", mode))
        raise _Stop()

    monkeypatch.setattr(generation, "_apply_tq4_attention_from_env", tq4)
    with pytest.raises(_Stop):
        generation.ResponseGenerator._initialize_model(_fake_self())
    return order


def test_load_order_with_the_mode(generation, monkeypatch):
    monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
    monkeypatch.setenv("MLX_VLM_MTP_VERIFY_SCAN", "joint_v1")
    monkeypatch.delenv("TQ_DECODE_2PASS_LEGACY", raising=False)
    monkeypatch.delenv("TQ_KV_QUANT_MODE", raising=False)
    order = _record_load_order(generation, monkeypatch, "m72_v1")
    assert [o[0] for o in order] == ["resolve", "policy", "verify", "tq4"]
    assert order[2][1]["tq4_mode"] == "m72_v1" and order[2][1]["kv_bits"] == 4
    assert order[3][1] == "m72_v1"


def test_load_order_without_the_mode_passes_nothing_new(generation, monkeypatch):
    order = _record_load_order(generation, monkeypatch, None)
    assert [o[0] for o in order] == ["resolve", "policy", "verify", "tq4"]
    assert "tq4_mode" not in order[2][1]
    assert order[3][1] is None


@pytest.mark.parametrize(
    "env,needle",
    [
        (
            {"MLX_VLM_ATTENTION_POLICY": "auto", "MLX_VLM_MTP_VERIFY_SCAN": "joint_v1"},
            "fused_v1",
        ),
        (
            {
                "MLX_VLM_ATTENTION_POLICY": "fused_v1",
                "MLX_VLM_MTP_VERIFY_SCAN": "per_query",
            },
            "joint_v1",
        ),
        (
            {
                "MLX_VLM_ATTENTION_POLICY": "fused_v1",
                "MLX_VLM_MTP_VERIFY_SCAN": "joint_v1",
                "TQ_DECODE_2PASS_LEGACY": "1",
            },
            "LEGACY",
        ),
        (
            {
                "MLX_VLM_ATTENTION_POLICY": "fused_v1",
                "MLX_VLM_MTP_VERIFY_SCAN": "joint_v1",
                "TQ_KV_QUANT_MODE": "prod",
            },
            "MSE",
        ),
    ],
)
def test_resolve_refuses_at_load(generation, monkeypatch, env, needle):
    for k in ("TQ_DECODE_2PASS_LEGACY", "TQ_KV_QUANT_MODE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MLX_VLM_TQ4_ATTENTION", "m72_v1")
    with pytest.raises(t4.Tq4AttentionError, match=needle):
        generation._resolve_tq4_mode_from_env(kv_bits=4, kv_quant_scheme="turboquant")


def test_resolve_refuses_an_unknown_value(generation, monkeypatch):
    monkeypatch.setenv("MLX_VLM_TQ4_ATTENTION", "m72_v9")
    with pytest.raises(t4.Tq4AttentionError, match="unknown"):
        generation._resolve_tq4_mode_from_env(kv_bits=4, kv_quant_scheme="turboquant")


def test_resolve_off_imports_nothing(generation, monkeypatch):
    for name in ("mlx_vlm.tq4_attention", "mlx_vlm.tq4_fast"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setenv("MLX_VLM_TQ4_ATTENTION", "off")
    assert (
        generation._resolve_tq4_mode_from_env(kv_bits=4, kv_quant_scheme="turboquant")
        is None
    )
    monkeypatch.delenv("MLX_VLM_TQ4_ATTENTION")
    assert (
        generation._resolve_tq4_mode_from_env(kv_bits=4, kv_quant_scheme="turboquant")
        is None
    )
    assert (
        "mlx_vlm.tq4_attention" not in sys.modules
        and "mlx_vlm.tq4_fast" not in sys.modules
    )


def test_apply_step_order_inside_the_tq4_loader(generation, monkeypatch):
    calls = []
    hook = SimpleNamespace(name="m72_v1")
    model = SimpleNamespace(tq4_attention=None)
    monkeypatch.setattr(t4, "require_policy_stamps", lambda m: calls.append("stamps"))
    monkeypatch.setattr(
        t4,
        "check_fp16_range",
        lambda m, caches=None: calls.append("range")
        or {"max_q_bound": 16.0, "max_codebook": 0.2},
    )

    def apply(m, mode):
        calls.append(("apply", mode))
        m.tq4_attention = hook
        return 2

    monkeypatch.setattr(t4, "apply_to_model", apply)
    monkeypatch.setattr(
        t4,
        "self_test_model",
        lambda m, mode: calls.append(("self_test", mode))
        or {"cells": 7, "elapsed_s": 0.1},
    )
    assert generation._apply_tq4_attention_from_env(model, "m72_v1") is hook
    assert calls == ["stamps", "range", ("apply", "m72_v1"), ("self_test", hook)]
    assert generation._apply_tq4_attention_from_env(model, None) is None


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="the self-test runs the kernels"
)
def test_apply_on_a_real_model_stamps_and_self_tests(generation):
    from test_tq4_impl_support import lm24

    from mlx_vlm import attention_policy as ap

    lm = lm24()
    model = SimpleNamespace(config=lm.config, language_model=lm)
    with pytest.raises(t4.Tq4AttentionError, match="fused_v1"):
        generation._apply_tq4_attention_from_env(
            model, "m72_exact"
        )  # stamps missing: refused
    ap.apply_to_model(model, ap.FusedV1Policy())
    mv.apply_to_model(model, mv.JointV1Policy())
    hook = generation._apply_tq4_attention_from_env(model, "m72_exact")
    assert hook.name == "m72_exact" and hook.exact
    attns = [lm.model.layers[i].self_attn for i in (1, 3)]
    assert all(a.tq4_attention is hook for a in attns)
    assert all(
        a.attention_policy.name == "fused_v1" for a in attns
    )  # the stamps stay on every module


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="the self-test runs the kernels"
)
def test_self_test_refuses_a_model_outside_the_kernel_shapes():
    from test_attention_policy import _tiny_lm

    lm = _tiny_lm()
    model = SimpleNamespace(config=lm.config, language_model=lm)
    t4.apply_to_model(model, "m72_v1")
    with pytest.raises(t4.Tq4AttentionError, match="shape"):
        t4.self_test_model(model, "m72_v1")


def test_apply_refuses_an_unqualified_family():
    import mlx.nn as nn

    model = SimpleNamespace(
        language_model=nn.Sequential(nn.Linear(2, 2)),
        config=SimpleNamespace(model_type="llama"),
    )
    with pytest.raises(t4.Tq4AttentionError, match="qualified"):
        t4.apply_to_model(model, "m72_v1")


# ================================================================== /health
@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import mlx_vlm.server as server

    with TestClient(server.app) as c:
        yield c, server


def test_health_reports_the_mode_and_route_counters(client, monkeypatch):
    c, server = client
    from mlx_vlm import tq4_fast

    tq4_fast.count("p2s")
    hook = t4.Tq4Mode("m72_exact")
    monkeypatch.setattr(
        server.runtime, "response_generator", SimpleNamespace(tq4_attention=hook)
    )
    body = c.get("/health").json()
    assert body["tq4_attention"]["mode"] == "m72_exact"
    assert body["tq4_attention"]["route_counts"] == tq4_fast.route_counts()
    assert body["tq4_attention"]["route_counts"]["p2s"] >= 1


def test_health_without_the_mode_is_unchanged(client, monkeypatch):
    c, server = client
    monkeypatch.setattr(
        server.runtime, "response_generator", SimpleNamespace(tq4_attention=None)
    )
    body = c.get("/health").json()
    assert "tq4_attention" not in body
    assert set(body) == {
        "status",
        "loaded_model",
        "loaded_adapter",
        "loaded_models",
        "loaded_context_size",
        "configured_context_limit",
        "effective_context_limit",
        "loaded_tool_parser",
        "continuous_batching_enabled",
        "apc_enabled",
    }
