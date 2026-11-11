"""M72 1.3: the mode's module interface, its environment refusals, the R1 range check (A7), the CLI flag / env, the server load
order and the load-time refusals (spec "Interfaces", "Server load order", A3 refusals unchanged, A7, A8 worker env).

``mlx_vlm.tq4_attention`` mirrors ``mlx_vlm.mtp_verify_scan``: ``MODES``, ``resolve_mode``, ``require_environment``,
``check_fp16_range``, ``apply_to_model``, ``self_test_model``. Every test is classed in its comment: [new] exercises new
behaviour, [invariant] preserves an invariant of today's code, [fix] reproduces a failure the specification fixes.
"""

from __future__ import annotations

import os
import re
import sys
from types import SimpleNamespace

import mlx.nn as nn
import pytest
from test_tq4_m72_support import (
    ENV_MODE,
    MODES,
    mode_module,
    mx,
    needs_metal,
    tiny_model,
    tq4_cache,
)

from mlx_vlm import attention_policy as ap
from mlx_vlm import mtp_verify_scan as mv
from mlx_vlm.models.qwen3_5.language import Qwen3_5Attention
from mlx_vlm.server import cli as cli_module
from mlx_vlm.server import generation as generation_module

VALID = dict(
    kv_bits=4,
    kv_quant_scheme="turboquant",
    attention_policy="fused_v1",
    mtp_verify_scan="joint_v1",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        ENV_MODE,
        "TQ_DECODE_2PASS_LEGACY",
        "MLX_VLM_ATTENTION_POLICY",
        "MLX_VLM_MTP_VERIFY_SCAN",
        "MLX_VLM_MTP_VERIFY_AB",
        "MLX_VLM_DRAFT_KIND",
        "MLX_VLM_DRAFT_MODEL",
        "MLX_VLM_MOE_EXPAND",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def ta():
    return mode_module()


# =============================================================================== resolve_mode / MODES
# [new] the registry's two values, exactly
def test_modes_are_exactly_the_two_registry_values(ta):
    assert tuple(ta.MODES) == MODES


# [new] absent resolves to None (today's behaviour: nothing stamped); a value resolves to itself
def test_resolve_mode_absent_and_values(ta):
    assert ta.resolve_mode(None) is None
    for mode in MODES:
        assert ta.resolve_mode(mode) == mode


# [new] anything else is refused, never mapped to a mode
@pytest.mark.parametrize(
    "value", ["m72_v2", "M72_V1", "on", "true", "m72", "m72_exact ", "1"]
)
def test_resolve_mode_refuses_unknown_values(ta, value):
    with pytest.raises(Exception):
        ta.resolve_mode(value)


# =============================================================================== require_environment
def call_env(ta, mode="m72_v1", **over):
    return ta.require_environment(mode, **{**VALID, **over})


# [new] the valid combination passes for both modes
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kv_bits", (4, 4.0))
def test_require_environment_accepts_the_valid_combination(ta, mode, kv_bits):
    assert call_env(ta, mode, kv_bits=kv_bits) is None


# [new] each knob of the four-way rule, alone wrong, refuses (A8: the worker-side mirror of the registry validation)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "over",
    [
        dict(kv_bits=0),
        dict(kv_bits=None),
        dict(kv_bits=8),
        dict(kv_bits=3),
        dict(kv_quant_scheme="uniform"),
        dict(kv_quant_scheme=None),
        dict(attention_policy="auto"),
        dict(attention_policy=None),
        dict(mtp_verify_scan="per_query"),
        dict(mtp_verify_scan=None),
    ],
    ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()),
)
def test_require_environment_refuses_each_wrong_knob(ta, mode, over):
    with pytest.raises(Exception):
        call_env(ta, mode, **over)


# [new] the legacy decode flag voids D2's bit-identity contract: refused (any spelling the cache itself treats as true)
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
def test_require_environment_refuses_when_the_legacy_flag_is_set(
    ta, monkeypatch, value
):
    monkeypatch.setenv("TQ_DECODE_2PASS_LEGACY", value)
    with pytest.raises(Exception):
        call_env(ta)


@pytest.mark.parametrize("value", ["0", "false", "no", ""])
def test_require_environment_accepts_the_legacy_flag_spelled_off(
    ta, monkeypatch, value
):
    monkeypatch.setenv("TQ_DECODE_2PASS_LEGACY", value)
    assert call_env(ta) is None


# [new] an unknown mode is a refusal, not a pass
def test_require_environment_refuses_an_unknown_mode(ta):
    with pytest.raises(Exception):
        call_env(ta, "m72_v3")


# =============================================================================== A3: today's refusals unchanged
# [invariant] with the mode absent, the MTP verify scan keeps refusing a quantized KV, TurboQuant included
@pytest.mark.parametrize("kv_bits", (4, 4.0, 8, 2.5))
def test_a3_mtp_verify_scan_still_refuses_quantized_kv_without_the_mode(kv_bits):
    with pytest.raises(mv.VerifyScanError, match="quantized KV"):
        mv.require_environment(
            draft_kind="mtp", kv_bits=kv_bits, kv_quant_scheme="turboquant"
        )


# [invariant] and keeps accepting the shipped native16 combination, and still demands the MTP draft kind
def test_a3_mtp_verify_scan_native_and_draft_kind_rules_unchanged():
    mv.require_environment(draft_kind="mtp", kv_bits=0, kv_quant_scheme="turboquant")
    mv.require_environment(draft_kind="mtp", kv_bits=None)
    with pytest.raises(mv.VerifyScanError, match="draft-kind mtp"):
        mv.require_environment(draft_kind="suffix", kv_bits=0)


# [invariant] JointV1Policy sends TurboQuant caches to per_query (the mode's hook runs first for them), len-1 blocks to len1
def test_joint_v1_classify_keeps_sending_turboquant_caches_to_per_query():
    cache = tq4_cache(300, cap=512, seed=2)
    policy = mv.JointV1Policy()
    q = mx.zeros((1, 24, 3, 256), dtype=mx.bfloat16)
    k = mx.zeros((1, 4, 300, 256), dtype=mx.bfloat16)
    decision = policy.classify(q, k, k, cache, "causal")
    assert (decision.route, decision.reason) == ("per_query", "cache")
    assert policy.classify(q[:, :, :1], k, k, cache, "causal").route == "len1"


# =============================================================================== check_fp16_range (R1, A7)
@pytest.fixture
def lm():
    return tiny_model(layers=4)


def full_attention_index(model):
    return [i for i, layer in enumerate(model.model.layers) if not layer.is_linear]


def set_q_norm_weight(model, layer, value):
    w = model.model.layers[layer].self_attn.q_norm.weight
    model.model.layers[layer].self_attn.q_norm.weight = mx.full(
        w.shape, value, dtype=w.dtype
    )
    mx.eval(model.model.layers[layer].self_attn.q_norm.weight)


# [new] A7: passes on a model inside both bounds and returns a dict of the recorded values
def test_a7_check_fp16_range_passes_and_records_values(ta, lm):
    out = ta.check_fp16_range(lm)
    assert isinstance(out, dict) and out


# [new] A7: the query bound is sqrt(D) * max|gamma| <= 2**14, i.e. max|gamma| <= 1024 at D = 256 (boundary inclusive)
@pytest.mark.parametrize(
    "gamma,ok", [(1000.0, True), (1024.0, True), (2048.0, False), (4096.0, False)]
)
def test_a7_q_norm_bound(ta, lm, gamma, ok):
    # amended (delivery 3): the weight is stored in the model's dtype (bf16), so the planted values are exactly
    # representable (1000 -> 1000, 1024, 2048, 4096) and the violations sit clearly above sqrt(256) * gamma = 2**14
    set_q_norm_weight(lm, full_attention_index(lm)[0], gamma)
    w = lm.model.layers[full_attention_index(lm)[0]].self_attn.q_norm.weight
    assert (
        float(mx.max(w.astype(mx.float32)).item()) == gamma
    )  # the planted value survived storage
    if ok:
        ta.check_fp16_range(lm)
    else:
        with pytest.raises(Exception):
            ta.check_fp16_range(lm)


# [new] A7: the violating layer is named in the start-up error (the last full-attention layer, index 3, planted)
def test_a7_the_error_names_the_layer(ta, lm):
    layer = full_attention_index(lm)[-1]
    set_q_norm_weight(lm, layer, 5000.0)
    with pytest.raises(Exception) as err:
        ta.check_fp16_range(lm)
    assert re.search(rf"(?<!\d){layer}(?!\d)", str(err.value)), str(err.value)


# [new] A7: the codebook bound max|codebook| <= 1, planted on a cache's codec
@needs_metal
@pytest.mark.parametrize("planted,ok", [(None, True), (1.0, True), (1.5, False)])
@pytest.mark.parametrize("which", ("key_codec", "value_codec"))
def test_a7_codebook_bound(ta, lm, which, planted, ok):
    # amended (delivery 3): today's values (max |c| = 0.17) pass; a violation is an ABSOLUTE entry (1.0 is the inclusive
    # limit, 1.5 is clearly above it), not a scale of today's codebook
    cache = lm.make_cache()
    cache = [c for c in cache]
    tq_cache = tq4_cache(300, cap=512, seed=4)
    codec = getattr(tq_cache, which)
    if planted is not None:
        cb = codec.codebook
        codec.codebook = mx.where(
            mx.arange(cb.shape[0]) == 0, mx.array(planted, dtype=cb.dtype), cb
        )
        mx.eval(codec.codebook)
        assert float(mx.max(mx.abs(codec.codebook)).item()) == planted
    caches = [tq_cache if i == 1 else c for i, c in enumerate(cache)]
    if ok:
        ta.check_fp16_range(lm, caches)
    else:
        with pytest.raises(Exception):
            ta.check_fp16_range(lm, caches)


# =============================================================================== apply_to_model / self_test_model
# [new] stamps the qwen3_5 attention modules, returns the count; refuses an unsupported family (like mtp_verify_scan)
def test_apply_to_model_counts_the_attention_modules(ta, lm):
    modules = [m for _, m in lm.named_modules() if type(m) is Qwen3_5Attention]
    assert len(modules) == 2
    assert ta.apply_to_model(lm, "m72_v1") == len(modules)


def test_apply_to_model_refuses_an_unsupported_family(ta):
    other = nn.Module()
    other.config = SimpleNamespace(model_type="llama")
    with pytest.raises(Exception):
        ta.apply_to_model(other, "m72_v1")


def test_apply_to_model_refuses_an_unknown_mode(ta, lm):
    with pytest.raises(Exception):
        ta.apply_to_model(lm, "m72_v9")


# [new] the model self-test runs the kernel self-test on the model's attention shape and returns a dict
@needs_metal
@pytest.mark.parametrize("mode", MODES)
def test_self_test_model_returns_a_dict(ta, mode):
    model = tiny_model(layers=4)
    ta.apply_to_model(model, mode)
    res = ta.self_test_model(model, mode)
    assert isinstance(res, dict) and res


# =============================================================================== CLI flag / env (A8)
@pytest.fixture
def cli_run(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        cli_module.uvicorn, "run", lambda *a, **k: seen.setdefault("ran", True)
    )
    monkeypatch.setattr(cli_module, "_apply_mlx_memory_limits", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_configure_session_manager", lambda **k: None)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["mlx_vlm.server", *argv])
        seen.clear()
        cli_module.main()
        assert seen.get("ran")

    return run


# [new] --tq4-attention <mode> reaches the worker environment as MLX_VLM_TQ4_ATTENTION; the next start does not inherit it
@pytest.mark.parametrize("mode", MODES)
def test_cli_flag_reaches_the_environment_and_never_goes_stale(cli_run, mode):
    cli_run("--tq4-attention", mode)
    assert os.environ[ENV_MODE] == mode
    cli_run()
    assert os.environ.get(ENV_MODE) not in MODES


# [new] values outside the registry set are refused by the parser
def test_cli_flag_rejects_unknown_values(cli_run):
    cli_run(
        "--tq4-attention", "m72_v1"
    )  # the flag exists (else the refusal below would be "unrecognized argument")
    with pytest.raises(SystemExit):
        cli_run("--tq4-attention", "m72_v2")


# =============================================================================== server load order and load-time refusals
STAGES = (
    "ta.resolve_mode",
    "ta.require_environment",
    "ap.apply_to_model",
    "mv.require_environment",
    "mv.apply_to_model",
    "ta.check_fp16_range",
    "ta.apply_to_model",
    "ta.self_test_model",
)


class Boom(Exception):
    pass


@pytest.fixture
def loader(monkeypatch, ta):
    """``ResponseGenerator._initialize_model`` on the synthetic model, with every stage recording its call; stages named in
    ``real`` run for real, ``fail`` names raise Boom."""
    lm = tiny_model(layers=4)

    def run(mode="m72_v1", *, real=(), fail=None, env=None, kv_bits=4, no_stamp=()):
        events = []
        # amended (delivery 3): the policy stages really stamp fused_v1 / joint_v1 (the spec: the mode requires both stamps at
        # load); `no_stamp` names a stage that is left as a recording no-op, i.e. a stamp that is missing
        real = tuple(
            set(real) | ({"ap.apply_to_model", "mv.apply_to_model"} - set(no_stamp))
        )
        for name, value in {
            "MLX_VLM_ATTENTION_POLICY": "fused_v1",
            "MLX_VLM_MTP_VERIFY_SCAN": "joint_v1",
            "MLX_VLM_DRAFT_KIND": "mtp",
            **({ENV_MODE: mode} if mode else {}),
            **(env or {}),
        }.items():
            monkeypatch.setenv(name, value)
        model = SimpleNamespace(language_model=lm, config=lm.config)
        monkeypatch.setattr(
            generation_module,
            "load_model_resources",
            lambda *a, **k: (model, SimpleNamespace(), lm.config),
        )

        def stage(label, owner, name, ret):
            real_fn = getattr(owner, name)

            def f(*a, **k):
                events.append(label)
                if fail == label:
                    raise Boom(label)
                if label in real:
                    return real_fn(*a, **k)
                return ret(a, k) if callable(ret) else ret

            monkeypatch.setattr(owner, name, f)

        stage(
            "ta.resolve_mode",
            ta,
            "resolve_mode",
            lambda a, k: a[0] if a else k.get("value"),
        )
        stage("ta.require_environment", ta, "require_environment", None)
        stage("ap.apply_to_model", ap, "apply_to_model", 2)
        stage("ap.self_test_model", ap, "self_test_model", [])
        stage("mv.require_environment", mv, "require_environment", None)
        stage("mv.apply_to_model", mv, "apply_to_model", 2)
        stage("mv.self_test_model", mv, "self_test_model", [])
        stage("ta.check_fp16_range", ta, "check_fp16_range", {})
        stage("ta.apply_to_model", ta, "apply_to_model", 2)
        stage("ta.self_test_model", ta, "self_test_model", {})
        fake = SimpleNamespace(
            model_path="x",
            adapter_path=None,
            draft_kind_override=None,
            draft_model_path=None,
            apc_manager=None,
            kv_bits=kv_bits,
            kv_quant_scheme="turboquant",
        )
        exc = None
        try:
            generation_module.ResponseGenerator._initialize_model(fake)
        except (
            Exception
        ) as e:  # noqa: BLE001 - later stages need a real processor / drafter (not under test)
            exc = e
        return SimpleNamespace(fake=fake, events=events, exc=exc, order=order(events))

    return run


def order(events):
    return [e for e in events if e in STAGES]


# [new] the spec's order: resolve -> require_environment -> attention policy -> MTP verify -> range check -> apply -> self-test
@pytest.mark.parametrize("mode", MODES)
def test_load_order(loader, mode):
    res = loader(mode)
    seen = res.order
    assert seen == [
        "ta.resolve_mode",
        "ta.require_environment",
        "ap.apply_to_model",
        "mv.require_environment",
        "mv.apply_to_model",
        "ta.check_fp16_range",
        "ta.apply_to_model",
        "ta.self_test_model",
    ], seen


# [new] a failure at any mode stage refuses to start: the error propagates (no silent fallback) and nothing later runs
@pytest.mark.parametrize(
    "stage_name",
    [
        "ta.require_environment",
        "ta.check_fp16_range",
        "ta.apply_to_model",
        "ta.self_test_model",
    ],
)
def test_a_failing_mode_stage_aborts_the_load(loader, stage_name):
    res = loader("m72_v1", fail=stage_name)
    assert isinstance(res.exc, Boom) and res.exc.args == (stage_name,), res.exc
    later = STAGES[STAGES.index(stage_name) + 1 :]
    assert not (set(later) & set(res.events)), res.events


# [new] a refusal in require_environment comes before the attention-policy and MTP-verify stages ran
def test_require_environment_failure_precedes_every_other_stage(loader):
    res = loader("m72_v1", fail="ta.require_environment")
    assert res.order == ["ta.resolve_mode", "ta.require_environment"], res.order


# [new] mode absent: no mode stage runs (nothing imported, stamped or checked)
def test_mode_absent_runs_no_mode_stage(loader):
    res = loader(None)
    assert not [e for e in res.events if e.startswith("ta.")], res.events


# [new] the MTP verify stage accepts kv_bits = 4 with a resolved mode (the real require_environment runs), for both modes
@pytest.mark.parametrize("mode", MODES)
def test_mtp_verify_accepts_quantized_kv_when_the_mode_resolved(loader, mode):
    res = loader(mode, real=("mv.require_environment", "ta.require_environment"))
    assert res.fake.mtp_verify_policy is not None, res.exc
    assert "ta.check_fp16_range" in res.order


# [invariant] [fix] and refuses it with the mode absent, exactly as today (A3)
def test_mtp_verify_refuses_quantized_kv_without_the_mode(loader):
    res = loader(None, real=("mv.require_environment",))
    assert isinstance(res.exc, mv.VerifyScanError) and "quantized KV" in str(res.exc)
    assert not [e for e in res.events if e.startswith("ta.")]


# [new] the real require_environment refuses a combination the mlx-serve rule would have refused; no later mode stage runs
@pytest.mark.parametrize(
    "env",
    [
        {"MLX_VLM_ATTENTION_POLICY": "auto"},
        {"MLX_VLM_MTP_VERIFY_SCAN": "per_query"},
        {"TQ_DECODE_2PASS_LEGACY": "1"},
    ],
    ids=lambda e: next(iter(e)),
)
def test_load_refuses_an_incompatible_companion(loader, env):
    res = loader("m72_v1", real=("ta.require_environment",), env=env)
    assert res.exc is not None
    assert (
        "ta.check_fp16_range" not in res.events
        and "ta.apply_to_model" not in res.events
    ), res.events


@pytest.mark.parametrize("kv_bits", (0, 8, None))
def test_load_refuses_the_mode_on_a_native_or_other_kv(loader, kv_bits):
    res = loader("m72_v1", real=("ta.require_environment",), kv_bits=kv_bits)
    assert res.exc is not None
    assert (
        "ta.check_fp16_range" not in res.events
        and "ta.apply_to_model" not in res.events
    ), res.events


# [new] (delivery 3) the mode requires both stamps at load: with `fused_v1` or `joint_v1` not stamped on the native layer the
# load is refused before the range check / stamp / self-test (spec ledger row 15)
@pytest.mark.parametrize(
    "missing",
    ["ap.apply_to_model", "mv.apply_to_model"],
    ids=["no-fused_v1-stamp", "no-joint_v1-stamp"],
)
def test_load_refuses_when_a_policy_stamp_is_missing(loader, missing):
    res = loader("m72_v1", real=("ta.require_environment",), no_stamp=(missing,))
    assert res.exc is not None
    assert not (
        {"ta.check_fp16_range", "ta.apply_to_model", "ta.self_test_model"}
        & set(res.events)
    ), res.events
