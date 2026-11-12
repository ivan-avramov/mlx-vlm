"""M77: ``tq4_attention`` served with MTP OFF (no drafter configured).

Criterion 2: ``require_environment`` / ``require_policy_stamps`` per configured drafter state (omitted = today, ``"mtp"``,
None / ``""``, other kinds). Criterion 3: ``_initialize_model`` on the synthetic Qwen3.5-family model with the mode and no
drafter (mode applied, no verify stamp, decode through ``d2`` / ``single_pass``, never ``fallback:*``), refusals with a
drafter configured and no scan, and with no drafter configured but one loaded. Criterion 4: greedy ``m72_exact`` MTP OFF
emits MTP ON's tokens (a repeat control of each arm first).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm import attention_policy as ap  # noqa: E402
from mlx_vlm import mtp_verify_scan as mv  # noqa: E402
from mlx_vlm import tq4_attention as t4  # noqa: E402

MODES = t4.MODES
BASE = dict(kv_bits=4, kv_quant_scheme="turboquant", attention_policy="fused_v1")
OFF_KINDS = (None, "")
OTHER_KINDS = ("suffix", "dflash", "eagle3")


@pytest.fixture(autouse=True)
def _clean_codec_env(monkeypatch):
    monkeypatch.delenv("TQ_DECODE_2PASS_LEGACY", raising=False)
    monkeypatch.delenv("TQ_KV_QUANT_MODE", raising=False)


# ================================================================== criterion 2: require_environment
@pytest.mark.parametrize("mode", MODES)
def test_env_omitted_draft_kind_is_today(mode):
    t4.require_environment(mode, **BASE, mtp_verify_scan="joint_v1")
    for scan in ("per_query", "", None):
        with pytest.raises(
            t4.Tq4AttentionError, match="requires mtp_verify_scan joint_v1"
        ):
            t4.require_environment(mode, **BASE, mtp_verify_scan=scan)


@pytest.mark.parametrize("mode", MODES)
def test_env_mtp_requires_joint_v1(mode):
    t4.require_environment(mode, **BASE, mtp_verify_scan="joint_v1", draft_kind="mtp")
    for scan in ("per_query", "", None):
        with pytest.raises(
            t4.Tq4AttentionError, match="requires mtp_verify_scan joint_v1"
        ):
            t4.require_environment(mode, **BASE, mtp_verify_scan=scan, draft_kind="mtp")


@pytest.mark.parametrize("draft_kind", OFF_KINDS)
@pytest.mark.parametrize("scan", ("per_query", "", None))
@pytest.mark.parametrize("mode", MODES)
def test_env_off_accepts_no_joint_scan(mode, scan, draft_kind):
    t4.require_environment(mode, **BASE, mtp_verify_scan=scan, draft_kind=draft_kind)


@pytest.mark.parametrize("draft_kind", OFF_KINDS)
@pytest.mark.parametrize("mode", MODES)
def test_env_off_refuses_joint_v1(mode, draft_kind):
    with pytest.raises(t4.Tq4AttentionError, match="joint_v1 requires a drafter"):
        t4.require_environment(
            mode, **BASE, mtp_verify_scan="joint_v1", draft_kind=draft_kind
        )


@pytest.mark.parametrize("mode", MODES)
def test_env_off_refuses_an_unknown_scan(mode):
    with pytest.raises(t4.Tq4AttentionError, match="per_query"):
        t4.require_environment(mode, **BASE, mtp_verify_scan="bogus", draft_kind=None)


@pytest.mark.parametrize("scan", ("joint_v1", "per_query"))
@pytest.mark.parametrize("draft_kind", OTHER_KINDS)
@pytest.mark.parametrize("mode", MODES)
def test_env_other_drafter_kinds_are_refused(mode, draft_kind, scan):
    with pytest.raises(t4.Tq4AttentionError, match="draft_kind mtp or no drafter"):
        t4.require_environment(
            mode, **BASE, mtp_verify_scan=scan, draft_kind=draft_kind
        )


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"kv_bits": 0}, "kv_bits 4"),
        ({"kv_quant_scheme": "uniform"}, "turboquant"),
        ({"attention_policy": "auto"}, "fused_v1"),
    ],
)
@pytest.mark.parametrize("mode", MODES)
def test_env_off_keeps_the_other_checks(mode, over, needle):
    with pytest.raises(t4.Tq4AttentionError, match=needle):
        t4.require_environment(
            mode, **dict(BASE, **over), mtp_verify_scan="per_query", draft_kind=None
        )


@pytest.mark.parametrize("mode", MODES)
def test_env_off_keeps_the_codec_and_legacy_checks(mode, monkeypatch):
    with pytest.raises(t4.Tq4AttentionError, match="MSE"):
        t4.require_environment(
            mode,
            **BASE,
            mtp_verify_scan="per_query",
            draft_kind=None,
            kv_quant_mode="prod",
        )
    monkeypatch.setenv("TQ_DECODE_2PASS_LEGACY", "1")
    with pytest.raises(t4.Tq4AttentionError, match="LEGACY"):
        t4.require_environment(
            mode, **BASE, mtp_verify_scan="per_query", draft_kind=None
        )


@pytest.mark.parametrize("draft_kind", (None, "", "mtp", "suffix"))
def test_env_absent_mode_is_a_noop(draft_kind):
    t4.require_environment(
        None,
        kv_bits=0,
        kv_quant_scheme=None,
        attention_policy="auto",
        mtp_verify_scan="per_query",
        draft_kind=draft_kind,
    )


# ================================================================== criterion 2: require_policy_stamps
def _stamped(verify):
    from test_tq4_m72_support import tiny_model

    lm = tiny_model(layers=4)
    ap.apply_to_model(lm, ap.FusedV1Policy())
    if verify:
        mv.apply_to_model(lm, mv.JointV1Policy())
    return lm


@pytest.mark.parametrize("draft_kind", ("omitted", "mtp"))
def test_stamps_on_requires_joint_v1(draft_kind):
    kw = {} if draft_kind == "omitted" else {"draft_kind": draft_kind}
    assert t4.require_policy_stamps(_stamped(True), **kw) == 2
    with pytest.raises(t4.Tq4AttentionError, match="joint_v1"):
        t4.require_policy_stamps(_stamped(False), **kw)


@pytest.mark.parametrize("draft_kind", OFF_KINDS)
def test_stamps_off_accepts_no_verify_stamp(draft_kind):
    assert t4.require_policy_stamps(_stamped(False), draft_kind=draft_kind) == 2


@pytest.mark.parametrize("draft_kind", OFF_KINDS)
def test_stamps_off_refuses_a_verify_stamp(draft_kind):
    with pytest.raises(t4.Tq4AttentionError, match="no drafter"):
        t4.require_policy_stamps(_stamped(True), draft_kind=draft_kind)


@pytest.mark.parametrize("draft_kind", ("omitted", "mtp", None, ""))
def test_stamps_always_require_fused_v1(draft_kind):
    from test_tq4_m72_support import tiny_model

    lm = tiny_model(layers=4)
    if draft_kind in ("omitted", "mtp"):
        mv.apply_to_model(lm, mv.JointV1Policy())
    kw = {} if draft_kind == "omitted" else {"draft_kind": draft_kind}
    with pytest.raises(t4.Tq4AttentionError, match="fused_v1"):
        t4.require_policy_stamps(lm, **kw)


@pytest.mark.parametrize("draft_kind", OTHER_KINDS)
def test_stamps_other_drafter_kinds_are_refused(draft_kind):
    with pytest.raises(t4.Tq4AttentionError, match="draft_kind mtp or no drafter"):
        t4.require_policy_stamps(_stamped(True), draft_kind=draft_kind)


# ================================================================== criterion 3: the server load order
needs_metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the self-tests run the kernels"
)


class _Stop(Exception):
    pass


@pytest.fixture
def generation():
    from mlx_vlm.server import generation as g

    return g


def _worker(**over):
    base = dict(
        model_path="x",
        adapter_path=None,
        draft_kind_override=None,
        draft_model_path=None,
        apc_manager=None,
        kv_bits=4,
        kv_quant_scheme="turboquant",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _init(
    generation,
    monkeypatch,
    *,
    mode="m72_exact",
    scan=None,
    draft_kind=None,
    draft_path=None,
    loaded=None,
    ab=False,
    **worker,
):
    """Run ``_initialize_model`` on the synthetic model up to the tokenizer pre-warm; returns (worker, lm)."""
    from test_tq4_m72_support import tiny_model

    lm = tiny_model(layers=4)
    model = SimpleNamespace(config=lm.config, language_model=lm)
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    monkeypatch.setattr(
        generation,
        "load_model_resources",
        lambda *a, **k: (model, processor, lm.config),
    )
    monkeypatch.setattr(generation, "resolve_eos_token_ids", lambda *a, **k: [])
    monkeypatch.setattr(generation, "get_configured_context_limit", lambda: 4096)

    def stop(_tok):
        raise _Stop()

    monkeypatch.setattr(generation, "prewarm_special_token_encodes", stop)
    for k in (
        "MLX_VLM_MOE_EXPAND",
        "MLX_VLM_MTP_VERIFY_AB",
        "MLX_VLM_DRAFT_MODEL",
        "MLX_VLM_DRAFT_KIND",
        "MLX_VLM_MTP_VERIFY_SCAN",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MLX_VLM_TQ4_ATTENTION", mode or "off")
    monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
    if scan is not None:
        monkeypatch.setenv("MLX_VLM_MTP_VERIFY_SCAN", scan)
    if draft_kind is not None:
        monkeypatch.setenv("MLX_VLM_DRAFT_KIND", draft_kind)
    if ab:
        monkeypatch.setenv("MLX_VLM_MTP_VERIFY_AB", "1")
    if loaded is not None:
        from mlx_vlm.speculative import drafters

        monkeypatch.setattr(drafters, "load_drafter", lambda path, kind=None: loaded)
        monkeypatch.setattr(
            drafters, "validate_drafter_compatibility", lambda *a, **k: None
        )
    w = _worker(draft_model_path=draft_path, **worker)
    with pytest.raises(_Stop):
        generation.ResponseGenerator._initialize_model(w)
    return w, lm


def _attns(lm):
    return [lm.model.layers[i].self_attn for i in (1, 3)]


@needs_metal
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scan", (None, "per_query", ""))
def test_init_mtp_off_applies_the_mode_without_a_verify_stamp(
    generation, monkeypatch, mode, scan
):
    w, lm = _init(generation, monkeypatch, mode=mode, scan=scan)
    assert w.tq4_attention is not None and w.tq4_attention.name == mode
    assert w.mtp_verify_policy is None and w.draft_model is None
    for a in _attns(lm):
        assert a.tq4_attention is w.tq4_attention
        assert a.attention_policy.name == "fused_v1"
        assert getattr(a, "mtp_verify_policy", None) is None


def _decode_routes(lm, T, n=4):
    from test_tq4_m72_support import counted, greedy, model_cache, prefill

    cache = model_cache(lm, "tq4", 4096)
    with counted() as c:
        last = prefill(lm, cache, T)
    prefill_delta = c.delta
    with counted() as c:
        greedy(lm, cache, last, n)
    return prefill_delta, c.delta


@needs_metal
@pytest.mark.parametrize(
    "T,route", [(2100, "d2"), (600, "single_pass")], ids=["above_2048", "at_most_2048"]
)
def test_init_mtp_off_decodes_through_the_mode_routes(
    generation, monkeypatch, T, route
):
    w, lm = _init(generation, monkeypatch, mode="m72_exact")
    pre, dec = _decode_routes(lm, T)
    assert pre.get("p2s", 0) > 0 and pre.get("native:prefill", 0) > 0, pre
    assert dec.get(route, 0) > 0 and dec.get("native:decode", 0) > 0, dec
    for d in (pre, dec):
        assert not [k for k in d if k.startswith("fallback:")], d
        assert not [k for k in d if "verify" in k], d


@needs_metal
@pytest.mark.parametrize("scan", (None, "per_query"))
def test_init_drafter_configured_without_a_scan_is_refused(
    generation, monkeypatch, scan
):
    with pytest.raises(t4.Tq4AttentionError, match="requires mtp_verify_scan joint_v1"):
        _init(generation, monkeypatch, scan=scan, draft_kind="mtp", draft_path="d")
    with pytest.raises(t4.Tq4AttentionError, match="requires mtp_verify_scan joint_v1"):
        _init(
            generation,
            monkeypatch,
            scan=scan,
            draft_kind_override="mtp",
            draft_path="d",
        )


@needs_metal
def test_init_no_drafter_with_joint_v1_is_refused(generation, monkeypatch):
    with pytest.raises(t4.Tq4AttentionError, match="joint_v1 requires a drafter"):
        _init(generation, monkeypatch, scan="joint_v1")


@needs_metal
@pytest.mark.parametrize("mode", MODES)
def test_init_no_drafter_with_verify_ab_is_refused(generation, monkeypatch, mode):
    with pytest.raises(
        mv.VerifyScanError, match="--mtp-verify-ab requires --mtp-verify-scan joint_v1"
    ):
        _init(generation, monkeypatch, mode=mode, ab=True)


@needs_metal
@pytest.mark.parametrize("kind", ("suffix",))
def test_init_other_drafter_kind_is_refused(generation, monkeypatch, kind):
    with pytest.raises(t4.Tq4AttentionError, match="draft_kind mtp or no drafter"):
        _init(generation, monkeypatch, draft_kind=kind)


@needs_metal
@pytest.mark.parametrize("resolved", ("mtp", "dflash"))
def test_init_off_with_a_loaded_drafter_is_refused(generation, monkeypatch, resolved):
    with pytest.raises(t4.Tq4AttentionError, match="configured with no drafter"):
        _init(generation, monkeypatch, draft_path="d", loaded=(object(), resolved))


def test_init_mtp_on_passes_the_m72_calls_unchanged(generation, monkeypatch):
    """MTP ON: the TQ4 apply call is exactly M72's (no draft_kind); the resolve call carries "mtp"."""
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))
    monkeypatch.setattr(
        generation,
        "load_model_resources",
        lambda *a, **k: (model, SimpleNamespace(), model.config),
    )
    monkeypatch.delenv("MLX_VLM_MOE_EXPAND", raising=False)
    monkeypatch.delenv("MLX_VLM_DRAFT_KIND", raising=False)
    monkeypatch.setattr(
        generation,
        "_resolve_tq4_mode_from_env",
        lambda **kw: calls.append(("resolve", kw)) or "m72_exact",
    )
    monkeypatch.setattr(generation, "_apply_attention_policy_from_env", lambda m: None)
    monkeypatch.setattr(
        generation,
        "_apply_mtp_verify_from_env",
        lambda m, **kw: calls.append(("verify", kw)),
    )

    def tq4(m, mode, **kw):
        calls.append(("tq4", mode, kw))
        raise _Stop()

    monkeypatch.setattr(generation, "_apply_tq4_attention_from_env", tq4)
    with pytest.raises(_Stop):
        generation.ResponseGenerator._initialize_model(
            _worker(draft_kind_override="mtp")
        )
    assert calls[0] == (
        "resolve",
        dict(kv_bits=4, kv_quant_scheme="turboquant", draft_kind="mtp"),
    )
    assert calls[1][1]["draft_kind"] == "mtp"
    assert calls[2] == ("tq4", "m72_exact", {})

    calls.clear()
    with pytest.raises(_Stop):
        generation.ResponseGenerator._initialize_model(_worker())
    assert calls[0][1]["draft_kind"] is None and calls[1][1]["draft_kind"] is None
    assert calls[2] == ("tq4", "m72_exact", {"draft_kind": None})


# ================================================================== criterion 4: greedy MTP OFF == MTP ON (m72_exact)
def _arm_off(n, T):
    """MTP OFF as the worker builds it: fused_v1 stamped, no verify stamp, the mode applied by the loader."""
    from test_tq4_m72_support import TQ4_KV, mtp_run, prompt_ids, tiny_model

    from mlx_vlm.server import generation as g

    lm = tiny_model(layers=4)
    model = SimpleNamespace(config=lm.config, language_model=lm)
    ap.apply_to_model(model, ap.FusedV1Policy())
    g._apply_tq4_attention_from_env(model, "m72_exact", draft_kind=None)
    toks, _ = mtp_run(lm, None, n=n, kv=TQ4_KV, block=4, prompt=prompt_ids(T))
    return toks


def _arm_on(n, T, block, path):
    """MTP ON: fused_v1 + joint_v1 stamped, the mode applied, an MTP drafter scripted from ``path``."""
    from test_mtp_exactness import mtp_drafter, scripted
    from test_tq4_m72_support import TQ4_KV, counted, mtp_run, prompt_ids, tiny_model

    from mlx_vlm.server import generation as g

    lm = tiny_model(layers=4)
    model = SimpleNamespace(config=lm.config, language_model=lm)
    ap.apply_to_model(model, ap.FusedV1Policy())
    mv.apply_to_model(model, mv.JointV1Policy())
    g._apply_tq4_attention_from_env(model, "m72_exact")
    drafter = mtp_drafter(lm, block=block)
    script = scripted(
        drafter, lambda p: path[p] if p < len(path) else None, high=True, vocab=64
    )
    with counted() as c:
        toks, _ = mtp_run(
            lm, drafter, n=n, kv=TQ4_KV, block=block, prompt=prompt_ids(T)
        )
    return toks, c.delta, script


@needs_metal
@pytest.mark.parametrize("block", (3, 5))
def test_greedy_exact_mtp_off_equals_mtp_on(block):
    n, T = 128, 2100
    off_a, off_b = _arm_off(n, T), _arm_off(n, T)
    assert off_a == off_b and len(off_a) == n  # repeat control, MTP OFF
    on_a, _, _ = _arm_on(n, T, block, off_a)
    on_b, delta, script = _arm_on(n, T, block, off_a)
    assert on_a == on_b  # repeat control, MTP ON
    assert script.rounds, "no speculative round ran"
    assert delta.get("d2_verify", 0) > 0, delta
    assert off_a == on_a
