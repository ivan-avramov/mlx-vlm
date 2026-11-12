"""M77 (serve the TQ4 attention mode with MTP OFF): the fork half.

Spec: ``.fe-material/m77-tq4-mtp-off.md`` (Interfaces: ``tq4_attention.require_environment`` / ``require_policy_stamps``,
``server/generation.py`` ``_resolve_tq4_mode_from_env`` / ``_apply_tq4_attention_from_env`` / ``_initialize_model``;
acceptance criteria 2, 3, 4, 5).

Data formats are the ones the code uses: keyword arguments of ``require_environment``, the environment variables the server
reads (``MLX_VLM_TQ4_ATTENTION``, ``MLX_VLM_ATTENTION_POLICY``, ``MLX_VLM_MTP_VERIFY_SCAN``, ``MLX_VLM_DRAFT_KIND``,
``MLX_VLM_DRAFT_MODEL``), the route counters of ``tq4_fast.route_counts()``, and the synthetic Qwen3.5-family model of
``test_tq4_m72_support``. Nothing loads a real model or starts a server.

Every test is classed in its comment: (a) reproduces a failure the specification fixes, (b) preserves an invariant,
(c) new behaviour. A test that passes the new ``draft_kind`` keyword (``require_environment``, ``require_policy_stamps``, the
two generation helpers) is (c) by interface whatever its comment says: a "(b)" on such a test means the row's behaviour is
today's, reached through the new keyword; on the baseline it fails with a TypeError.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_mtp_exactness import mtp_drafter, scripted
from test_tq4_m72_routing import stamp_policies
from test_tq4_m72_support import (
    ENV_MODE,
    MODES,
    TQ4_KV,
    counted,
    greedy,
    mode_module,
    model_cache,
    mtp_run,
    mx,
    needs_metal,
    prefill,
    prompt_ids,
    tiny_model,
)

from mlx_vlm import attention_policy as ap
from mlx_vlm import mtp_verify_scan as mv
from mlx_vlm.server import generation as generation_module

VALID = dict(
    kv_bits=4,
    kv_quant_scheme="turboquant",
    attention_policy="fused_v1",
    mtp_verify_scan="joint_v1",
)
OFF_KW = dict(VALID, mtp_verify_scan="per_query")
OTHER_KINDS = ("suffix", "dflash", "eagle3")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        ENV_MODE,
        "TQ_DECODE_2PASS_LEGACY",
        "TQ_KV_QUANT_MODE",
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


def refuses(ta, fn, *a, **k):
    with pytest.raises(ta.Tq4AttentionError) as e:
        fn(*a, **k)
    return str(e.value)


# =============================================================================== require_environment (criterion 2)
# --- sentinel: omitted draft_kind is today's check
# (b) omitted: joint_v1 accepted
@pytest.mark.parametrize("mode", MODES)
def test_env_omitted_draft_kind_accepts_joint_v1(ta, mode):
    assert ta.require_environment(mode, **VALID) is None


# (b) omitted: any scan other than joint_v1 refused (today, and it is the sentinel's whole meaning)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scan", ["per_query", "", None])
def test_env_omitted_draft_kind_refuses_a_missing_joint_scan(ta, mode, scan):
    msg = refuses(ta, ta.require_environment, mode, **dict(VALID, mtp_verify_scan=scan))
    assert "joint_v1" in msg


# --- draft_kind "mtp": today's checks
# (b)
@pytest.mark.parametrize("mode", MODES)
def test_env_mtp_accepts_joint_v1(ta, mode):
    assert ta.require_environment(mode, draft_kind="mtp", **VALID) is None


# (b)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scan", ["per_query", "", None])
def test_env_mtp_refuses_a_missing_joint_scan(ta, mode, scan):
    msg = refuses(
        ta,
        ta.require_environment,
        mode,
        draft_kind="mtp",
        **dict(VALID, mtp_verify_scan=scan),
    )
    assert "joint_v1" in msg


# --- draft_kind None / "": OFF
# (c) no drafter, no joint scan: accepted for each no-scan spelling
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", [None, ""])
@pytest.mark.parametrize("scan", ["per_query", "", None])
def test_env_off_accepts_no_joint_scan(ta, mode, kind, scan):
    kw = dict(VALID, mtp_verify_scan=scan)
    assert ta.require_environment(mode, draft_kind=kind, **kw) is None


# (c) no drafter + joint_v1: refused, naming the drafter
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", [None, ""])
def test_env_off_refuses_joint_v1(ta, mode, kind):
    msg = refuses(ta, ta.require_environment, mode, draft_kind=kind, **VALID)
    assert "drafter" in msg


# (c) the other checks still apply with the drafter OFF (each knob alone wrong)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "over",
    [
        dict(kv_bits=0),
        dict(kv_bits=None),
        dict(kv_bits=8),
        dict(kv_quant_scheme="uniform"),
        dict(kv_quant_scheme=None),
        dict(attention_policy="auto"),
        dict(attention_policy=None),
    ],
    ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()),
)
def test_env_off_still_refuses_each_wrong_knob(ta, mode, over):
    refuses(ta, ta.require_environment, mode, draft_kind=None, **dict(OFF_KW, **over))


# (c) the OFF path accepts kv_bits 4.0 and the MSE codec spelled explicitly; refuses the prod codec
@pytest.mark.parametrize("mode", MODES)
def test_env_off_codec_and_bits_spellings(ta, mode):
    assert (
        ta.require_environment(
            mode, draft_kind=None, **dict(OFF_KW, kv_bits=4.0), kv_quant_mode="mse"
        )
        is None
    )
    refuses(
        ta,
        ta.require_environment,
        mode,
        draft_kind=None,
        **OFF_KW,
        kv_quant_mode="prod",
    )


# (c) the legacy decode flag is refused with the drafter OFF too
@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_env_off_refuses_the_legacy_decode_flag(ta, monkeypatch, value):
    monkeypatch.setenv("TQ_DECODE_2PASS_LEGACY", value)
    msg = refuses(ta, ta.require_environment, "m72_exact", draft_kind=None, **OFF_KW)
    assert "TQ_DECODE_2PASS_LEGACY" in msg


# --- any other kind is refused (with or without a joint scan)
# (b) refused today by the scan rule when no scan; (a/c) with joint_v1 today's check accepts it, the change must refuse
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", OTHER_KINDS + ("bogus", "MTP"))
def test_env_other_kind_refused_without_scan(ta, mode, kind):
    refuses(ta, ta.require_environment, mode, draft_kind=kind, **OFF_KW)


# (a) today's require_environment ignores the kind and accepts joint_v1: the mode's verify routes are MTP-scoped
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", OTHER_KINDS + ("bogus", "MTP"))
def test_env_other_kind_refused_with_joint_scan(ta, mode, kind):
    refuses(ta, ta.require_environment, mode, draft_kind=kind, **VALID)


# --- mode None: a no-op whatever the draft kind
# (c) the draft_kind keyword exists and a None mode is a no-op for every value
@pytest.mark.parametrize("kind", [None, "", "mtp", "suffix", "bogus"])
def test_env_mode_none_is_a_noop_for_every_draft_kind(ta, kind):
    assert (
        ta.require_environment(
            None,
            draft_kind=kind,
            kv_bits=0,
            kv_quant_scheme=None,
            attention_policy=None,
            mtp_verify_scan=None,
        )
        is None
    )


# (b) the refusal goes through ``_fail``: the module's error type and its message prefix
@pytest.mark.parametrize("kind", [None, "suffix"])
def test_env_refusals_use_the_modules_error_type_and_prefix(ta, kind, capsys):
    msg = refuses(ta, ta.require_environment, "m72_v1", draft_kind=kind, **VALID)
    assert msg.startswith("tq4-attention: ")
    assert msg in capsys.readouterr().err


# =============================================================================== require_policy_stamps (criterion 2)
def staged(fused=True, joint=False):
    lm = tiny_model(layers=4)
    if fused:
        ap.apply_to_model(lm, ap.FusedV1Policy())
    if joint:
        mv.apply_to_model(lm, mv.JointV1Policy())
    return lm


def modules_of(ta, lm):
    return [m for _, m in ta._attention_modules(lm)[1]]


# --- sentinel / "mtp": today's checks
# (b) both stamps: accepted, returns the module count
@pytest.mark.parametrize("kw", [{}, {"draft_kind": "mtp"}], ids=["sentinel", "mtp"])
def test_stamps_with_joint_accepted(ta, kw):
    lm = staged(joint=True)
    assert ta.require_policy_stamps(lm, **kw) == len(modules_of(ta, lm)) >= 2


# (b) a missing verify stamp is refused
@pytest.mark.parametrize("kw", [{}, {"draft_kind": "mtp"}], ids=["sentinel", "mtp"])
def test_stamps_without_joint_refused(ta, kw):
    msg = refuses(ta, ta.require_policy_stamps, staged(joint=False), **kw)
    assert "joint_v1" in msg


# (b) one module without the verify stamp is enough to refuse
@pytest.mark.parametrize("kw", [{}, {"draft_kind": "mtp"}], ids=["sentinel", "mtp"])
def test_stamps_one_module_missing_joint_refused(ta, kw):
    lm = staged(joint=True)
    modules_of(ta, lm)[-1].mtp_verify_policy = None
    refuses(ta, ta.require_policy_stamps, lm, **kw)


# --- fused_v1 is always required
# (b) missing fused_v1 refused for every draft_kind
@pytest.mark.parametrize(
    "kw", [{}, {"draft_kind": "mtp"}, {"draft_kind": None}, {"draft_kind": ""}]
)
def test_stamps_without_fused_refused_always(ta, kw):
    lm = staged(fused=False, joint=kw.get("draft_kind", "mtp") == "mtp")
    msg = refuses(ta, ta.require_policy_stamps, lm, **kw)
    assert "fused_v1" in msg


# (b) one module without fused_v1 is enough to refuse (OFF and ON)
@pytest.mark.parametrize("kw", [{"draft_kind": "mtp"}, {"draft_kind": None}])
def test_stamps_one_module_missing_fused_refused(ta, kw):
    lm = staged(joint=kw["draft_kind"] == "mtp")
    modules_of(ta, lm)[0].attention_policy = None
    refuses(ta, ta.require_policy_stamps, lm, **kw)


# --- None / "": OFF
# (c) fused only, no verify stamp: accepted, returns the module count
@pytest.mark.parametrize("kind", [None, ""])
def test_stamps_off_accepts_fused_only(ta, kind):
    lm = staged(joint=False)
    assert ta.require_policy_stamps(lm, draft_kind=kind) == len(modules_of(ta, lm)) >= 2


# (c) an explicit ``mtp_verify_policy = None`` is "absent"
@pytest.mark.parametrize("kind", [None, ""])
def test_stamps_off_accepts_an_explicit_none_verify_stamp(ta, kind):
    lm = staged(joint=True)
    for m in modules_of(ta, lm):
        m.mtp_verify_policy = None
    assert ta.require_policy_stamps(lm, draft_kind=kind) >= 2


# (c) a verify stamp present with no drafter is refused: on every module, and on a single one
@pytest.mark.parametrize("kind", [None, ""])
def test_stamps_off_refuses_a_verify_stamp(ta, kind):
    refuses(ta, ta.require_policy_stamps, staged(joint=True), draft_kind=kind)


@pytest.mark.parametrize("kind", [None, ""])
@pytest.mark.parametrize("which", [0, -1])
def test_stamps_off_refuses_a_single_stamped_module(ta, kind, which):
    lm = staged(joint=False)
    modules_of(ta, lm)[which].mtp_verify_policy = mv.JointV1Policy()
    refuses(ta, ta.require_policy_stamps, lm, draft_kind=kind)


# =============================================================================== generation helpers
def model_of(lm):
    return SimpleNamespace(language_model=lm, config=lm.config)


def set_env(monkeypatch, mode="m72_exact", *, scan=None, kind=None, policy="fused_v1"):
    monkeypatch.setenv(ENV_MODE, mode)
    if policy:
        monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", policy)
    if scan:
        monkeypatch.setenv("MLX_VLM_MTP_VERIFY_SCAN", scan)
    if kind is not None:
        monkeypatch.setenv("MLX_VLM_DRAFT_KIND", kind)


def resolve(draft_kind, kv_bits=4):
    return generation_module._resolve_tq4_mode_from_env(
        kv_bits=kv_bits, kv_quant_scheme="turboquant", draft_kind=draft_kind
    )


# --- _resolve_tq4_mode_from_env
# (c) configured OFF, no scan: the mode resolves
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", [None, ""])
def test_resolve_off_without_scan_returns_the_mode(monkeypatch, mode, kind):
    set_env(monkeypatch, mode)
    assert resolve(kind) == mode


# (c) configured OFF + scan per_query spelled out
def test_resolve_off_with_explicit_per_query(monkeypatch):
    set_env(monkeypatch, scan="per_query")
    assert resolve(None) == "m72_exact"


# (c) configured OFF + joint_v1 in the environment: refused (no drafter)
@pytest.mark.parametrize("kind", [None, ""])
def test_resolve_off_with_joint_scan_refused(ta, monkeypatch, kind):
    set_env(monkeypatch, scan="joint_v1")
    refuses(ta, resolve, kind)


# (b) configured mtp + joint_v1: resolves; mtp + no scan: refused
def test_resolve_mtp_requires_joint_scan(ta, monkeypatch):
    set_env(monkeypatch, scan="joint_v1")
    assert resolve("mtp") == "m72_exact"
    monkeypatch.delenv("MLX_VLM_MTP_VERIFY_SCAN")
    refuses(ta, resolve, "mtp")


# (a) other kinds refused even with joint_v1 (today's check accepts the scan)
@pytest.mark.parametrize("kind", OTHER_KINDS)
@pytest.mark.parametrize("scan", [None, "joint_v1"])
def test_resolve_other_kinds_refused(ta, monkeypatch, kind, scan):
    set_env(monkeypatch, scan=scan)
    refuses(ta, resolve, kind)


# (c) the OFF refusals of the other knobs still fire (attention policy, kv_bits)
def test_resolve_off_still_checks_policy_and_bits(ta, monkeypatch):
    set_env(monkeypatch, policy="auto")
    refuses(ta, resolve, None)
    set_env(monkeypatch)
    refuses(ta, resolve, None, kv_bits=0)


# (b) mode absent: None whatever the draft kind
@pytest.mark.parametrize("kind", [None, "", "mtp", "suffix"])
def test_resolve_mode_absent_returns_none(kind):
    assert resolve(kind) is None


# --- _apply_tq4_attention_from_env
def apply(lm, mode, draft_kind):
    return generation_module._apply_tq4_attention_from_env(
        model_of(lm), mode, draft_kind=draft_kind
    )


# (b) mode None: None, nothing stamped, whatever the draft kind
@pytest.mark.parametrize("kind", [None, "mtp"])
def test_apply_mode_none_is_a_noop(kind):
    lm = staged()
    assert apply(lm, None, kind) is None
    assert not any(
        getattr(m, "tq4_attention", None) for m in modules_of(mode_module(), lm)
    )


# (c) OFF stamps: fused only -> hook applied on every module
@needs_metal
@pytest.mark.parametrize("mode", MODES)
def test_apply_off_stamps_the_mode(ta, mode):
    lm = staged(joint=False)
    hook = apply(lm, mode, None)
    assert hook is not None and hook.name == mode
    assert all(m.tq4_attention is hook for m in modules_of(ta, lm))


# (c) OFF + a verify stamp: refused, nothing stamped
def test_apply_off_with_verify_stamp_refused(ta):
    lm = staged(joint=True)
    refuses(ta, apply, lm, "m72_exact", None)
    assert not any(getattr(m, "tq4_attention", None) for m in modules_of(ta, lm))


# (b) mtp + both stamps applied; mtp + fused only refused
@needs_metal
def test_apply_mtp_requires_the_verify_stamp(ta):
    lm = staged(joint=True)
    assert apply(lm, "m72_exact", "mtp") is not None
    refuses(ta, apply, staged(joint=False), "m72_exact", "mtp")


# =============================================================================== _initialize_model load order (criterion 3)
@pytest.fixture
def loader(monkeypatch, ta):
    """``ResponseGenerator._initialize_model`` over the synthetic model, with the real mode stages, the real attention-policy
    and verify-scan stamping, a patched drafter loader (records its calls) and no processor.
    """

    def run(
        mode="m72_exact",
        *,
        scan=None,
        kind=None,
        override=None,
        draft_path=None,
        loaded_kind="mtp",
        kv_bits=4,
    ):
        lm = tiny_model(layers=4)
        set_env(monkeypatch, mode, scan=scan, kind=kind)
        if not mode:
            monkeypatch.delenv(ENV_MODE, raising=False)
        if draft_path:
            monkeypatch.setenv("MLX_VLM_DRAFT_MODEL", draft_path)
        model = model_of(lm)
        monkeypatch.setattr(
            generation_module,
            "load_model_resources",
            lambda *a, **k: (model, SimpleNamespace(), lm.config),
        )
        monkeypatch.setattr(
            generation_module, "resolve_eos_token_ids", lambda *a, **k: []
        )
        monkeypatch.setattr(
            generation_module, "prewarm_special_token_encodes", lambda *a, **k: None
        )
        monkeypatch.setattr(mv, "self_test_model", lambda *a, **k: [])
        calls = []
        from mlx_vlm.speculative import drafters

        sentinel = object()

        def fake_load(path, kind=None):
            calls.append((path, kind))
            return sentinel, loaded_kind

        monkeypatch.setattr(drafters, "load_drafter", fake_load)
        monkeypatch.setattr(
            drafters, "validate_drafter_compatibility", lambda *a, **k: None
        )
        fake = SimpleNamespace(
            model_path="x",
            adapter_path=None,
            draft_kind_override=override,
            draft_model_path=None,
            apc_manager=None,
            kv_bits=kv_bits,
            kv_quant_scheme="turboquant",
        )
        exc = None
        try:
            generation_module.ResponseGenerator._initialize_model(fake)
        except Exception as e:  # noqa: BLE001 - the tests assert on the type
            exc = e
        return SimpleNamespace(fake=fake, lm=lm, exc=exc, loads=calls, drafter=sentinel)

    return run


# (c) the traced success path: mode, no drafter, no scan -> loads with the mode applied and no verify stamp
@needs_metal
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", [None, ""], ids=["kind-unset", "kind-empty"])
def test_load_off_applies_the_mode_without_a_verify_stamp(loader, ta, mode, kind):
    r = loader(mode, kind=kind)
    assert r.exc is None, repr(r.exc)
    assert r.fake.tq4_attention is not None and r.fake.tq4_attention.name == mode
    assert r.fake.mtp_verify_policy is None
    assert r.fake.draft_model is None
    mods = modules_of(ta, r.lm)
    assert mods and all(m.tq4_attention is r.fake.tq4_attention for m in mods)
    assert all(getattr(m, "mtp_verify_policy", None) is None for m in mods)
    assert all(m.attention_policy.name == "fused_v1" for m in mods)
    assert r.loads == []


# (c) the same with the scan spelled per_query
@needs_metal
def test_load_off_with_explicit_per_query_scan(loader):
    r = loader("m72_exact", scan="per_query")
    assert r.exc is None, repr(r.exc)
    assert r.fake.mtp_verify_policy is None and r.fake.tq4_attention is not None


# (b) MTP ON today: mtp + joint_v1 + a loaded MTP drafter -> loads with both stamps (env or override spelling)
@needs_metal
@pytest.mark.parametrize("via", ["env", "override"])
def test_load_mtp_with_joint_v1_unchanged(loader, ta, via):
    r = loader(
        scan="joint_v1",
        kind="mtp" if via == "env" else None,
        override="mtp" if via == "override" else None,
        draft_path="/drafter",
    )
    assert r.exc is None, repr(r.exc)
    assert r.fake.tq4_attention is not None
    assert (
        r.fake.mtp_verify_policy is not None
        and r.fake.mtp_verify_policy.name == "joint_v1"
    )
    assert r.fake.draft_model is r.drafter and r.fake.draft_kind == "mtp"
    assert all(m.mtp_verify_policy is not None for m in modules_of(ta, r.lm))


# (b) a drafter configured with no scan is refused before the drafter is loaded (spec: "with a drafter configured and no scan
# it refuses"); today by the scan rule
@pytest.mark.parametrize("via", ["env", "override"])
def test_load_mtp_without_scan_refused_before_the_drafter_loads(loader, ta, via):
    r = loader(
        kind="mtp" if via == "env" else None,
        override="mtp" if via == "override" else None,
        draft_path="/drafter",
    )
    assert isinstance(r.exc, ta.Tq4AttentionError), repr(r.exc)
    assert r.loads == []
    assert not hasattr(r.fake, "tq4_attention")


# (b) mtp + joint_v1 but the drafter resolves to another kind: refused as today (require_loaded_mtp_drafter)
def test_load_mtp_with_a_drafter_resolving_elsewhere_refused(loader):
    r = loader(scan="joint_v1", kind="mtp", draft_path="/drafter", loaded_kind="dflash")
    assert r.exc is not None and r.loads


# (b) joint_v1 and no drafter configured: refused (no drafter for the verify stamp), no drafter loaded
@pytest.mark.parametrize("kind", [None, ""])
def test_load_joint_v1_without_a_drafter_refused(loader, kind):
    r = loader(scan="joint_v1", kind=kind)
    assert r.exc is not None
    assert r.loads == []
    assert getattr(r.fake, "tq4_attention", None) is None


# (b)/(a) other kinds with the mode: refused without applying the mode (with and without a scan)
@pytest.mark.parametrize("kind", OTHER_KINDS)
@pytest.mark.parametrize("scan", [None, "joint_v1"])
def test_load_other_kinds_refused(loader, kind, scan):
    r = loader(
        scan=scan,
        kind=kind,
        draft_path="/drafter" if kind != "suffix" else None,
        loaded_kind=kind,
    )
    assert r.exc is not None
    assert getattr(r.fake, "tq4_attention", None) is None


# (c) configured OFF with a LOADED drafter: refused after the load (the drafter really loaded: the refusal is the post-load one)
@pytest.mark.parametrize("loaded_kind", ["mtp", "dflash"])
def test_load_off_with_a_loaded_drafter_refused_after_load(loader, ta, loaded_kind):
    r = loader(kind=None, draft_path="/drafter", loaded_kind=loaded_kind)
    assert r.loads, "the drafter was never loaded: the refusal came before the load"
    assert isinstance(r.exc, ta.Tq4AttentionError), repr(r.exc)


# (c) ... also when the OFF spelling is the empty kind
def test_load_off_empty_kind_with_a_loaded_drafter_refused_after_load(loader, ta):
    r = loader(kind="", draft_path="/drafter")
    assert r.loads
    assert isinstance(r.exc, ta.Tq4AttentionError), repr(r.exc)


# (b) mode absent: drafter OFF and drafter loaded both load as today, no mode stage runs
@pytest.mark.parametrize("draft_path", [None, "/drafter"])
def test_load_mode_absent_unchanged(loader, draft_path):
    r = loader(None, kind="mtp" if draft_path else None, draft_path=draft_path)
    assert r.exc is None, repr(r.exc)
    assert r.fake.tq4_attention is None and r.fake.mtp_verify_policy is None
    assert (r.fake.draft_model is not None) == bool(draft_path)


# =============================================================================== routes (criterion 3)
def off_model(mode, *, layers=4, cap=4096):
    lm = tiny_model(layers=layers)
    ap.apply_to_model(lm, ap.FusedV1Policy())
    mode_module().apply_to_model(lm, mode)
    return lm, model_cache(lm, "tq4", cap)


def route_names(delta, prefix):
    return {k: v for k, v in delta.items() if k.startswith(prefix)}


# (b) MTP OFF, mode on, TQ4 cache, no verify stamp: prefill through p2s, decode through d2 above 2,048 keys, native last layer
# through its own prefill/decode counters, never a fallback
@needs_metal
@pytest.mark.parametrize("mode", MODES)
def test_routes_above_2048(mode):
    lm, cache = off_model(mode)
    with counted() as c:
        last = prefill(lm, cache, 2300)
        greedy(lm, cache, last, 8)
    d = c.delta
    assert d.get("p2s", 0) > 0 and d.get("d2", 0) > 0, d
    assert d.get("native:prefill", 0) > 0 and d.get("native:decode", 0) > 0, d
    assert not route_names(d, "fallback:"), d
    assert d.get("mma", 0) == 0 and d.get("d2_verify", 0) == 0, d


# (b) below 2,048 keys decode is the single-pass kernel
@needs_metal
@pytest.mark.parametrize("mode", MODES)
def test_routes_below_2048(mode):
    lm, cache = off_model(mode)
    with counted() as c:
        last = prefill(lm, cache, 600)
        greedy(lm, cache, last, 8)
    d = c.delta
    assert d.get("single_pass", 0) > 0 and d.get("d2", 0) == 0, d
    assert not route_names(d, "fallback:"), d


# (b) the model carries no verify stamp in this configuration (the route test above is the OFF state, not the ON one)
def test_off_model_has_no_verify_stamp(ta):
    lm, _ = off_model("m72_exact")
    assert all(
        getattr(m, "mtp_verify_policy", None) is None for m in modules_of(ta, lm)
    )


# =============================================================================== greedy identity (criterion 4)
T_ID, N_ID, BLOCK = 2100, 64, 3


def run_off(mode):
    lm = tiny_model(layers=4)
    ap.apply_to_model(lm, ap.FusedV1Policy())
    mode_module().apply_to_model(lm, mode)
    toks, _ = mtp_run(
        lm,
        None,
        n=N_ID,
        kv=TQ4_KV,
        block=BLOCK,
        prompt=prompt_ids(T_ID),
        temperature=0.0,
    )
    return toks


def run_on(mode, proposals, *, high=True):
    lm = tiny_model(layers=4)
    stamp_policies(lm)
    mode_module().apply_to_model(lm, mode)
    drafter = mtp_drafter(lm, block=BLOCK)
    script = scripted(
        drafter,
        lambda p: proposals[p] if p < len(proposals) else None,
        high=high,
        vocab=64,
    )
    toks, _ = mtp_run(
        lm,
        drafter,
        n=N_ID,
        kv=TQ4_KV,
        block=BLOCK,
        prompt=prompt_ids(T_ID),
        temperature=0.0,
    )
    return toks, script


# (b) m72_exact greedy: MTP OFF (fused_v1 only, no verify stamp) = MTP ON (both stamps), accepting and rejecting drafters;
# each arm is repeated first (a control: the arms are deterministic in-process), and the ON arm really ran speculative rounds
# and routed its verify blocks through D2's exact kernel. (Passes on the baseline: the identity is of existing code paths; the
# load-time refusal that the spec lifts is exercised by the loader tests above.)
@needs_metal
@pytest.mark.parametrize("high", [True, False], ids=["accepting", "rejecting"])
def test_greedy_m72_exact_mtp_off_equals_mtp_on(high):
    off = run_off("m72_exact")
    assert len(off) == N_ID
    assert run_off("m72_exact") == off  # repeat control, OFF arm
    first, _ = run_on("m72_exact", off, high=high)
    with counted() as c:
        on, script = run_on("m72_exact", off, high=high)
    assert on == first  # repeat control, ON arm
    assert script.rounds, "no speculative round ran"
    assert (
        c.delta.get("d2_verify", 0) > 0 or c.delta.get("per_query:short", 0) > 0
    ), c.delta
    assert c.delta.get("mma", 0) == 0, c.delta
    assert on == off


# (b) the repeat control has teeth: a different prompt gives different tokens
@needs_metal
def test_greedy_control_distinguishes_prompts():
    a = run_off("m72_exact")
    lm = tiny_model(layers=4)
    ap.apply_to_model(lm, ap.FusedV1Policy())
    mode_module().apply_to_model(lm, "m72_exact")
    b, _ = mtp_run(
        lm,
        None,
        n=N_ID,
        kv=TQ4_KV,
        block=BLOCK,
        prompt=prompt_ids(T_ID, seed=9),
        temperature=0.0,
    )
    assert a != b


# =============================================================================== mode absent (criterion 5)
# (b) mode absent: the fork's mode module is not needed to stamp policies and plain MTP-OFF decode is untouched
@needs_metal
def test_mode_absent_decode_moves_no_tq4_route_counter():
    lm = tiny_model(layers=4)
    ap.apply_to_model(lm, ap.FusedV1Policy())
    cache = model_cache(lm, "tq4", 4096)
    with counted() as c:
        last = prefill(lm, cache, 2300)
        greedy(lm, cache, last, 4)
    assert not c.delta, c.delta


# (b) require_environment signature: the keyword the spec adds is keyword-only with a default (existing callers unchanged)
def test_require_environment_existing_call_shape_still_works(ta):
    assert ta.require_environment("m72_v1", **VALID) is None
    assert (
        ta.require_environment(
            None,
            kv_bits=0,
            kv_quant_scheme=None,
            attention_policy=None,
            mtp_verify_scan=None,
        )
        is None
    )
    lm = staged(joint=True)
    assert ta.require_policy_stamps(lm) >= 2


_ = mx  # imported for the fixtures' module-level skip logic
