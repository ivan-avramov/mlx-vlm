"""M77 fix round: MTP OFF + ``MLX_VLM_MTP_VERIFY_AB=1`` is refused at load (``_initialize_model``).

The behaviour pre-exists M77 (``mtp_verify_scan.resolve_policy`` refuses an A/B request without ``joint_v1``, reached through
``_apply_mtp_verify_from_env``); this file pins it for the new MTP-OFF configuration, with the mode on and absent.

Every test is classed in its comment: (a) reproduces a failure the specification fixes, (b) preserves an invariant,
(c) new behaviour. Both tests here are (b): they pass on the baseline-plus-candidate before the fix round and after it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_tq4_m72_support import ENV_MODE, MODES, mode_module, needs_metal, tiny_model

from mlx_vlm import mtp_verify_scan as mv
from mlx_vlm.server import generation as generation_module

ENVS = (
    ENV_MODE,
    "TQ_DECODE_2PASS_LEGACY",
    "TQ_KV_QUANT_MODE",
    "MLX_VLM_ATTENTION_POLICY",
    "MLX_VLM_MTP_VERIFY_SCAN",
    "MLX_VLM_MTP_VERIFY_AB",
    "MLX_VLM_DRAFT_KIND",
    "MLX_VLM_DRAFT_MODEL",
    "MLX_VLM_MOE_EXPAND",
)


def init(monkeypatch, *, mode, scan=None):
    for name in ENVS:
        monkeypatch.delenv(name, raising=False)
    if mode:
        monkeypatch.setenv(ENV_MODE, mode)
    monkeypatch.setenv("MLX_VLM_ATTENTION_POLICY", "fused_v1")
    monkeypatch.setenv("MLX_VLM_MTP_VERIFY_AB", "1")
    if scan:
        monkeypatch.setenv("MLX_VLM_MTP_VERIFY_SCAN", scan)
    lm = tiny_model(layers=4)
    model = SimpleNamespace(language_model=lm, config=lm.config)
    monkeypatch.setattr(
        generation_module,
        "load_model_resources",
        lambda *a, **k: (model, SimpleNamespace(), lm.config),
    )
    loads = []
    from mlx_vlm.speculative import drafters

    monkeypatch.setattr(
        drafters, "load_drafter", lambda *a, **k: loads.append(a) or (object(), "mtp")
    )
    fake = SimpleNamespace(
        model_path="x",
        adapter_path=None,
        draft_kind_override=None,
        draft_model_path=None,
        apc_manager=None,
        kv_bits=4,
        kv_quant_scheme="turboquant",
    )
    exc = None
    try:
        generation_module.ResponseGenerator._initialize_model(fake)
    except Exception as e:  # noqa: BLE001 - asserted on below
        exc = e
    return SimpleNamespace(fake=fake, lm=lm, exc=exc, loads=loads)


# (b) mode on, no drafter, A/B requested: refused with the verify-scan error naming the A/B flag, before any drafter loads and
# before the mode is applied to any module; both with the scan unset and with it spelled per_query
@needs_metal
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scan", [None, "per_query"])
def test_mtp_off_with_verify_ab_is_refused_at_load(monkeypatch, mode, scan):
    r = init(monkeypatch, mode=mode, scan=scan)
    assert isinstance(r.exc, mv.VerifyScanError), repr(r.exc)
    assert "mtp-verify-ab" in str(r.exc)
    assert r.loads == []
    assert not hasattr(r.fake, "tq4_attention")
    ta = mode_module()
    assert not any(
        getattr(m, "tq4_attention", None) for _, m in ta._attention_modules(r.lm)[1]
    )


# (b) mode absent, no drafter, A/B requested: refused the same way (the pre-existing rule, unchanged by M77)
def test_mode_absent_with_verify_ab_is_refused_at_load(monkeypatch):
    r = init(monkeypatch, mode=None)
    assert isinstance(r.exc, mv.VerifyScanError), repr(r.exc)
    assert "mtp-verify-ab" in str(r.exc)
    assert r.loads == []
