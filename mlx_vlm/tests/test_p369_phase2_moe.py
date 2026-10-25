"""P369 v13 (V12 / V14): Ornith-1.0-35B-mlx-uniform-4bit's drafter construction path on tiny models (fork-only, CPU).

Ornith's MTP head was TRANSPLANTED: split from a Qwen3-Next-style source (`Qwen3NextMTPSplitter`) into a `qwen3_5_mtp`
drafter. The splitter copies the source's nested `text_config` verbatim, while `Qwen3_5MTPDraftModel` builds MoE decoder
layers only when that nested `model_type` contains "moe". These tests pin both shapes: the uploaded drafter's (nested
`qwen3_5_moe_text`) must load MoE layers, and an adversarial nested `qwen3_next` source must never yield a dense-runtime
drafter holding expert weights (it must be rewritten to the MoE type or fail).
"""

import json
from contextlib import nullcontext
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.qwen3_5_moe.config import TextConfig as MoeTextConfig
from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import (
    Qwen3_5MTPConfig as MTPConfig,
)
from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import Qwen3_5MTPDraftModel

EXPERTS, TOP_K = 4, 2


def tiny_moe_text(model_type="qwen3_5_moe_text"):
    return dict(
        model_type=model_type,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        shared_expert_intermediate_size=32,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=512,
        head_dim=16,
        mtp_num_hidden_layers=1,
    )


def _moe_drafter_weights():
    """A tiny MoE qwen3_5_mtp drafter's parameters (the shape the uploaded Ornith drafter has)."""
    mx.random.seed(3)
    cfg = MoeTextConfig.from_dict(tiny_moe_text())
    cfg.mtp_num_hidden_layers = 1
    drafter = Qwen3_5MTPDraftModel(MTPConfig(text_config=cfg, block_size=3))
    mx.eval(drafter.parameters())
    return dict(tree_flatten(drafter.parameters()))


def _qwen3_next_source(tmp, nested_type):
    """A Qwen3-Next-style HF source: `mtp.*` keys, SEPARATE per-expert gate / up / down tensors."""
    tensors = {}
    for key, value in _moe_drafter_weights().items():
        if ".mlp.switch_mlp." in key:
            stem, proj = key.split(".mlp.switch_mlp.")
            proj = proj.split(".")[0]
            for e in range(value.shape[0]):
                tensors[f"mtp.{stem}.mlp.experts.{e}.{proj}.weight"] = value[e]
        else:
            tensors["mtp." + key] = value
    path = Path(tmp) / "source"
    path.mkdir()
    text = tiny_moe_text(nested_type)
    if nested_type == "qwen3_next":
        text["intermediate_size"] = (
            64  # a real Qwen3-Next config carries it (a dense TextConfig then builds)
        )
    # A real Qwen3-Next config carries these; the splitter derives the
    # drafter's RoPE from them (#2457).
    text.update(partial_rotary_factor=0.25, rope_theta=10000.0)
    config = dict(model_type="qwen3_next", text_config=text)
    (path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / "model.safetensors"), tensors)
    return path


def _split_and_load(tmp, nested_type):
    from mlx_vlm.speculative.drafters import load_drafter
    from mlx_vlm.split_mtp import split_mtp

    out = Path(tmp) / "drafter"
    split_mtp(str(_qwen3_next_source(tmp, nested_type)), str(out))
    written = json.loads((out / "config.json").read_text())
    model, kind = load_drafter(str(out), kind="mtp")
    return written, model, kind


def test_the_uploaded_shape_nested_moe_type_loads_moe_layers(tmp_path):
    written, model, kind = _split_and_load(tmp_path, "qwen3_5_moe_text")
    assert kind == "mtp" and written["text_config"]["model_type"] == "qwen3_5_moe_text"
    assert [type(layer).__name__ for layer in model.layers] == [
        "Qwen3_5MoeDecoderLayer"
    ]
    mlp = model.layers[0].mlp
    assert mlp.switch_mlp.gate_proj.weight.shape[0] == EXPERTS


def test_an_adversarial_nested_qwen3_next_source_never_yields_a_dense_drafter_with_experts(
    tmp_path,
):
    """V14: the splitter preserves the nested `qwen3_next`; the runtime must not then build dense layers. Measured
    2026-10-20: the dense config builds, and the STRICT load refuses ("Received 8 parameters not in model: …
    switch_mlp / shared_expert …") — fail-loud, no production defect."""
    try:
        written, model, _ = _split_and_load(tmp_path, "qwen3_next")
    except (ValueError, KeyError, RuntimeError) as exc:
        assert (
            "switch_mlp" in str(exc)
            or "expert" in str(exc)
            or "moe" in str(exc).lower()
        ), exc
        return  # refused during split / strict load, naming the expert weights: acceptable
    layers = [type(layer).__name__ for layer in model.layers]
    assert layers == [
        "Qwen3_5MoeDecoderLayer"
    ], f"a nested {written['text_config']['model_type']!r} source built {layers}"


# ------------------------------------------------------------------ V3 / V15: the harness observes the MoE stack
import copy  # noqa: E402
import sys  # noqa: E402

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import observe, telemetry  # noqa: E402
from test_mtp_exactness import PROMPT, deployed_sampling, fake_model  # noqa: E402
from test_mtp_stop_state import turn  # noqa: E402

from mlx_vlm.generate.common import PromptCacheState  # noqa: E402
from mlx_vlm.models.qwen3_5_moe.config import (
    ModelConfig as MoeModelConfig,
)  # noqa: E402
from mlx_vlm.models.qwen3_5_moe.config import (
    VisionConfig as MoeVisionConfig,
)  # noqa: E402
from mlx_vlm.models.qwen3_5_moe.language import (
    LanguageModel as MoeLanguageModel,
)  # noqa: E402

LAYERS = 4  # full attention every 2nd: GDN, attention, GDN, attention
PARAMS = deployed_sampling("Qwen3.8-27B-mlx-uniform-4bit")


def tiny_moe_target(seed=0):
    mx.random.seed(seed)
    cfg = MoeTextConfig.from_dict(
        dict(tiny_moe_text(), num_hidden_layers=LAYERS, full_attention_interval=2)
    )
    vision = MoeVisionConfig(
        model_type="qwen3_5_moe",
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
    lm = MoeLanguageModel(
        cfg,
        config=MoeModelConfig(
            text_config=cfg, vision_config=vision, model_type="qwen3_5_moe"
        ),
    )
    lm.model.norm.weight = lm.model.norm.weight * 12.0
    mx.eval(lm.parameters())
    return lm


def tiny_moe_drafter(lm, seed=1):
    tc = copy.deepcopy(lm.config.text_config)
    tc.mtp_num_hidden_layers = 1
    mx.random.seed(seed)
    d = Qwen3_5MTPDraftModel(MTPConfig(text_config=tc, block_size=3))
    mx.eval(d.parameters())
    return d


def served_moe_target(seed=0):
    """The SERVED shape for bitwise / verdict claims (P390: bitwise claims are made on bf16 quantized fixtures, not dense
    float32): bf16, hidden 512, every linear and routed bank affine int4 / g64, a GDN + full-attention hybrid.
    """
    import mlx.nn as nn

    mx.random.seed(seed)
    text = dict(
        tiny_moe_text(),
        hidden_size=512,
        num_attention_heads=8,
        head_dim=64,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_hidden_layers=2,
        full_attention_interval=2,
    )
    cfg = MoeTextConfig.from_dict(text)
    vision = MoeVisionConfig(
        model_type="qwen3_5_moe",
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        out_hidden_size=512,
        num_heads=1,
        in_channels=3,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=1,
        num_position_embeddings=4,
    )
    lm = MoeLanguageModel(
        cfg,
        config=MoeModelConfig(
            text_config=cfg, vision_config=vision, model_type="qwen3_5_moe"
        ),
    )
    lm.model.norm.weight = lm.model.norm.weight * 10.0
    lm.set_dtype(mx.bfloat16)
    nn.quantize(lm, group_size=64, bits=4, mode="affine")
    mx.eval(lm.parameters())
    return lm


def served_moe_drafter(lm, seed=1):
    import mlx.nn as nn

    d = tiny_moe_drafter(lm, seed=seed)
    d.set_dtype(mx.bfloat16)
    nn.quantize(d, group_size=64, bits=4, mode="affine")
    mx.eval(d.parameters())
    return d


SERVED_LAYERS = 2


def test_v3a_the_target_is_structurally_moe():
    lm = tiny_moe_target()
    assert type(lm).__name__ == "LanguageModel" and type(lm).__module__.endswith(
        "qwen3_5_moe.language"
    )
    layers = lm.model.layers
    assert len(layers) == LAYERS
    assert [type(layer).__name__ for layer in layers] == [
        "Qwen3_5MoeDecoderLayer"
    ] * LAYERS
    assert [layer.is_linear for layer in layers] == [
        True,
        False,
        True,
        False,
    ]  # GDN / full-attention split
    moe = [
        layer.mlp
        for layer in layers
        if type(layer.mlp).__name__ == "Qwen3_5MoeSparseMoeBlock"
    ]
    assert len(moe) == LAYERS and {(m.num_experts, m.top_k) for m in moe} == {
        (EXPERTS, TOP_K)
    }


def test_v3c_the_manifest_requires_every_routed_bank_projection():
    lm = served_moe_target()
    manifest = telemetry.manifest_from_model(lm)
    routed = {n: i for n, i in manifest.items() if i["kind"] == "routed"}
    want = {
        f"model.layers.{i}.mlp.switch_mlp.{p}"
        for i in range(SERVED_LAYERS)
        for p in ("gate_proj", "up_proj", "down_proj")
    }
    assert set(routed) == want
    assert all(
        i["on_path"] and i["expected"] == dict(routed=["routed_gather_quantized"])
        for i in routed.values()
    )


METAL = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the served kernels are Metal"
)


def _observed_turn(lm, drafter, *, sabotage=None, monkeypatch=None, n=12):
    tel = telemetry.KernelTelemetry(lm, drafter=drafter)
    keys = observe.KeyLengths()  # the verifier block boundaries
    with tel, keys, observe.Recorder(mode="on", lm=lm, probe_token=5) as rec:
        # code round 2 N2: the sabotage lives in its own patch context, undone BEFORE the telemetry restores its
        # wrappers (``monkeypatch`` itself never records a telemetry wrapper as an original)
        with monkeypatch.context() if sabotage else nullcontext() as mp:
            if sabotage:
                sabotage(mp)
            out, _ = turn(
                fake_model(lm),
                PROMPT,
                PromptCacheState(),
                drafter=drafter,
                n=n,
                seed=5,
                params=PARAMS,
            )
    rec.blocks = keys.blocks
    return tel, rec, list(out)


@pytest.fixture(scope="module")
def served():
    lm = served_moe_target()
    return lm, served_moe_drafter(lm)


def _verdict(tel, lm):
    return telemetry.verdict(tel, telemetry.manifest_from_model(lm))


@METAL
def test_v3b_v3c_v15_one_mtp_request_is_fully_observed(served):
    """(b) recorder rows, commits and verifier blocks on the MoE stack; (c) the router, shared expert and each routed
    bank projection exactly once per verify-scope invocation, incl. a width > 1 block; V15 the drafter's own MoE path
    in its draft scope, with selected experts."""
    lm, drafter = served
    tel, rec, out = _observed_turn(lm, drafter)
    assert len(out) == 12 and rec.rows and rec.commits and rec.blocks
    v = _verdict(tel, lm)
    assert v["pass"], v["problems"]
    invocations = telemetry.routed_invocations(tel, scope="verify")
    assert invocations and all(
        sorted(i["projections"]) == ["down_proj", "gate_proj", "up_proj"]
        for i in invocations
    )
    assert any(i["width"] > 1 for i in invocations)
    d = telemetry.draft_verdict(tel, drafter)
    assert d["pass"], d["problems"]
    assert d["invocations"] and all(i["selected_experts"] for i in d["invocations"])


@METAL
def test_v3e_a_verifier_bypassing_a_routed_projection_fails_the_verdict(
    served, monkeypatch
):
    """The invocation observer stays; the target's routed gate bank is replaced by a plain function that runs the up
    bank instead — each invocation then records up / up / down and the verdict must refuse it.
    """
    lm, drafter = served
    for layer in lm.model.layers:
        switch = layer.mlp.switch_mlp
        up = switch.up_proj
        monkeypatch.setattr(
            switch, "gate_proj", lambda x, idx, *a, _up=up, **k: _up(x, idx, *a, **k)
        )
    tel, _, _ = _observed_turn(lm, drafter)
    v = _verdict(tel, lm)
    assert not v["pass"] and any(
        "routed invocation ran" in p for p in v["problems"]
    ), v["problems"]


@METAL
def test_v15_a_drafter_skipping_its_routed_experts_fails_the_draft_verdict(
    served, monkeypatch
):
    """The drafter's MoE block still runs under its observer; only the routed experts are skipped."""
    lm, drafter = served
    mlp = drafter.layers[0].mlp
    zeros = lambda x, inds, *a, **k: mx.zeros(
        (*inds.shape, x.shape[-1]), dtype=x.dtype
    )  # noqa: E731
    monkeypatch.setattr(mlp, "switch_mlp", zeros)
    tel, _, _ = _observed_turn(lm, drafter)
    d = telemetry.draft_verdict(tel, drafter)
    assert (
        not d["pass"] and d["invocations"] and any("ran" in p for p in d["problems"])
    ), d


@METAL
def test_v3e_an_absent_routed_hook_fails_the_verdict(served, monkeypatch):
    """Disabling the routed BANK hook (the projections run unobserved) fails coverage, never passes."""
    from mlx_vlm.models import switch_layers

    real = {
        c: c.__call__
        for c in (switch_layers.SwitchLinear, switch_layers.QuantizedSwitchLinear)
    }
    lm, drafter = served

    def unhook(mp):
        for cls, fn in real.items():
            mp.setattr(cls, "__call__", fn)

    tel, _, _ = _observed_turn(lm, drafter, sabotage=unhook, monkeypatch=monkeypatch)
    v = _verdict(tel, lm)
    assert not v["pass"] and any(
        "never observed" in p or "routed invocation ran" in p for p in v["problems"]
    )
    assert not telemetry.draft_verdict(tel, drafter)["pass"]


@METAL
def test_m4_the_split_artifact_itself_binds_and_runs_a_draft_verify_round(tmp_path):
    """M4: the split → strict-load result is bound to a MoE target and executed (not a freshly constructed drafter),
    with its certified shape asserted (float tiny fixture: the split artifact's own dims).
    """
    written, drafter, kind = _split_and_load(tmp_path, "qwen3_5_moe_text")
    assert drafter.config.block_size == 3  # depth 1 + the splitter's block_size_extra 2
    mlp = drafter.layers[0].mlp
    assert (mlp.num_experts, mlp.top_k) == (EXPERTS, TOP_K)
    assert all(
        hasattr(mlp.switch_mlp, p) for p in ("gate_proj", "up_proj", "down_proj")
    )
    lm = tiny_moe_target()
    off, _ = turn(
        fake_model(lm), PROMPT, PromptCacheState(), n=12, seed=5, params=PARAMS
    )
    tel, rec, on = _observed_turn(lm, drafter)
    assert on == list(off) and rec.draft_rounds > 0
    assert telemetry.draft_verdict(tel, drafter)["pass"]


@METAL
def test_m5_a_fused_helper_records_each_bank_exactly_once():
    """M5: SwitchGLU's exact fused path (1 < L <= DECODE_BLOCK_SIZE) — its helpers call the (wrapped) banks; each bank
    is attributed once per invocation, with the bank's own family."""
    import mlx.nn as nn

    from mlx_vlm.models import switch_layers

    mx.random.seed(4)
    glu = switch_layers.SwitchGLU(64, 64, EXPERTS)
    nn.quantize(glu, group_size=64, bits=4)
    mx.eval(glu.parameters())
    tel = telemetry.KernelTelemetry(tiny_moe_target())
    tel._names[id(glu)] = "probe.switch_mlp"
    for p in ("gate_proj", "up_proj", "down_proj"):
        tel._names[id(getattr(glu, p))] = f"probe.switch_mlp.{p}"
    x = mx.random.normal((1, 2, 64))
    indices = mx.array([[[0, 1], [2, 3]]], dtype=mx.int32)
    with tel:
        mx.eval(glu(x, indices))
    (frame,) = tel.routed_frames
    assert sorted(frame["projections"]) == ["down_proj", "gate_proj", "up_proj"], frame
    fams = {a["module"]: a["family"] for a in tel.attributions if a["kind"] == "routed"}
    assert set(fams.values()) == {"routed_gather_quantized"}, fams


@METAL
def test_m6_an_absent_invocation_hook_fails_the_verdict(served, monkeypatch):
    """M6: the bank hooks still run but the per-invocation observer is gone — never a pass."""
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv

    real = sv.Qwen3_5BatchInvariantForward._switch_glu
    lm, drafter = served
    tel, _, _ = _observed_turn(
        lm,
        drafter,
        sabotage=lambda mp: mp.setattr(
            sv.Qwen3_5BatchInvariantForward, "_switch_glu", real
        ),
        monkeypatch=monkeypatch,
    )
    v = _verdict(tel, lm)
    assert not v["pass"] and any("invocation" in p for p in v["problems"]), v[
        "problems"
    ]


@METAL
def test_m7_cuts_full_and_partial_and_every_commit_probe_is_neutral(served):
    """M7: concrete emission-cut categories (a fully accepted block and a partial one) and a neutral probe at every
    commit, on the served MoE stack."""
    from test_mtp_exactness import scripted

    lm, _ = served
    plain, _ = turn(
        fake_model(lm), PROMPT, PromptCacheState(), n=16, seed=5, params=PARAMS
    )
    plain = list(plain)
    drafter = served_moe_drafter(lm)
    scripted(  # in place: follows the plain path except every 5th position — full accepts and rejections
        drafter,
        lambda p: (
            (plain[p] if p % 5 else (plain[p] + 1) % 64) if p < len(plain) else None
        ),
        high=True,
        vocab=64,
    )
    with observe.Recorder(mode="on", lm=lm, probe_token=5) as rec:
        out, _ = turn(
            fake_model(lm),
            PROMPT,
            PromptCacheState(),
            drafter=drafter,
            n=16,
            seed=5,
            params=PARAMS,
        )
    assert list(out) == plain
    assert any(
        c["n_emit"] == c["rows"] for c in rec.cuts
    ), rec.cuts  # a fully accepted block (+ bonus)
    assert any(
        0 < c["n_emit"] < c["rows"] for c in rec.cuts
    ), rec.cuts  # a rejection mid-block
    assert rec.commits and all(
        c["probe"] is not None and c["neutral"] for c in rec.commits
    )
    # code round 2 M7: the prompt-end snapshot exists, carries a probe and is neutral; every emitted token after the
    # prefill's first is accounted for by exactly one emission cut
    end = rec.prompt_end
    assert end is not None and end["kind"] == "prompt_end", end
    assert end["probe"] is not None and end["neutral"] is True, end
    assert 1 + sum(c["n_emit"] for c in rec.cuts) == len(plain), rec.cuts


@METAL
@pytest.mark.parametrize("part", ["router", "shared"])
def test_m7_omitting_the_router_or_the_shared_expert_inside_an_observed_block_fails(
    monkeypatch, part
):
    """M7: the MoE-block observer stays; only one part runs unobserved (a plain function standing in)."""
    lm = served_moe_target()
    drafter = served_moe_drafter(lm)
    mlp = drafter.layers[0].mlp
    if part == "router":
        real = mlp.gate
        monkeypatch.setattr(
            mlp,
            "gate",
            lambda x: mx.zeros((*x.shape[:-1], EXPERTS), dtype=x.dtype)
            + 0 * real.scales.sum(),
        )
    else:
        monkeypatch.setattr(mlp, "shared_expert", lambda x: mx.zeros_like(x))
    tel, _, _ = _observed_turn(lm, drafter)
    d = telemetry.draft_verdict(tel, drafter)
    assert (
        not d["pass"] and d["invocations"] and any("ran" in p for p in d["problems"])
    ), d


def _moe_prefilled(lm, prompt=PROMPT):
    cache = lm.make_cache()
    lm(mx.array([prompt], dtype=mx.int32), cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def _capture_routed(monkeypatch):
    from mlx_vlm.models import switch_layers
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv

    seen = dict(block=[], plain=[], block_out=[], plain_out=[])
    real_glu = sv.Qwen3_5BatchInvariantForward._switch_glu
    real_switch = switch_layers.SwitchGLU.__call__

    def glu(self, switch_mlp, x, indices):
        out = real_glu(self, switch_mlp, x, indices)
        seen["block"].append(indices)
        seen["block_out"].append(out)
        return out

    def switch(self, x, indices, *a, **k):
        out = real_switch(self, x, indices, *a, **k)
        seen["plain"].append(indices)
        seen["plain_out"].append(out)
        return out

    monkeypatch.setattr(sv.Qwen3_5BatchInvariantForward, "_switch_glu", glu)
    monkeypatch.setattr(switch_layers.SwitchGLU, "__call__", switch)
    return seen


@METAL
@pytest.mark.parametrize("width", [2, 3])
def test_v3d_m8_a_served_moe_verify_block_equals_sequential_singletons(
    served, monkeypatch, width
):
    """V3 (d) / M8 — the MoE analogue of P400, measured not assumed, on the SERVED shape (bf16, every routed bank
    QuantizedSwitchLinear int4 / g64): one width-``width`` ``speculative_verify`` call vs sequential singletons on an
    INDEPENDENT, identically prefilled cache — logits rows, each layer's selected experts and routed outputs, and every
    cache leaf (count, type, offset, keys, arrays — both directions) bitwise."""
    from mlx_vlm.speculative.cache_state import iter_leaf_caches

    lm, _ = served
    banks = [m for _, m in lm.named_modules() if "SwitchLinear" in type(m).__name__]
    assert banks and all(
        type(b).__name__ == "QuantizedSwitchLinear"
        and (b.bits, b.group_size) == (4, 64)
        for b in banks
    )
    block_cache, plain_cache = _moe_prefilled(lm), _moe_prefilled(lm)
    seen = _capture_routed(monkeypatch)
    tokens = mx.array([[5, 9, 13][:width]], dtype=mx.int32)
    verify = lm(tokens, cache=block_cache, speculative_verify=True)
    rows = [
        lm(tokens[:, i : i + 1], cache=plain_cache).logits[:, -1, :]
        for i in range(width)
    ]
    mx.eval(verify.logits, rows, *[a for v in seen.values() for a in v])
    for i in range(width):
        assert mx.array_equal(verify.logits[:, i, :], rows[i]).item(), f"logits row {i}"
    assert (
        len(seen["block"]) == SERVED_LAYERS
        and len(seen["plain"]) == SERVED_LAYERS * width
    )
    for k in range(SERVED_LAYERS):
        for j in range(width):
            p = j * SERVED_LAYERS + k
            got_idx, want_idx = (
                seen["block"][k][0, j],
                seen["plain"][p].reshape(-1, TOP_K)[0],
            )
            og, ow = mx.argsort(got_idx), mx.argsort(want_idx)
            assert mx.array_equal(got_idx[og], want_idx[ow]).item(), (k, j)
            got = seen["block_out"][k][0, j][og]
            want = seen["plain_out"][p].reshape(
                -1, TOP_K, seen["plain_out"][p].shape[-1]
            )[0][ow]
            assert mx.array_equal(
                got, want
            ).item(), f"layer {k} token {j} routed output"
    a_leaves, b_leaves = list(iter_leaf_caches(block_cache)), list(
        iter_leaf_caches(plain_cache)
    )
    assert len(a_leaves) == len(b_leaves)
    for a, b in zip(a_leaves, b_leaves, strict=True):
        assert type(a) is type(b) and getattr(a, "offset", None) == getattr(
            b, "offset", None
        )
        fa, fb = tree_flatten(a.state), tree_flatten(b.state)
        assert [k for k, _ in fa] == [k for k, _ in fb]
        for (key, x), (_, y) in zip(fa, fb, strict=True):
            if isinstance(x, mx.array):
                assert x.shape == y.shape and mx.array_equal(x, y).item(), (
                    type(a).__name__,
                    key,
                )


# ------------------------------------------------------------------ P369 v13 code round 2 M3 / M5 / N2 / N3 / N4
SERVED_DRAFTER_SHAPE = dict(
    layer="Qwen3_5MoeDecoderLayer",
    depth=1,
    experts=(EXPERTS, TOP_K),
    quant=[(4, 64)],
    text_type="qwen3_5_moe_text",
    block_size=3,
    banks=("QuantizedSwitchLinear", 4, 64),
)


def test_m3r2_a_freshly_loaded_drafter_passes_the_startup_gate():
    """Code round 2 M3 / P412: a drafter as loaded is UNBOUND (``reset`` binds it at the first request); the worker's
    startup gate binds it to the served target before reading its identity, and the certified shape arrives as JSON.
    """
    from p369_phase2 import worker

    lm = served_moe_target()
    drafter = served_moe_drafter(lm)
    assert drafter._input_embed is None and drafter._lm_head_fn is None  # as loaded
    shape = json.loads(json.dumps(SERVED_DRAFTER_SHAPE))
    ident, problems = worker.check_loaded_drafter(lm, drafter, shape)
    assert problems == [], problems
    assert ident["embed"] == id(lm.model.embed_tokens)
    other = served_moe_target(seed=3)
    assert worker.drafter_problems(ident, worker.target_identity(other), shape)


def test_m5r2_a_nested_successful_helper_is_attributed_once():
    """Code round 2 M5: ``exact_quantized_moe_hc_expand`` calls the (wrapped) ``exact_quantized_selected_linear``; the
    outer successful helper alone records its bank."""
    tel = telemetry.KernelTelemetry(tiny_moe_target())

    class Bank:
        pass

    bank = Bank()
    tel._names[id(bank)] = "probe.switch_mlp.down_proj"
    inner = tel._routed_helper(lambda module, x, indices: "inner", None)
    outer = tel._routed_helper(
        lambda module, x, indices: inner(module, x, indices), None
    )
    assert outer(bank, None, mx.array([[[0, 1]]], dtype=mx.int32)) == "inner"
    routed = [a for a in tel.attributions if a["kind"] == "routed"]
    assert [a["module"] for a in routed] == ["probe.switch_mlp.down_proj"], routed


@METAL
def test_m5r2_the_fused_affine_gate_up_branch_attributes_each_bank_once(monkeypatch):
    """Code round 2 M5: width 512 so ``exact_affine_switch_gate_up`` TAKES its fused branch (asserted), and the exact
    per-bank attribution — each of gate / up / down once, each with the quantized bank family.
    """
    import collections

    import mlx.nn as nn

    from mlx_vlm.models import fast_ops, switch_layers

    mx.random.seed(4)
    glu = switch_layers.SwitchGLU(512, 512, EXPERTS)
    glu.set_dtype(mx.bfloat16)
    nn.quantize(glu, group_size=64, bits=4, mode="affine")
    glu.eval()  # the exact fused path is inference-only (``not self.training``)
    mx.eval(glu.parameters())
    real, took = fast_ops.exact_affine_switch_gate_up, []

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        took.append(out is not None)
        return out

    monkeypatch.setattr(
        fast_ops, "exact_affine_switch_gate_up", spy
    )  # before the telemetry: it wraps the spy
    tel = telemetry.KernelTelemetry(tiny_moe_target())
    tel._names[id(glu)] = "probe.switch_mlp"
    for p in ("gate_proj", "up_proj", "down_proj"):
        tel._names[id(getattr(glu, p))] = f"probe.switch_mlp.{p}"
    x = mx.random.normal((1, 2, 512)).astype(mx.bfloat16)
    indices = mx.array([[[0, 1], [2, 3]]], dtype=mx.int32)
    with tel:
        mx.eval(glu(x, indices))
    assert took == [True]
    routed = collections.Counter(
        (a["module"], a["family"]) for a in tel.attributions if a["kind"] == "routed"
    )
    assert routed == collections.Counter(
        {
            (f"probe.switch_mlp.{p}", "routed_gather_quantized"): 1
            for p in ("gate_proj", "up_proj", "down_proj")
        }
    ), routed


@METAL
def test_n2_a_sabotaged_turn_leaves_no_telemetry_wrapper_behind(served):
    """Code round 2 N2: a sabotage applied inside the telemetry context is undone before the telemetry restores its
    own patches — pytest's later undo must not reinstall a stale telemetry wrapper."""
    from mlx_vlm.models import switch_layers

    classes = (switch_layers.SwitchLinear, switch_layers.QuantizedSwitchLinear)
    real = {c: c.__call__ for c in classes}
    lm, drafter = served

    def unhook(mp):
        for cls, fn in real.items():
            mp.setattr(cls, "__call__", fn)

    mp = pytest.MonkeyPatch()
    _observed_turn(lm, drafter, sabotage=unhook, monkeypatch=mp, n=4)
    mp.undo()  # what pytest's teardown does
    assert all(cls.__call__ is real[cls] for cls in classes)


@METAL
def test_n3_a_routed_invocation_running_another_layers_bank_fails(served):
    """Code round 2 N3: swapping the gate banks of the two layers keeps every projection name in every frame and every
    bank observed globally — the verdict must still refuse a frame running a bank its container does not own.
    """
    lm, drafter = served
    s0, s1 = (layer.mlp.switch_mlp for layer in lm.model.layers)

    def swap(mp):
        g0, g1 = s0.gate_proj, s1.gate_proj
        mp.setattr(s0, "gate_proj", g1)
        mp.setattr(s1, "gate_proj", g0)

    with pytest.MonkeyPatch.context() as mp:
        tel, _, _ = _observed_turn(lm, drafter, sabotage=swap, monkeypatch=mp)
    v = _verdict(tel, lm)
    assert not v["pass"] and any("owner" in p for p in v["problems"]), v["problems"]


def test_n4_the_served_fixture_quantizes_every_eligible_linear():
    """Code round 2 N4: the served-shape fixtures hold no dense linear / embedding / routed bank — every one affine
    int4 / g64 with bf16 scales and biases."""
    import mlx.nn as nn

    from mlx_vlm.models import switch_layers

    lm = served_moe_target()
    for root in (lm, served_moe_drafter(lm)):
        modules = list(root.named_modules())
        dense = [
            n
            for n, m in modules
            if isinstance(m, (nn.Linear, nn.Embedding))
            or type(m) is switch_layers.SwitchLinear
        ]
        assert not dense, dense
        quantized = [(n, m) for n, m in modules if hasattr(m, "bits")]
        assert quantized
        for n, m in quantized:
            assert (m.bits, m.group_size, m.mode) == (4, 64, "affine"), n
            assert m.scales.dtype == mx.bfloat16 and m.biases.dtype == mx.bfloat16, n


def test_p413_a_bound_drafter_never_renames_the_targets_modules():
    """P413: a BOUND drafter holds the target's embedding (and head) as attributes; the telemetry's module names stay
    the target's own (``drafter._input_embed`` / ``drafter._lm_head_fn`` would be unknown to the target manifest).
    """
    lm = served_moe_target()
    drafter = served_moe_drafter(lm).bind(lm)
    tel = telemetry.KernelTelemetry(lm, drafter=drafter)
    targets = {id(m): n for n, m in lm.named_modules()}
    assert all(tel._names[i] == n for i, n in targets.items())
    assert tel._names[id(lm.model.embed_tokens)] == "model.embed_tokens"
    own = {
        id(m)
        for n, m in drafter.named_modules()
        if not n.startswith(("_input_embed", "_lm_head_fn"))
    }
    assert all(tel._names[i].startswith("drafter.") for i in own - set(targets))


# ------------------------------------------------------------------ P369 v13 code round 3 R1 / R2
BIND_FIELDS = ("_input_embed", "_input_embed_scale", "_lm_head_fn", "_greedy_argmax_fn")


def test_r1_the_startup_gate_leaves_the_loaded_drafter_unbound():
    """Code round 3 R1: the boot check binds a COPY of the binding state only — the served drafter reaches its first
    request exactly as loaded, so only production's request-time ``reset`` can bind it.
    """
    from p369_phase2 import worker

    lm = served_moe_target()
    drafter = served_moe_drafter(lm)
    before = {k: getattr(drafter, k) for k in BIND_FIELDS}
    keys = (set(drafter), set(vars(drafter)))
    _, problems = worker.check_loaded_drafter(
        lm, drafter, json.loads(json.dumps(SERVED_DRAFTER_SHAPE))
    )
    assert problems == []
    assert {k: getattr(drafter, k) for k in BIND_FIELDS} == before
    assert (set(drafter), set(vars(drafter))) == keys


def test_r1_every_request_time_reset_is_checked_live_without_rebinding():
    """Code round 3 R1: after production's own ``reset`` binds the drafter at a request's start, the worker checks the
    LIVE identity (no harness bind) and refuses a reset that bound anything but the served target.
    """
    from p369_phase2 import worker

    lm = served_moe_target()
    drafter = served_moe_drafter(lm)
    shape = json.loads(json.dumps(SERVED_DRAFTER_SHAPE))
    checks = []
    undo = worker.watch_drafter_resets(lm, drafter, shape, checks)
    try:
        drafter.reset(lm)
        assert checks == [[]]
        with pytest.raises(RuntimeError, match="served drafter mismatch"):
            drafter.reset(served_moe_target(seed=3))
    finally:
        undo()
    assert "reset" not in vars(drafter)  # the class method again


@METAL
def test_r2_a_drafter_running_a_foreign_routed_bank_fails_the_draft_verdict(served):
    """Code round 3 R2 (direct bank path): the drafter's gate bank replaced by the TARGET's compatible bank — every
    part name and selected experts are still recorded, so only ownership can refuse it.
    """
    lm, drafter = served
    switch = drafter.layers[0].mlp.switch_mlp
    foreign = lm.model.layers[0].mlp.switch_mlp.gate_proj

    def swap(mp):
        mp.setattr(switch, "gate_proj", foreign)

    with pytest.MonkeyPatch.context() as mp:
        tel, _, _ = _observed_turn(lm, drafter, sabotage=swap, monkeypatch=mp)
    d = telemetry.draft_verdict(tel, drafter)
    assert not d["pass"] and d["invocations"], d
    assert any("model.layers.0.mlp.switch_mlp.gate_proj" in p for p in d["problems"]), d


@METAL
def test_r2_a_fused_helper_resolves_each_bank_by_identity(monkeypatch):
    """Code round 3 R2 (fused path): ``exact_affine_switch_gate_up`` stands for its container's gate / up banks; each
    is named by ITS module identity, so a foreign gate bank inside a drafter MoE frame is recorded as foreign.
    """
    import mlx.nn as nn

    from mlx_vlm.models import switch_layers

    mx.random.seed(4)

    def glu_():
        g = switch_layers.SwitchGLU(512, 512, EXPERTS)
        g.set_dtype(mx.bfloat16)
        nn.quantize(g, group_size=64, bits=4, mode="affine")
        g.eval()
        mx.eval(g.parameters())
        return g

    glu, other = glu_(), glu_()
    monkeypatch.setattr(glu, "gate_proj", other.gate_proj)
    tel = telemetry.KernelTelemetry(tiny_moe_target())
    block = object()
    tel._names.update(
        {
            id(block): "drafter.layers.0.mlp",
            id(glu): "drafter.layers.0.mlp.switch_mlp",
            id(glu.up_proj): "drafter.layers.0.mlp.switch_mlp.up_proj",
            id(glu.down_proj): "drafter.layers.0.mlp.switch_mlp.down_proj",
            id(other.gate_proj): "model.layers.0.mlp.switch_mlp.gate_proj",
        }
    )
    x = mx.random.normal((1, 2, 512)).astype(mx.bfloat16)
    indices = mx.array([[[0, 1], [2, 3]]], dtype=mx.int32)
    with tel:
        tel._draft_depth += 1
        try:
            run = tel._draft_moe_block(lambda b, x_: glu(x_, indices))
            mx.eval(run(block, x))
        finally:
            tel._draft_depth -= 1
    (frame,) = tel.draft_moe
    assert "switch_mlp.gate_proj" not in frame["parts"], frame
    assert any(
        "model.layers.0.mlp.switch_mlp.gate_proj" in p for p in frame["parts"]
    ), frame


# ------------------------------------------------------------------ P417 (smoke 16): a bound drafter's quantization is its own
def test_p417_a_bound_drafter_never_reports_the_targets_quantization():
    """P417: a BOUND drafter holds the target's embedding / head as module children; its identity's quantization set
    must be the drafter's own modules only (an 8-bit target embedding is not the drafter's).
    """
    import mlx.nn as nn

    from p369_phase2 import worker

    lm = served_moe_target()
    emb = lm.model.embed_tokens
    dense = nn.Embedding(emb.weight.shape[0], 512)
    lm.model.embed_tokens = nn.QuantizedEmbedding.from_embedding(
        dense, group_size=64, bits=8
    )
    drafter = served_moe_drafter(lm)
    drafter.bind(lm)
    assert worker.drafter_identity(drafter)["quant"] == [[4, 64]]
    _, problems = worker.check_loaded_drafter(
        lm, served_moe_drafter(lm), json.loads(json.dumps(SERVED_DRAFTER_SHAPE))
    )
    assert problems == [], problems
