"""P369 v13 (V12 / V14): Ornith-1.0-35B-mlx-uniform-4bit's drafter construction path on tiny models (fork-only, CPU).

Ornith's MTP head was TRANSPLANTED: split from a Qwen3-Next-style source (`Qwen3NextMTPSplitter`) into a `qwen3_5_mtp`
drafter. The splitter copies the source's nested `text_config` verbatim, while `Qwen3_5MTPDraftModel` builds MoE decoder
layers only when that nested `model_type` contains "moe". These tests pin both shapes: the uploaded drafter's (nested
`qwen3_5_moe_text`) must load MoE layers, and an adversarial nested `qwen3_next` source must never yield a dense-runtime
drafter holding expert weights (it must be rewritten to the MoE type or fail).
"""

import json
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
    lm = tiny_moe_target()
    manifest = telemetry.manifest_from_model(lm)
    routed = {n: i for n, i in manifest.items() if i["kind"] == "routed"}
    want = {
        f"model.layers.{i}.mlp.switch_mlp.{p}"
        for i in range(LAYERS)
        for p in ("gate_proj", "up_proj", "down_proj")
    }
    assert set(routed) == want
    assert all(
        i["on_path"] and i["expected"] == dict(routed=["routed_gather"])
        for i in routed.values()
    )


def _observed_turn(lm, drafter, *, sabotage=None, monkeypatch=None):
    tel = telemetry.KernelTelemetry(lm, drafter=drafter)
    keys = observe.KeyLengths()  # the verifier block boundaries
    with tel, keys, observe.Recorder(mode="on", lm=lm, probe_token=5) as rec:
        if sabotage:
            sabotage(monkeypatch)
        out, _ = turn(
            fake_model(lm),
            PROMPT,
            PromptCacheState(),
            drafter=drafter,
            n=12,
            seed=5,
            params=PARAMS,
        )
    rec.blocks = keys.blocks
    return tel, rec, list(out)


def test_v3b_v3c_v15_one_mtp_request_is_fully_observed():
    """(b) recorder rows, emission cuts, commit snapshots and verifier blocks on the MoE stack; (c) the router, shared
    expert and each routed bank projection exactly once per verify-scope invocation; V15 the drafter's own MoE path —
    router, shared expert and routed gate / up / down — in its draft scope, with the selected experts recorded.
    """
    lm = tiny_moe_target()
    drafter = tiny_moe_drafter(lm)
    tel, rec, out = _observed_turn(lm, drafter)
    assert len(out) == 12 and rec.rows and rec.commits and rec.blocks
    v = telemetry.verdict(tel, telemetry.manifest_from_model(lm))
    assert v["pass"], v["problems"]
    invocations = telemetry.routed_invocations(tel, scope="verify")
    assert invocations and all(
        sorted(i["projections"]) == ["down_proj", "gate_proj", "up_proj"]
        for i in invocations
    )
    assert any(
        i["width"] > 1 for i in invocations
    )  # a real verify block, not only singletons
    d = telemetry.draft_verdict(tel, drafter)
    assert d["pass"], d["problems"]
    assert d["invocations"] and all(i["selected_experts"] for i in d["invocations"])


def test_v3e_a_verifier_bypassing_a_routed_projection_fails_the_verdict(monkeypatch):
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv

    def no_gate(
        self, switch_mlp, x, indices
    ):  # sabotage: the routed gate bank never runs
        flat = x.reshape(-1, x.shape[-1])
        flat = mx.expand_dims(flat, (-2, -3))
        idx = indices.reshape(-1, indices.shape[-1])
        up = switch_mlp.up_proj(flat, idx, sorted_indices=False)
        out = switch_mlp.down_proj(up, idx, sorted_indices=False)
        return out.squeeze(-2).reshape(*x.shape[:2], indices.shape[-1], -1)

    lm = tiny_moe_target()
    drafter = tiny_moe_drafter(lm)
    tel, _, _ = _observed_turn(
        lm,
        drafter,
        sabotage=lambda mp: mp.setattr(
            sv.Qwen3_5BatchInvariantForward, "_switch_glu", no_gate
        ),
        monkeypatch=monkeypatch,
    )
    assert not telemetry.verdict(tel, telemetry.manifest_from_model(lm))["pass"]


def test_v15_a_drafter_skipping_its_routed_experts_fails_the_draft_verdict(monkeypatch):
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    def shared_only(self, x):  # sabotage: only the shared expert runs
        return self.shared_expert(x) * mx.sigmoid(self.shared_expert_gate(x))

    lm = tiny_moe_target()
    drafter = tiny_moe_drafter(lm)
    tel, _, _ = _observed_turn(
        lm,
        drafter,
        sabotage=lambda mp: mp.setattr(
            Qwen3_5MoeSparseMoeBlock, "__call__", shared_only
        ),
        monkeypatch=monkeypatch,
    )
    assert not telemetry.draft_verdict(tel, drafter)["pass"]


def test_v3e_an_absent_routed_hook_fails_the_verdict(monkeypatch):
    """Disabling the new routed hook (the projections run unobserved) must fail coverage, never pass."""
    from mlx_vlm.models import switch_layers

    real = {
        c: c.__call__
        for c in (switch_layers.SwitchLinear, switch_layers.QuantizedSwitchLinear)
    }
    lm = tiny_moe_target()
    drafter = tiny_moe_drafter(lm)

    def unhook(mp):
        for cls, fn in real.items():
            mp.setattr(cls, "__call__", fn)

    tel, _, _ = _observed_turn(lm, drafter, sabotage=unhook, monkeypatch=monkeypatch)
    assert not telemetry.verdict(tel, telemetry.manifest_from_model(lm))["pass"]
    assert not telemetry.draft_verdict(tel, drafter)["pass"]


# ------------------------------------------------------------------ V3 (d): the MoE analogue of P400, measured
def _moe_prefilled(lm, prompt=PROMPT):
    cache = lm.make_cache()
    lm(mx.array([prompt], dtype=mx.int32), cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def _capture_experts(monkeypatch):
    """Per-call selected experts: the verifier's ``_switch_glu`` (block) and the plain MoE block's routing (singletons)."""
    from mlx_vlm.models import switch_layers
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv

    seen = dict(block=[], plain=[])
    real_glu = sv.Qwen3_5BatchInvariantForward._switch_glu
    real_switch = switch_layers.SwitchGLU.__call__

    def glu(self, switch_mlp, x, indices):
        seen["block"].append(indices)
        return real_glu(self, switch_mlp, x, indices)

    def switch(self, x, indices, *a, **k):
        seen["plain"].append(indices)
        return real_switch(self, x, indices, *a, **k)

    monkeypatch.setattr(sv.Qwen3_5BatchInvariantForward, "_switch_glu", glu)
    monkeypatch.setattr(switch_layers.SwitchGLU, "__call__", switch)
    return seen


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="the served verifier kernels are Metal"
)
@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("quantized", [False, True], ids=["dense", "int4"])
def test_v3d_a_moe_verify_block_equals_sequential_singletons(
    monkeypatch, width, quantized
):
    """One width-``width`` ``speculative_verify`` call vs ``width`` plain singleton calls on an INDEPENDENT, identically
    prefilled cache, on the MoE stack (routed banks dispatched for the whole block at once — the MoE analogue of P400):
    logits rows, every full-attention KV and recurrent state, and each layer's selected experts, bitwise.
    """
    import mlx.nn as nn

    lm = tiny_moe_target()
    if quantized:  # the served Ornith banks are QuantizedSwitchLinear (int4)
        nn.quantize(lm, group_size=32, bits=4)
        mx.eval(lm.parameters())
        assert any(
            type(m).__name__ == "QuantizedSwitchLinear" for _, m in lm.named_modules()
        )
    block_cache, plain_cache = _moe_prefilled(lm), _moe_prefilled(lm)
    seen = _capture_experts(monkeypatch)
    tokens = mx.array([[5, 9, 13][:width]], dtype=mx.int32)
    verify = lm(tokens, cache=block_cache, speculative_verify=True)
    rows = [
        lm(tokens[:, i : i + 1], cache=plain_cache).logits[:, -1, :]
        for i in range(width)
    ]
    mx.eval(verify.logits, rows, seen["block"], seen["plain"])
    for i in range(width):
        assert mx.array_equal(verify.logits[:, i, :], rows[i]).item(), f"logits row {i}"
    # experts: block call k covers tokens 0..width-1 of layer k; singleton j of layer k is plain call j * LAYERS + k
    assert len(seen["block"]) == LAYERS and len(seen["plain"]) == LAYERS * width
    for k, block_idx in enumerate(seen["block"]):
        for j in range(width):
            plain_idx = seen["plain"][j * LAYERS + k]
            got = mx.sort(block_idx[0, j], axis=-1)
            want = mx.sort(plain_idx.reshape(-1, plain_idx.shape[-1])[0], axis=-1)
            assert mx.array_equal(got, want).item(), f"layer {k} token {j} experts"
    from mlx_vlm.speculative.cache_state import iter_leaf_caches

    compared = 0
    for a, b in zip(iter_leaf_caches(block_cache), iter_leaf_caches(plain_cache)):
        fa, fb = tree_flatten(a.state), tree_flatten(b.state)
        for (key, x), (_, y) in zip(fa, fb):
            if isinstance(x, mx.array):
                assert x.shape == y.shape and mx.array_equal(x, y).item(), (
                    type(a).__name__,
                    key,
                )
                compared += 1
    assert compared > 0
