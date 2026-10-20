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
