"""Tests for qwen3_5_moe.Model.sanitize expert-weight layout handling.

The fork's SwitchGLU wants a FUSED 3D expert tensor [num_experts, out, in].
Checkpoints ship experts two ways:
  - FUSED (Qwen3.6-VL): `mlp.experts.gate_up_proj` + `mlp.experts.down_proj`
  - UNFUSED (Ornith / Qwen3-Next style): `mlp.experts.{e}.{gate,up,down}_proj.weight`
sanitize() must produce the same `switch_mlp.*` stacked weights from either layout.

sanitize only reads self.config.text_config.{num_hidden_layers, num_experts,
tie_word_embeddings}, so we duck-type `self` and call the method unbound — no need
to instantiate the (heavy) full multimodal Model.

The upstream packing implementation is retained. Required expert weights must
still fail loudly when missing; upstream's optional scales/biases handling must
not make a missing gate weight look like a loadable checkpoint.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.models.qwen3_5_moe.qwen3_5_moe import Model


def _fake_self(num_layers, num_experts):
    tc = SimpleNamespace(
        num_hidden_layers=num_layers,
        num_experts=num_experts,
        tie_word_embeddings=False,
    )
    return SimpleNamespace(config=SimpleNamespace(text_config=tc))


# After sanitize_key, `model.language_model.*` -> `language_model.model.*`.
def _sw(proj):
    return f"language_model.model.layers.0.mlp.switch_mlp.{proj}.weight"


def test_sanitize_stacks_unfused_experts():
    E, L, H, I = 4, 1, 8, 6
    w = {}
    for e in range(E):
        # gate/up: [intermediate, hidden]; down: [hidden, intermediate] (nn.Linear weight = [out, in])
        w[f"model.language_model.layers.0.mlp.experts.{e}.gate_proj.weight"] = mx.full(
            (I, H), float(e)
        )
        w[f"model.language_model.layers.0.mlp.experts.{e}.up_proj.weight"] = mx.ones(
            (I, H)
        )
        w[f"model.language_model.layers.0.mlp.experts.{e}.down_proj.weight"] = mx.ones(
            (H, I)
        )

    out = Model.sanitize(_fake_self(L, E), dict(w))

    assert out[_sw("gate_proj")].shape == (E, I, H)
    assert out[_sw("up_proj")].shape == (E, I, H)
    assert out[_sw("down_proj")].shape == (E, H, I)
    # expert ordering preserved (gate of expert e was filled with value e)
    for e in range(E):
        assert float(out[_sw("gate_proj")][e, 0, 0]) == float(e)
    # the per-expert source keys are fully consumed
    assert not any(".experts." in k for k in out)


def test_sanitize_fused_experts_still_supported():
    # Regression: the original Qwen3.6-VL fused layout must keep working.
    E, H, I = 4, 8, 6
    w = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": mx.zeros(
            (E, 2 * I, H)
        ),
        "model.language_model.layers.0.mlp.experts.down_proj": mx.zeros((E, H, I)),
    }
    out = Model.sanitize(_fake_self(1, E), dict(w))
    assert out[_sw("gate_proj")].shape == (E, I, H)
    assert out[_sw("up_proj")].shape == (E, I, H)
    assert out[_sw("down_proj")].shape == (E, H, I)
    assert not any(".experts." in k for k in out)


def test_a_partial_expert_layout_raises():
    """Optional quantization suffixes must not hide a missing gate weight."""
    E, H, I = 3, 8, 6
    prefix = "model.language_model.layers.0.mlp"
    w = {}
    for e in range(E):
        w[f"{prefix}.experts.{e}.up_proj.weight"] = mx.ones((I, H))
        w[f"{prefix}.experts.{e}.down_proj.weight"] = mx.ones((H, I))

    with pytest.raises(KeyError, match="gate_proj"):
        Model.sanitize(_fake_self(1, E), dict(w))


@pytest.mark.parametrize("suffix", ["scales", "biases"])
def test_sanitize_stacks_optional_quantization_sidecars(suffix):
    prefix = "model.language_model.layers.0.mlp"
    weights = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for expert in range(3):
            weights[f"{prefix}.experts.{expert}.{projection}.weight"] = mx.ones((8, 8))
            weights[f"{prefix}.experts.{expert}.{projection}.{suffix}"] = mx.full(
                (8, 1), float(expert)
            )
    result = Model.sanitize(_fake_self(1, 3), weights)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        sidecar = result[_sw(projection).removesuffix("weight") + suffix]
        assert sidecar.shape == (3, 8, 1)
        assert sidecar[:, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert not any(".experts." in key for key in result)


def test_mtp_postprocess_uses_upstream_packing_without_shifting_native_norms():
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.split import _stack_separate_experts

    norm = mx.full((8,), 0.25)
    tensors = {"norm.weight": norm}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for expert in range(2):
            for suffix in ("weight", "scales", "biases"):
                tensors[f"layers.0.mlp.experts.{expert}.{projection}.{suffix}"] = (
                    mx.full((8, 8 if suffix == "weight" else 1), float(expert))
                )
    _stack_separate_experts(tensors, {"num_experts": 2})
    assert mx.array_equal(tensors["norm.weight"], norm).item()
    assert not any(".experts." in key for key in tensors)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("weight", "scales", "biases"):
            stacked = tensors[f"layers.0.mlp.switch_mlp.{projection}.{suffix}"]
            assert stacked[:, 0, 0].tolist() == [0.0, 1.0]


@pytest.mark.parametrize("already_stacked", [False, True])
def test_mtp_postprocess_rejects_incomplete_configured_expert_count(already_stacked):
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.split import _stack_separate_experts

    tensors = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        if already_stacked:
            tensors[f"layers.0.mlp.switch_mlp.{projection}.weight"] = mx.ones((2, 8, 8))
        else:
            for expert in range(2):
                tensors[f"layers.0.mlp.experts.{expert}.{projection}.weight"] = mx.ones(
                    (8, 8)
                )
    with pytest.raises(ValueError, match="expected 3 stacked experts"):
        _stack_separate_experts(tensors, {"num_experts": 3})
