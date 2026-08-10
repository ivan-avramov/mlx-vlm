"""Tests for qwen3_5_moe.Model.sanitize expert-weight layout handling.

The fork's SwitchGLU wants a FUSED 3D expert tensor [num_experts, out, in].
Checkpoints ship experts two ways:
  - FUSED (Qwen3.6-VL): `mlp.experts.gate_up_proj` + `mlp.experts.down_proj`
  - UNFUSED (Ornith / Qwen3-Next style): `mlp.experts.{e}.{gate,up,down}_proj.weight`
sanitize() must produce the same `switch_mlp.*` stacked weights from either layout.

sanitize only reads self.config.text_config.{num_hidden_layers, num_experts,
tie_word_embeddings}, so we duck-type `self` and call the method unbound — no need
to instantiate the (heavy) full multimodal Model.

This file is fork-only, and after 4a54a2be it is the *only* thing carrying the fork's
knowledge about these two layouts: `qwen3_5_moe.py` itself is now byte-identical to
upstream. The fork had its own unfused branch (`f0d50c90`) four days before upstream's
`b590c747` (#1472) added an equivalent one; the fork's differed only in probing
`experts.0.gate_proj.weight` where upstream probes `experts.0.up_proj.weight`, and
carrying it made the file a permanent conflict site for no behavioural gain. The
probes were shown identical on every layout that can exist — all three projections
present, none present, and the fused `gate_up_proj` tensor — and exact mirror images
on the two that cannot: each raises KeyError for the partial layout that omits its own
probe key and passes the other through unstacked. `test_a_partial_expert_layout_raises`
below pins that, so nobody reintroduces the fork probe to "fix" it.
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
    """Pins which projection upstream's probe keys on, and that it fails loudly.

    A SwiGLU expert without a gate projection is not SwiGLU, so this layout cannot
    ship — but WHICH partial layout raises is the whole (and only) difference between
    upstream's probe and the fork's, so it is worth a test rather than a comment. With
    `up_proj` as the probe, up+down raises inside the branch and gate+down skips it;
    the fork's `gate_proj` probe mirrored that exactly. Neither is safer, which is why
    the fork's was retired in favour of staying byte-identical to upstream.
    """
    E, H, I = 3, 8, 6
    prefix = "model.language_model.layers.0.mlp"
    w = {}
    for e in range(E):
        w[f"{prefix}.experts.{e}.up_proj.weight"] = mx.ones((I, H))
        w[f"{prefix}.experts.{e}.down_proj.weight"] = mx.ones((H, I))

    with pytest.raises(KeyError, match="gate_proj"):
        Model.sanitize(_fake_self(1, E), dict(w))
