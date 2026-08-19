"""nemotron_h_mtp: native-MTP drafter for the `nemotron_h` hybrid family.

Covers construction from the checkpoint's `mtp_layers_block_type` inventory
(observed on the BF16 base as `["attention", "moe"]`), target binding against
nemotron_h's `backbone.embeddings`/`lm_head` naming (which has no
`embed_tokens` anywhere the sibling drafters' `bind()` looks), the
draft/accept round trip, weight-name sanitization (including MoE expert
stacking), and the extraction+quantization tool -- all against tiny synthetic
configs/weights, never the real checkpoint.
"""

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.nemotron_h.config import ModelConfig as NemotronHConfig
from mlx_vlm.speculative.drafters import DRAFTER_KIND_BY_MODEL_TYPE
from mlx_vlm.speculative.drafters.nemotron_h_mtp import ModelConfig as NemotronHMTPConfig
from mlx_vlm.speculative.drafters.nemotron_h_mtp import NemotronHMTPDraftModel
from mlx_vlm.speculative.drafters.nemotron_h_mtp.split import split_nemotron_h_mtp

# mx.quantize only supports group sizes {32, 64, 128}, so every dim below is
# a multiple of 32 -- small enough to stay a fast unit test, large enough
# that the quantization test exercises a real group size (not the real
# deployment's 64, but the identical `_quantize` code path).
HIDDEN_SIZE = 32
VOCAB_SIZE = 32
N_ROUTED_EXPERTS = 4
MOE_INTERMEDIATE_SIZE = 32
SHARED_INTERMEDIATE_SIZE = 32


def _tiny_nemotron_h_text_config(**overrides) -> NemotronHConfig:
    kwargs = dict(
        model_type="nemotron_h",
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=32,
        num_hidden_layers=2,
        max_position_embeddings=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        attention_bias=False,
        mamba_num_heads=2,
        mamba_head_dim=8,
        mamba_proj_bias=False,
        ssm_state_size=8,
        conv_kernel=4,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=True,
        hybrid_override_pattern="*E",
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
        moe_shared_expert_intermediate_size=SHARED_INTERMEDIATE_SIZE,
        moe_latent_size=None,
        n_group=1,
        n_routed_experts=N_ROUTED_EXPERTS,
        n_shared_experts=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )
    kwargs.update(overrides)
    return NemotronHConfig(**kwargs)


def _drafter_config(**overrides) -> NemotronHMTPConfig:
    kwargs = dict(
        text_config=_tiny_nemotron_h_text_config(),
        block_size=2,
        mtp_block_types=["attention", "moe"],
    )
    kwargs.update(overrides)
    return NemotronHMTPConfig(**kwargs)


def _target_model():
    return SimpleNamespace(
        language_model=SimpleNamespace(
            backbone=SimpleNamespace(embeddings=nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)),
            lm_head=nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False),
        )
    )


class TestRegistration:
    def test_nemotron_h_mtp_maps_to_mtp_kind(self):
        assert DRAFTER_KIND_BY_MODEL_TYPE["nemotron_h_mtp"] == "mtp"


class TestConfig:
    def test_default_block_types_are_attention_then_moe(self):
        cfg = NemotronHMTPConfig(text_config=_tiny_nemotron_h_text_config())
        assert cfg.block_type_chars() == ["*", "E"]

    def test_runtime_block_size_defaults_to_native_nextn_depth(self):
        cfg = NemotronHMTPConfig.from_dict(
            {
                "model_type": "nemotron_h_mtp",
                "text_config": {
                    **_tiny_nemotron_h_text_config().to_dict(),
                    "num_nextn_predict_layers": 1,
                },
                "block_size": 5,
            }
        )
        assert cfg.block_size == 5
        assert cfg.runtime_block_size == 2

    def test_from_dict_picks_up_mtp_block_types_from_text_config(self):
        cfg = NemotronHMTPConfig.from_dict(
            {
                "model_type": "nemotron_h_mtp",
                "text_config": {
                    **_tiny_nemotron_h_text_config().to_dict(),
                    "mtp_layers_block_type": ["attention", "moe"],
                },
            }
        )
        assert cfg.mtp_block_types == ["attention", "moe"]
        assert cfg.block_type_chars() == ["*", "E"]


class TestConstruction:
    def test_mamba_block_type_is_rejected(self):
        with pytest.raises(NotImplementedError, match="mamba"):
            NemotronHMTPDraftModel(_drafter_config(mtp_block_types=["mamba", "moe"]))

    def test_missing_text_config_raises(self):
        with pytest.raises(ValueError, match="text_config must be set"):
            NemotronHMTPDraftModel(NemotronHMTPConfig(text_config=None))

    def test_cache_is_allocated_only_for_the_attention_sublayer(self):
        drafter = NemotronHMTPDraftModel(_drafter_config())
        assert len(drafter.make_cache()) == 1


class TestBind:
    def test_bind_finds_backbone_embeddings_and_lm_head(self):
        """nemotron_h has no `embed_tokens` anywhere in the chain the sibling
        drafters' `bind()` checks -- its embedding table is
        `language_model.backbone.embeddings`. This is the fallback branch
        `nemotron_h_mtp` adds on top of the shared sibling pattern."""
        drafter = NemotronHMTPDraftModel(_drafter_config())
        target = _target_model()

        drafter.bind(target)

        assert drafter._input_embed is target.language_model.backbone.embeddings
        assert drafter._lm_head_fn is target.language_model.lm_head

    def test_bind_raises_when_neither_shape_matches(self):
        drafter = NemotronHMTPDraftModel(_drafter_config())
        with pytest.raises(AttributeError, match="embed_tokens"):
            drafter.bind(SimpleNamespace())


class TestDraftBlock:
    def test_draft_block_smoke(self):
        drafter = NemotronHMTPDraftModel(_drafter_config(block_size=3))
        target = _target_model()
        drafter.reset(target)
        drafter.set_shared_kv({}, kv_offset=4, position=3, kv_valid_len=4)

        hidden = mx.zeros((1, 1, HIDDEN_SIZE), dtype=mx.float32)
        tokens = drafter.draft_block(
            7,
            hidden,
            None,
            3,
            lambda logits: mx.argmax(logits, axis=-1),
            mx.int32,
            greedy=True,
        )
        mx.eval(tokens)

        assert tokens.shape == (1, 2)
        assert drafter._round_appended == 2
        assert drafter._cache[0].offset == 2

    def test_prefill_from_target_hidden_seeds_next_draft_block(self):
        drafter = NemotronHMTPDraftModel(_drafter_config(block_size=3))
        target = _target_model()
        drafter.reset(target)

        input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        hidden = mx.zeros((1, 3, HIDDEN_SIZE), dtype=mx.float32)
        drafter.prefill_from_target_hidden(
            input_ids,
            hidden,
            bonus_token=9,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            greedy=True,
        )

        assert drafter._seed_token is not None
        assert drafter._seed_hidden is not None
        # The whole 3-token shifted sequence goes through the attention
        # sub-layer's KVCache in one forward, not incrementally.
        assert drafter._cache[0].offset == 3


class TestAcceptVerifiedTokensBatch:
    def test_uniform_acceptance_updates_cache_and_seed(self):
        drafter = NemotronHMTPDraftModel(_drafter_config(block_size=3))
        target = _target_model()
        drafter.reset(target)
        drafter.set_shared_kv({}, kv_offset=4, position=3, kv_valid_len=4)

        draft_tokens = drafter.draft_block(
            mx.array([7, 8], dtype=mx.int32),
            mx.zeros((2, 1, HIDDEN_SIZE), dtype=mx.float32),
            None,
            3,
            lambda logits: mx.argmax(logits, axis=-1),
            mx.int32,
            greedy=True,
        )
        verify_hidden = mx.zeros((2, 3, HIDDEN_SIZE), dtype=mx.float32)

        drafter.accept_verified_tokens_batch(
            verify_hidden,
            draft_tokens,
            accepted=[0, 0],
            new_tokens=[[3], [4]],
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            token_dtype=mx.int32,
            greedy=True,
        )
        mx.eval(drafter._seed_token)

        assert drafter._seed_token.shape == (2, 1)
        assert drafter._seed_hidden.shape == (2, 1, HIDDEN_SIZE)
        assert drafter._round_appended == 0
        assert drafter._cache[0].offset == 1

    def test_ragged_acceptance_raises(self):
        drafter = NemotronHMTPDraftModel(_drafter_config(block_size=3))
        target = _target_model()
        drafter.reset(target)
        drafter.set_shared_kv({}, kv_offset=4, position=3, kv_valid_len=4)

        draft_tokens = drafter.draft_block(
            mx.array([7, 8], dtype=mx.int32),
            mx.zeros((2, 1, HIDDEN_SIZE), dtype=mx.float32),
            None,
            3,
            lambda logits: mx.argmax(logits, axis=-1),
            mx.int32,
            greedy=True,
        )
        verify_hidden = mx.zeros((2, 3, HIDDEN_SIZE), dtype=mx.float32)

        with pytest.raises(ValueError, match="uniform acceptance"):
            drafter.accept_verified_tokens_batch(
                verify_hidden,
                draft_tokens,
                accepted=[0, 1],
                new_tokens=[[3], [4]],
                sampler=lambda logits: mx.argmax(logits, axis=-1),
                token_dtype=mx.int32,
                greedy=True,
            )


def _raw_mtp_weights(text_config: NemotronHConfig) -> dict:
    h = text_config.hidden_size
    head_dim = h // text_config.num_attention_heads
    q_dim = text_config.num_attention_heads * head_dim
    kv_dim = text_config.num_key_value_heads * head_dim
    n_experts = text_config.n_routed_experts
    moe_h = text_config.moe_intermediate_size
    shared_h = text_config.moe_shared_expert_intermediate_size

    weights = {
        "mtp.layers.0.eh_proj.weight": mx.zeros((h, 2 * h)),
        "mtp.layers.0.enorm.weight": mx.ones((h,)),
        "mtp.layers.0.hnorm.weight": mx.ones((h,)),
        "mtp.layers.0.norm.weight": mx.ones((h,)),
        "mtp.layers.0.mixer.q_proj.weight": mx.zeros((q_dim, h)),
        "mtp.layers.0.mixer.k_proj.weight": mx.zeros((kv_dim, h)),
        "mtp.layers.0.mixer.v_proj.weight": mx.zeros((kv_dim, h)),
        "mtp.layers.0.mixer.o_proj.weight": mx.zeros((h, q_dim)),
        "mtp.layers.1.norm.weight": mx.ones((h,)),
        "mtp.layers.1.mixer.gate.weight": mx.zeros((n_experts, h)),
        "mtp.layers.1.mixer.gate.e_score_correction_bias": mx.zeros((n_experts,)),
        "mtp.layers.1.mixer.shared_experts.up_proj.weight": mx.zeros((shared_h, h)),
        "mtp.layers.1.mixer.shared_experts.down_proj.weight": mx.zeros((h, shared_h)),
        "mtp.layers.1.final_layernorm.weight": mx.ones((h,)),
    }
    for e in range(n_experts):
        weights[f"mtp.layers.1.mixer.experts.{e}.up_proj.weight"] = mx.zeros((moe_h, h))
        weights[f"mtp.layers.1.mixer.experts.{e}.down_proj.weight"] = mx.zeros((h, moe_h))
    return weights


class TestSanitize:
    def test_maps_hf_layout_to_drafter_layout_and_stacks_experts(self):
        text_config = _tiny_nemotron_h_text_config()
        drafter = NemotronHMTPDraftModel(
            NemotronHMTPConfig(text_config=text_config, block_size=2)
        )

        out = drafter.sanitize(_raw_mtp_weights(text_config))

        assert not any(key.startswith("mtp.") for key in out)
        assert "eh_proj.weight" in out
        assert "enorm.weight" in out
        assert "hnorm.weight" in out
        assert "final_layernorm.weight" in out
        assert "layers.0.norm.weight" in out
        assert "layers.0.mixer.q_proj.weight" in out
        assert "layers.1.mixer.gate.weight" in out
        assert "layers.1.mixer.switch_mlp.fc1.weight" in out
        assert "layers.1.mixer.switch_mlp.fc2.weight" in out
        assert out["layers.1.mixer.switch_mlp.fc1.weight"].shape == (
            N_ROUTED_EXPERTS,
            MOE_INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        )
        assert out["layers.1.mixer.switch_mlp.fc2.weight"].shape == (
            N_ROUTED_EXPERTS,
            HIDDEN_SIZE,
            MOE_INTERMEDIATE_SIZE,
        )
        assert not any(".experts." in key for key in out)

    def test_sanitize_is_idempotent_on_already_mapped_weights(self):
        text_config = _tiny_nemotron_h_text_config()
        drafter = NemotronHMTPDraftModel(
            NemotronHMTPConfig(text_config=text_config, block_size=2)
        )
        once = drafter.sanitize(_raw_mtp_weights(text_config))

        twice = drafter.sanitize(once)

        assert set(once.keys()) == set(twice.keys())
        for key in once:
            assert once[key].shape == twice[key].shape

    def test_missing_expert_raises(self):
        text_config = _tiny_nemotron_h_text_config()
        drafter = NemotronHMTPDraftModel(
            NemotronHMTPConfig(text_config=text_config, block_size=2)
        )
        weights = _raw_mtp_weights(text_config)
        del weights[
            f"mtp.layers.1.mixer.experts.{text_config.n_routed_experts - 1}.up_proj.weight"
        ]

        with pytest.raises(ValueError, match="missing expert tensors"):
            drafter.sanitize(weights)


class TestSplit:
    def _write_source(self, tmp_path, text_config: NemotronHConfig):
        source = tmp_path / "source"
        source.mkdir()
        config = {
            "model_type": "nemotron_h",
            "num_nextn_predict_layers": 1,
            "mtp_layers_block_type": ["attention", "moe"],
            **text_config.to_dict(),
        }
        (source / "config.json").write_text(json.dumps(config))

        weights = _raw_mtp_weights(text_config)
        # A non-mtp tensor in the same shard must never leak into the sidecar.
        weights["backbone.layers.0.norm.weight"] = mx.ones((text_config.hidden_size,))
        mx.save_safetensors(
            str(source / "model-00001-of-00001.safetensors"), weights, metadata={}
        )
        (source / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        key: "model-00001-of-00001.safetensors" for key in weights
                    }
                }
            )
        )
        return source

    def test_writes_bf16_sidecar_with_discovery_metadata(self, tmp_path):
        text_config = _tiny_nemotron_h_text_config()
        source = self._write_source(tmp_path, text_config)
        output = tmp_path / "mtp"

        split_nemotron_h_mtp(str(source), str(output))

        with open(output / "config.json") as f:
            cfg = json.load(f)
        assert cfg["model_type"] == "nemotron_h_mtp"
        assert cfg["mtp_file"] == "mtp.safetensors"
        assert cfg["block_size"] == 2
        assert cfg["mtp_block_types"] == ["attention", "moe"]
        assert "quantization" not in cfg

        weights = mx.load(str(output / "mtp.safetensors"))
        assert "eh_proj.weight" in weights
        assert "layers.1.mixer.switch_mlp.fc1.weight" in weights
        assert not any(key.startswith("backbone.") for key in weights)
        assert not any(key.startswith("mtp.") for key in weights)

    def test_quantizes_to_int4_when_requested(self, tmp_path):
        # `mx.quantize` only supports group sizes {32, 64, 128}. The real
        # deployment target is 64 (the base checkpoint's hidden_size=2688,
        # moe_intermediate_size=1856 etc. are all multiples of 64); every
        # tiny dim here is a multiple of 32 instead, so group_size=32
        # exercises the identical `_quantize` code path at unit-test scale.
        text_config = _tiny_nemotron_h_text_config()
        source = self._write_source(tmp_path, text_config)
        output = tmp_path / "mtp-4bit"

        split_nemotron_h_mtp(str(source), str(output), q_bits=4, q_group_size=32)

        with open(output / "config.json") as f:
            cfg = json.load(f)
        assert cfg["quantization"] == {"group_size": 32, "bits": 4, "mode": "affine"}
        assert cfg["quantization_config"] == cfg["quantization"]

        weights = mx.load(str(output / "mtp.safetensors"))
        # The router gate stays full precision for routing stability.
        assert "layers.1.mixer.gate.weight" in weights
        assert "layers.1.mixer.gate.scales" not in weights
        # Everything else 2D+ gets scales/biases alongside the packed weight.
        assert "eh_proj.scales" in weights
        assert "eh_proj.biases" in weights
        assert "layers.1.mixer.switch_mlp.fc1.scales" in weights
        # Norms and the fp32 correction bias are never quantized.
        assert "final_layernorm.weight" in weights
        assert weights["final_layernorm.weight"].dtype != mx.uint32
        assert "layers.1.mixer.gate.e_score_correction_bias" in weights

    def test_raises_when_source_has_no_mtp_tensors(self, tmp_path):
        text_config = _tiny_nemotron_h_text_config()
        source = tmp_path / "no_mtp"
        source.mkdir()
        (source / "config.json").write_text(
            json.dumps({"model_type": "nemotron_h", **text_config.to_dict()})
        )
        mx.save_safetensors(
            str(source / "model.safetensors"),
            {"backbone.layers.0.norm.weight": mx.ones((text_config.hidden_size,))},
            metadata={},
        )

        with pytest.raises(ValueError, match="No mtp.\\* tensors found"):
            split_nemotron_h_mtp(str(source), str(tmp_path / "out"))
