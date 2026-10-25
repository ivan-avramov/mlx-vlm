"""nemotron_h_mtp: the native-MTP drafter for the `nemotron_h` hybrid family.

Fork: the drafter itself is upstream's (#2528), taken byte-identically at the
2026-10-25 sync; the fork's own implementation was retired after a bit-exact
equivalence check (docs/proposals/nemotron-h-mtp-convergence.md). Upstream's
own tests are ported into test_speculative.py. This file keeps the contracts
the fork relies on that those tests do not pin: drafter-kind registration,
config rejections, cache allocation, target binding against nemotron_h's
`backbone.embeddings`, and uniform batch acceptance -- all on tiny synthetic
configs.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.nemotron_h import Model as NemotronHTarget
from mlx_vlm.models.nemotron_h.config import ModelConfig as NemotronHConfig
from mlx_vlm.speculative.drafters import DRAFTER_KIND_BY_MODEL_TYPE
from mlx_vlm.speculative.drafters.nemotron_h_mtp import (
    ModelConfig as NemotronHMTPConfig,
)
from mlx_vlm.speculative.drafters.nemotron_h_mtp import NemotronHMTPDraftModel

HIDDEN_SIZE = 32
VOCAB_SIZE = 32


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
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        moe_latent_size=None,
        n_group=1,
        n_routed_experts=4,
        n_shared_experts=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        num_nextn_predict_layers=1,
        mtp_layers_block_type=["attention", "moe"],
    )
    kwargs.update(overrides)
    return NemotronHConfig(**kwargs)


def _drafter(**text_overrides) -> NemotronHMTPDraftModel:
    text = _tiny_nemotron_h_text_config(**text_overrides)
    return NemotronHMTPDraftModel(NemotronHMTPConfig(text_config=text, block_size=2))


def _target() -> NemotronHTarget:
    return NemotronHTarget(_tiny_nemotron_h_text_config())


def _greedy(logits):
    return mx.argmax(logits, axis=-1)


class TestRegistration:
    def test_nemotron_h_mtp_maps_to_mtp_kind(self):
        assert DRAFTER_KIND_BY_MODEL_TYPE["nemotron_h_mtp"] == "mtp"


class TestConfig:
    def test_layer_pattern_comes_from_mtp_layers_block_type(self):
        assert _drafter().config.layer_pattern == ["*", "E"]

    @pytest.mark.parametrize("block_type", ["mamba", "mlp"])
    def test_non_attention_moe_block_types_are_rejected(self, block_type):
        with pytest.raises(ValueError, match="attention/MoE layer pattern"):
            _drafter(mtp_layers_block_type=["attention", block_type])

    def test_missing_text_config_raises(self):
        with pytest.raises(ValueError, match="requires text_config"):
            NemotronHMTPConfig()

    def test_more_than_one_nextn_layer_is_rejected(self):
        with pytest.raises(ValueError, match="num_nextn_predict_layers=1"):
            _drafter(num_nextn_predict_layers=2)


class TestConstruction:
    def test_cache_is_allocated_only_for_the_attention_sublayer(self):
        cache = _drafter().make_cache()
        assert len(cache) == 1 and isinstance(cache[0], KVCache)


class TestBind:
    def test_bind_uses_the_targets_backbone_embeddings_and_lm_head(self):
        target = _target()
        drafter = _drafter()
        drafter.bind(target)
        lm = target.language_model
        assert drafter._input_embed is lm.backbone.embeddings
        assert drafter._lm_head_fn is lm.lm_head

    def test_bind_rejects_a_non_nemotron_target(self):
        target = SimpleNamespace(
            language_model=SimpleNamespace(
                embed_tokens=nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE),
                lm_head=nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False),
            )
        )
        with pytest.raises(ValueError, match="Nemotron-H target"):
            _drafter().bind(target)


class TestRounds:
    def _prefilled(self):
        mx.random.seed(0)
        drafter = _drafter()
        drafter.reset(_target())
        hidden = mx.random.normal((1, 4, HIDDEN_SIZE))
        drafter.prefill_from_target_hidden(
            mx.array([[3, 7, 11, 5]]), hidden, 9, _greedy, greedy=True
        )
        return drafter, hidden

    def test_draft_block_emits_block_minus_one_tokens(self):
        drafter, hidden = self._prefilled()
        tokens = drafter.draft_block(
            mx.array([9]), hidden[:, -1:], None, 4, _greedy, greedy=True
        )
        assert tokens.shape == (1, 3)
        assert drafter._cache[0].offset == 4 + 2

    def test_ragged_batch_acceptance_raises(self):
        drafter, _ = self._prefilled()
        with pytest.raises(ValueError, match="uniform batch acceptance"):
            drafter.accept_verified_tokens_batch(
                mx.zeros((2, 3, HIDDEN_SIZE)),
                mx.zeros((2, 2), dtype=mx.int32),
                [1, 2],
                mx.array([4, 5]),
                _greedy,
                greedy=True,
            )


class TestChunkedPrefill:
    @pytest.mark.parametrize("prefill_step_size", [2, 4])
    def test_mtp_on_a_long_prompt_matches_plain_decode(self, prefill_step_size):
        # Upstream's drafter rejects a prefill whose hidden states do not cover
        # every prompt token; the fork's nemotron_h target has its own
        # chunked-prefill policy. A prompt longer than prefill_step_size must
        # still reach the drafter whole, and speculation must stay exact.
        from mlx_vlm.generate.ar import generate_step

        mx.random.seed(5)
        text = _tiny_nemotron_h_text_config(
            num_hidden_layers=3, hybrid_override_pattern="M*E"
        )
        target = NemotronHTarget(text)
        draft = NemotronHMTPDraftModel(NemotronHMTPConfig(text_config=text))
        ids = mx.array([[1, 5, 9, 2, 7, 3, 8, 4, 6, 11, 13]])
        common = dict(
            max_tokens=10, temperature=0.0, prefill_step_size=prefill_step_size
        )
        expected = [t for t, _ in generate_step(ids, target, None, None, **common)]
        actual = [
            t
            for t, _ in generate_step(
                ids,
                target,
                None,
                None,
                draft_model=draft,
                draft_kind="mtp",
                draft_block_size=3,
                **common,
            )
        ]
        assert actual == expected
        assert draft.draft_lens
