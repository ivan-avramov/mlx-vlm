"""Input contract of `nemotron_h`'s backbone.

Fork-only. Upstream's `NemotronHModel.__call__` guard read
`if (inputs is None) == (inputs_embeds is None)`, which rejects both-supplied
as well as neither-supplied. Only neither is unserviceable: the branch right
below the guard already handles both-supplied by preferring `inputs_embeds`.

Rejecting it broke the shared autoregressive loop, which passes both --
`generate/ar.py` calls `model.language_model(y, inputs_embeds=inputs_embeds,
...)` because for a VLM `inputs_embeds` is the vision-merged embedding while
the token ids are still carried alongside. So `mlx_vlm.generate` raised on
every nemotron_h model, while the server path (which passes `inputs_embeds`
only) worked -- which is why the AR loop is right and the guard was the
outlier.

No weights needed; a tiny synthetic config is enough.
"""

import mlx.core as mx
import pytest

from mlx_vlm.models.nemotron_h.config import ModelConfig
from mlx_vlm.models.nemotron_h.language import NemotronHModel

HIDDEN_SIZE = 16


def _config() -> ModelConfig:
    # "M-" is one Mamba block plus one MLP block: the smallest pattern that
    # still exercises the hybrid cache indexing in __call__.
    return ModelConfig(
        model_type="nemotron_h",
        vocab_size=32,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=32,
        num_hidden_layers=2,
        max_position_embeddings=64,
        num_attention_heads=2,
        num_key_value_heads=1,
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
        hybrid_override_pattern="M-",
    )


@pytest.fixture
def backbone() -> NemotronHModel:
    return NemotronHModel(_config())


@pytest.fixture
def input_ids() -> mx.array:
    return mx.array([[1, 2, 3]])


@pytest.fixture
def foreign_embeds() -> mx.array:
    """Embeddings that are deliberately NOT `embeddings(input_ids)`.

    A vision-merged embedding does not equal the token embedding of the ids
    carried beside it, and using a distinct signal here is what makes the
    preference assertion below meaningful rather than vacuous.
    """
    return mx.ones((1, 3, HIDDEN_SIZE)) * 0.5


class TestBackboneInputContract:
    def test_input_ids_alone_are_accepted(self, backbone, input_ids):
        out = backbone(inputs=input_ids)
        mx.eval(out)

        assert out.shape == (1, 3, HIDDEN_SIZE)

    def test_inputs_embeds_alone_are_accepted(self, backbone, foreign_embeds):
        out = backbone(inputs_embeds=foreign_embeds)
        mx.eval(out)

        assert out.shape == (1, 3, HIDDEN_SIZE)

    def test_both_supplied_is_accepted_and_prefers_inputs_embeds(
        self, backbone, input_ids, foreign_embeds
    ):
        """The AR-loop call shape. Upstream's guard raised here."""
        both = backbone(inputs=input_ids, inputs_embeds=foreign_embeds)
        embeds_only = backbone(inputs_embeds=foreign_embeds)
        ids_only = backbone(inputs=input_ids)
        mx.eval(both, embeds_only, ids_only)

        # Not just "it didn't raise": the ids must be genuinely ignored, so the
        # result has to match the embeds-only call and differ from the ids-only
        # one. Without the second assertion this would still pass if the guard
        # were widened and the branch then re-embedded the ids.
        assert mx.allclose(both, embeds_only).item()
        assert not mx.allclose(both, ids_only).item()

    def test_neither_supplied_still_raises(self, backbone):
        with pytest.raises(ValueError, match="Provide inputs or inputs_embeds"):
            backbone()

    def test_a_backbone_without_embeddings_rejects_bare_input_ids(self, input_ids):
        """`with_embeddings=False` has no table to look ids up in.

        Widening the first guard must not let this case through to an
        AttributeError on the missing `self.embeddings`.
        """
        backbone = NemotronHModel(_config(), with_embeddings=False)

        with pytest.raises(ValueError, match="no token embedding table"):
            backbone(inputs=input_ids)
