"""M34 -- layer-scoped expert-budget expansion.

Fork-only. Tiny synthetic configs only; never loads a real checkpoint. Spec:
`docs/specs/m34-moe-expert-expansion.md` in the mlx_local_stack repo.
"""

import os

import mlx.core as mx
import pytest

from mlx_vlm.models.moe_expand import (
    MoeExpansion,
    active_expert_count,
    apply_moe_expansion,
    decay_factor,
    expand_route,
    expand_route_with_weight_base,
    format_moe_expand,
    parse_moe_expand,
)
from mlx_vlm.models.nemotron_h.language import NemotronHMoE


def _to_dense(inds: mx.array, weights: mx.array, e: int) -> mx.array:
    """Scatter (inds, weights) -- both shape (..., n) -- into a dense (..., e)
    array. Lets tests compare kept-sets/weights permutation-invariantly."""
    dense = mx.zeros(inds.shape[:-1] + (e,))
    return mx.put_along_axis(dense, inds, weights, axis=-1)


# ---------------------------------------------------------------------------
# expand_route: pure function
# ---------------------------------------------------------------------------


class TestExpandRoute:
    def test_t_zero_keeps_exactly_n(self):
        p = mx.array([0.30, 0.05, 0.20, 0.01, 0.15, 0.02, 0.25, 0.02])
        k, n = 2, 5
        inds, weights = expand_route(p, k, n, t=0.0, d=0.5)
        assert inds.shape == (n,)
        assert weights.shape == (n,)
        # every one of the n ranks is kept -> every weight nonzero
        assert bool(mx.all(weights != 0).item())

    def test_floor_n4_always_kept_even_under_a_harsh_threshold(self):
        # A steep drop-off after rank 2: with T close to 1, everything past
        # rank floor(N/4)=2 that isn't within the native top-K should be
        # prunable -- but floor(N/4) ranks (or K, whichever is larger) must
        # never be pruned.
        p = mx.array([0.9, 0.09, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025])
        k, n = 1, 8  # floor(N/4) = 2 > k=1
        inds, weights = expand_route(p, k, n, t=0.999, d=0.5)
        floor_guard = max(n // 4, k)
        assert bool(mx.all(weights[:floor_guard] != 0).item())

    def test_never_more_than_n_experts(self):
        p = mx.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        k, n = 2, 4
        inds, weights = expand_route(p, k, n, t=0.0, d=0.5)
        assert inds.shape[-1] == n
        assert weights.shape[-1] == n

    def test_tie_break_by_lowest_expert_id(self):
        # Four-way tie at the top -- must resolve to ascending id order.
        p = mx.array([0.25, 0.25, 0.25, 0.25, 0.1, 0.1, 0.1, 0.1])
        k, n = 2, 4
        inds, _ = expand_route(p, k, n, t=0.0, d=0.5)
        assert inds.tolist() == [0, 1, 2, 3]

    def test_tie_break_is_deterministic_across_repeated_calls(self):
        # Guards against relying on argsort tie-order happening to be stable.
        p = mx.array([0.1, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05])
        k, n = 2, 8
        first, _ = expand_route(p, k, n, t=0.0, d=0.5)
        for _ in range(10):
            again, _ = expand_route(p, k, n, t=0.0, d=0.5)
            assert first.tolist() == again.tolist()
        # ranks 1/2/3 are the p=0.3 three-way tie -> ascending id (1, 2, 3)
        assert first.tolist()[0:3] == [1, 2, 3]

    def test_renormalized_weights_sum_to_one_over_kept(self):
        p = mx.array([0.30, 0.05, 0.20, 0.01, 0.15, 0.02, 0.25, 0.02])
        k, n = 2, 6
        _, weights = expand_route(p, k, n, t=0.5, d=0.5)
        assert abs(float(weights.sum().item()) - 1.0) < 1e-5

    def test_appendix_a_decay_factors(self):
        # N=20, K=8, D=0.5 -> ranks 9/12/15/18/20: 0.990/0.856/0.723/0.589/0.500
        k, n, d = 8, 20, 0.5
        expected = {9: 0.990, 12: 0.856, 15: 0.723, 18: 0.589, 20: 0.500}
        for j, exp_val in expected.items():
            assert abs(decay_factor(j, k, n, d) - exp_val) < 1e-3

    def test_decay_factor_ranks_at_or_below_k_are_unweighted(self):
        for j in range(1, 9):
            assert decay_factor(j, k=8, n=20, d=0.5) == 1.0

    def test_n_equals_k_plus_1_fallback(self):
        # D=0.5 -> factor = (0.99 + 0.5) / 2 = 0.745
        assert abs(decay_factor(9, k=8, n=9, d=0.5) - 0.745) < 1e-9

    def test_n_equals_k_returns_native_top_k_weights_bit_identically(self):
        # k=2 sidesteps float-summation-order concerns entirely: a+b is exact
        # regardless of gather order (only 3+-term sums can reassociate).
        p = mx.array([0.1, 0.05, 0.6, 0.02, 0.03, 0.15, 0.03, 0.02])
        k = 2
        native_inds = mx.argpartition(p, kth=-k, axis=-1)[..., -k:]
        native_scores = mx.take_along_axis(p, native_inds, axis=-1)
        native_scores = native_scores / native_scores.sum(axis=-1, keepdims=True)
        native_dense = _to_dense(native_inds, native_scores, p.shape[-1])

        for t in (0.0, 0.3, 0.999):
            inds, weights = expand_route(p, k, k, t=t, d=0.5)
            dense = _to_dense(inds, weights, p.shape[-1])
            assert mx.array_equal(dense, native_dense), t

    def test_batched_shape_is_preserved(self):
        p = mx.random.uniform(shape=(2, 3, 8))
        inds, weights = expand_route(p, k=2, n=5, t=0.5, d=0.5)
        assert inds.shape == (2, 3, 5)
        assert weights.shape == (2, 3, 5)


# ---------------------------------------------------------------------------
# expand_route_with_weight_base (nemotron_h's split rank/weight arrays)
# ---------------------------------------------------------------------------


class TestExpandRouteWithWeightBase:
    def test_weight_base_differs_from_rank_base(self):
        rank_p = mx.array([0.1, 0.9, 0.2, 0.05, 0.3, 0.02, 0.15, 0.01])
        weight_p = mx.array([9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        inds, weights = expand_route_with_weight_base(
            rank_p, weight_p, k=2, n=4, t=0.0, d=0.5
        )
        # Ranking picks by rank_p (expert 1 is top), but the weight numerator
        # must come from weight_p, not rank_p.
        assert int(inds[0].item()) == 1
        assert abs(float(weights.sum().item()) - 1.0) < 1e-5

    def test_normalize_false_skips_renormalization(self):
        rank_p = mx.array([0.1, 0.9, 0.2, 0.05])
        weight_p = mx.array([1.0, 2.0, 3.0, 4.0])
        inds, weights = expand_route_with_weight_base(
            rank_p, weight_p, k=2, n=2, t=0.0, d=0.5, normalize=False
        )
        dense = _to_dense(inds, weights, 4)
        # raw = weight_p * factor(1) * kept(1) for the top-2 ranks, unnormalized
        assert abs(float(dense[1].item()) - 2.0) < 1e-6
        assert abs(float(dense[2].item()) - 3.0) < 1e-6


# ---------------------------------------------------------------------------
# active_expert_count
# ---------------------------------------------------------------------------


def test_active_expert_count_outside_gate_reads_k():
    p = mx.array([0.4, 0.3, 0.2, 0.1])
    inds = mx.argpartition(p, kth=-2, axis=-1)[..., -2:]
    scores = mx.take_along_axis(p, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    assert int(active_expert_count(scores).item()) == 2


def test_active_expert_count_inside_gate_is_in_range():
    p = mx.array([0.30, 0.05, 0.20, 0.01, 0.15, 0.02, 0.25, 0.02])
    k, n = 2, 6
    _, weights = expand_route(p, k, n, t=0.5, d=0.5)
    count = int(active_expert_count(weights).item())
    assert n // 4 <= count <= n


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class TestParseMoeExpand:
    def test_round_trip(self):
        s = "27-39:20:0.8:0.5"
        exp = parse_moe_expand(s)
        assert exp == MoeExpansion(layers=(27, 39), n=20, t=0.8, d=0.5)
        assert format_moe_expand(exp) == s

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "27:39:20:0.8:0.5",  # missing the '-' in the layer range
            "27-39-20:0.8:0.5",  # wrong number of ':' groups
            "abc-39:20:0.8:0.5",  # non-integer layer
            "39-27:20:0.8:0.5",  # LE < LS
            "-1-39:20:0.8:0.5",  # negative LS -- also malformed as '-'-split
            "27-39:0:0.8:0.5",  # N < 1
            "27-39:20:1.5:0.5",  # T out of [0, 1]
            "27-39:20:0.8:0",  # D not in (0, 1]
            "27-39:20:0.8:1.5",  # D out of (0, 1]
        ],
    )
    def test_malformed_strings_raise(self, bad):
        with pytest.raises(ValueError):
            parse_moe_expand(bad)


# ---------------------------------------------------------------------------
# Tiny qwen3_5_moe model: identity + range-scoped forward
# ---------------------------------------------------------------------------


def _qwen_config(num_hidden_layers=4):
    from mlx_vlm.models.qwen3_5_moe.config import TextConfig

    return TextConfig(
        model_type="qwen3_5_moe",
        hidden_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-5,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=64,
        full_attention_interval=4,
        head_dim=8,
    )


def _qwen_model_config():
    # get_rope_index reads these four fields unconditionally, even on the
    # pure-text (no image_grid_thw) path -- a bare `LanguageModel(text_config)`
    # (config=None) raises AttributeError before ever reaching the MoE block.
    from types import SimpleNamespace

    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=999999,
        video_token_id=999998,
        vision_start_token_id=999997,
    )


class TestQwenModelIntegration:
    def _model(self):
        from mlx_vlm.models.qwen3_5_moe.language import LanguageModel

        config = _qwen_config()
        model = LanguageModel(config, config=_qwen_model_config())
        prompt = mx.array([[1, 2, 3, 4, 5]])
        return model, prompt

    def test_expansion_unset_is_byte_identical(self):
        model, prompt = self._model()
        cache = model.make_cache()
        baseline = model(prompt, cache=cache).logits
        mx.eval(baseline)

        cache2 = model.make_cache()
        model.set_moe_expansion(None)
        out = model(prompt, cache=cache2).logits
        mx.eval(out)
        assert mx.array_equal(baseline, out)

    def test_n_equals_k_is_byte_identical(self):
        from mlx_vlm.models.moe_expand import MoeExpansion

        model, prompt = self._model()
        cache = model.make_cache()
        baseline = model(prompt, cache=cache).logits
        mx.eval(baseline)

        cache2 = model.make_cache()
        model.set_moe_expansion(MoeExpansion(layers=(0, 3), n=2, t=0.5, d=0.5))
        out = model(prompt, cache=cache2).logits
        mx.eval(out)
        assert mx.array_equal(baseline, out)

    def test_out_of_range_layers_is_byte_identical(self):
        from mlx_vlm.models.moe_expand import MoeExpansion

        model, prompt = self._model()
        cache = model.make_cache()
        baseline = model(prompt, cache=cache).logits
        mx.eval(baseline)

        cache2 = model.make_cache()
        # 4-layer model: layer indices 0..3. Range starting past the last
        # layer can never match.
        model.set_moe_expansion(MoeExpansion(layers=(10, 12), n=6, t=0.5, d=0.5))
        out = model(prompt, cache=cache2).logits
        mx.eval(out)
        assert mx.array_equal(baseline, out)

    def test_expansion_only_changes_in_range_layers(self):
        # nn.Module dunder lookup goes through the TYPE, not the instance, so
        # this patches the CLASS's __call__ (keyed by the block's own
        # `layer_idx`) rather than assigning to `layer.mlp.__call__`, which
        # `mlp(x)` would silently never invoke.
        from mlx_vlm.models.moe_expand import MoeExpansion
        from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

        model, prompt = self._model()
        captured = {}
        orig_call = Qwen3_5MoeSparseMoeBlock.__call__

        def wrapped(self, x, target_verify=False):
            out = orig_call(self, x, target_verify)
            captured.setdefault(self.layer_idx, []).append(out)
            return out

        Qwen3_5MoeSparseMoeBlock.__call__ = wrapped
        try:
            cache = model.make_cache()
            model.set_moe_expansion(None)
            model(prompt, cache=cache)

            cache2 = model.make_cache()
            model.set_moe_expansion(MoeExpansion(layers=(2, 3), n=6, t=0.5, d=0.5))
            model(prompt, cache=cache2)
        finally:
            Qwen3_5MoeSparseMoeBlock.__call__ = orig_call

        for i in (0, 1):
            mx.eval(captured[i][0], captured[i][1])
            assert mx.array_equal(captured[i][0], captured[i][1]), i
        for i in (2, 3):
            mx.eval(captured[i][0], captured[i][1])
            assert not mx.array_equal(captured[i][0], captured[i][1]), i

    def test_experts_per_token_outside_and_inside_gate(self):
        from mlx_vlm.models.moe_expand import MoeExpansion
        from mlx_vlm.models.switch_layers import SwitchGLU

        model, prompt = self._model()

        k = model.model.layers[0].mlp.top_k
        n = 6

        layer_of_switch_mlp = {
            id(layer.mlp.switch_mlp): i for i, layer in enumerate(model.model.layers)
        }
        captured_k = {}
        orig_call = SwitchGLU.__call__

        def wrapped(self, x, indices):
            idx = layer_of_switch_mlp.get(id(self))
            if idx is not None:
                captured_k.setdefault(idx, []).append(indices.shape[-1])
            return orig_call(self, x, indices)

        SwitchGLU.__call__ = wrapped
        try:
            model.set_moe_expansion(MoeExpansion(layers=(2, 3), n=n, t=0.5, d=0.5))
            cache = model.make_cache()
            model(prompt, cache=cache)
        finally:
            SwitchGLU.__call__ = orig_call

        for i in (0, 1):
            assert captured_k[i][0] == k
        for i in (2, 3):
            assert captured_k[i][0] == n

    def test_set_moe_expansion_reports_in_range_layer_count(self):
        from mlx_vlm.models.moe_expand import MoeExpansion

        model, _ = self._model()
        count = model.set_moe_expansion(MoeExpansion(layers=(2, 3), n=6, t=0.5, d=0.5))
        assert count == 2
        assert model.set_moe_expansion(None) == 0


# ---------------------------------------------------------------------------
# Tiny nemotron_h model: identity + norm_topk_prob/routed_scaling_factor
# ---------------------------------------------------------------------------


def _nemotron_config(hybrid_override_pattern=("M", "*", "-", "E", "E")):
    from mlx_vlm.models.nemotron_h.config import ModelConfig

    return ModelConfig(
        model_type="nemotron_h",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=len(hybrid_override_pattern),
        max_position_embeddings=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_bias=False,
        mamba_num_heads=2,
        mamba_head_dim=8,
        mamba_proj_bias=False,
        ssm_state_size=16,
        conv_kernel=4,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=True,
        hybrid_override_pattern=list(hybrid_override_pattern),
        moe_intermediate_size=16,
        n_group=1,
        n_routed_experts=8,
        n_shared_experts=1,
        moe_shared_expert_intermediate_size=16,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=2.0,
    )


class TestNemotronModelIntegration:
    def _model(self, **kwargs):
        from mlx_vlm.models import nemotron_h

        config = _nemotron_config(**kwargs)
        return nemotron_h.Model(config), mx.array([[1, 2, 3, 4, 5]])

    def test_expansion_unset_is_byte_identical(self):
        model, prompt = self._model()
        cache = model.make_cache()
        baseline = model(prompt, cache=cache).logits
        mx.eval(baseline)

        cache2 = model.make_cache()
        model.language_model.set_moe_expansion(None)
        out = model(prompt, cache=cache2).logits
        mx.eval(out)
        assert mx.array_equal(baseline, out)

    def test_n_equals_k_is_byte_identical(self):
        from mlx_vlm.models.moe_expand import MoeExpansion

        model, prompt = self._model()
        cache = model.make_cache()
        baseline = model(prompt, cache=cache).logits
        mx.eval(baseline)

        cache2 = model.make_cache()
        model.language_model.set_moe_expansion(
            MoeExpansion(layers=(0, 4), n=2, t=0.5, d=0.5)
        )
        out = model(prompt, cache=cache2).logits
        mx.eval(out)
        assert mx.array_equal(baseline, out)

    def test_out_of_range_layers_is_byte_identical(self):
        from mlx_vlm.models.moe_expand import MoeExpansion

        model, prompt = self._model()
        cache = model.make_cache()
        baseline = model(prompt, cache=cache).logits
        mx.eval(baseline)

        cache2 = model.make_cache()
        model.language_model.set_moe_expansion(
            MoeExpansion(layers=(10, 12), n=6, t=0.5, d=0.5)
        )
        out = model(prompt, cache=cache2).logits
        mx.eval(out)
        assert mx.array_equal(baseline, out)

    def test_expansion_changes_only_in_range_moe_layers(self):
        # Two "E" layers at absolute indices 3 and 4. Patches the CLASS's
        # __call__ (keyed by the mixer's own `layer_idx`) -- nn.Module dunder
        # lookup goes through the type, so instance-level assignment would
        # silently never fire.
        from mlx_vlm.models.moe_expand import MoeExpansion

        model, prompt = self._model()

        captured = {}
        orig_call = NemotronHMoE.__call__

        def wrapped(self, x):
            out = orig_call(self, x)
            captured.setdefault(self.layer_idx, []).append(out)
            return out

        NemotronHMoE.__call__ = wrapped
        try:
            cache = model.make_cache()
            model.language_model.set_moe_expansion(None)
            model(prompt, cache=cache)

            cache2 = model.make_cache()
            model.language_model.set_moe_expansion(
                MoeExpansion(layers=(4, 4), n=6, t=0.5, d=0.5)
            )
            model(prompt, cache=cache2)
        finally:
            NemotronHMoE.__call__ = orig_call

        mx.eval(captured[3][0], captured[3][1])
        assert mx.array_equal(captured[3][0], captured[3][1])
        mx.eval(captured[4][0], captured[4][1])
        assert not mx.array_equal(captured[4][0], captured[4][1])

    def test_routed_scaling_and_norm_topk_prob_still_applied(self):
        from mlx_vlm.models.moe_expand import MoeExpansion
        from mlx_vlm.models.nemotron_h.language import group_expert_select

        gates = mx.array([[0.1, 2.0, -0.3, 0.4, 0.05, -0.2, 0.15, 0.02]])
        bias = mx.zeros((8,))
        inds, scores = group_expert_select(
            gates,
            bias,
            top_k=2,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=2.0,
            norm_topk_prob=True,
            expand_n=6,
            expand_t=0.5,
            expand_d=0.5,
        )
        mx.eval(inds, scores)
        # norm_topk_prob divides by the kept sum, THEN routed_scaling_factor
        # multiplies -- so scores should sum to routed_scaling_factor exactly.
        assert abs(float(scores.sum().item()) - 2.0) < 1e-4

    def test_set_moe_expansion_reports_in_range_layer_count(self):
        from mlx_vlm.models.moe_expand import MoeExpansion

        model, _ = self._model()
        count = model.language_model.set_moe_expansion(
            MoeExpansion(layers=(3, 4), n=6, t=0.5, d=0.5)
        )
        assert count == 2
        assert model.language_model.set_moe_expansion(None) == 0


# ---------------------------------------------------------------------------
# Drafter isolation
# ---------------------------------------------------------------------------


def test_set_moe_expansion_leaves_an_attached_drafters_blocks_unset():
    from mlx_vlm.models.moe_expand import MoeExpansion
    from mlx_vlm.models.qwen3_5_moe.language import LanguageModel

    config = _qwen_config()
    target = LanguageModel(config)

    # A second, independent MoE language model standing in for an MTP
    # drafter -- structurally identical, deliberately attached as a plain
    # attribute to make sure set_moe_expansion does not walk arbitrary
    # attributes / nested modules looking for MoE blocks.
    drafter = LanguageModel(_qwen_config(num_hidden_layers=2))
    target.drafter = drafter

    exp = MoeExpansion(layers=(0, 3), n=6, t=0.5, d=0.5)
    target.set_moe_expansion(exp)

    for layer in target.model.layers:
        assert layer.mlp.moe_expand == exp
    for layer in drafter.model.layers:
        assert layer.mlp.moe_expand is None


def test_apply_moe_expansion_helper_targets_language_model_only():
    from mlx_vlm.models.qwen3_5_moe.language import LanguageModel

    config = _qwen_config()
    target = LanguageModel(config)
    n_in_range = apply_moe_expansion(target, "2-3:6:0.5:0.5")
    assert n_in_range == 2
    for i, layer in enumerate(target.model.layers):
        assert layer.mlp.moe_expand is not None
        assert layer.mlp.moe_expand.in_range(i) == (2 <= i <= 3)


def test_apply_moe_expansion_raises_for_non_moe_model():
    class _Dense:
        pass

    with pytest.raises(ValueError):
        apply_moe_expansion(_Dense(), "0-3:6:0.5:0.5")


# ---------------------------------------------------------------------------
# Verifier fixes on b95130c9
# ---------------------------------------------------------------------------


def _to_bf16(model):
    from mlx.utils import tree_map

    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    return model


class TestFix1Bf16DtypePreservation:
    """FIX-1 (blocker): `factor` in `_normalized_weights` was always float32,
    so on a bf16 model `weight_base * factor` promoted the qwen3_5_moe
    residual stream -- and every downstream KV cache -- to float32.
    nemotron_h was already protected by its own `.astype(y.dtype)` at the end
    of `NemotronHMoE.__call__`; its case here is a regression guard, not a
    blocker reproduction.
    """

    def test_qwen_expanded_forward_stays_bf16(self):
        from mlx_vlm.models.cache import KVCache
        from mlx_vlm.models.qwen3_5_moe.language import (
            LanguageModel,
            Qwen3_5MoeSparseMoeBlock,
        )

        config = _qwen_config()
        model = _to_bf16(LanguageModel(config, config=_qwen_model_config()))
        prompt = mx.array([[1, 2, 3, 4, 5]])

        captured = {}
        orig_call = Qwen3_5MoeSparseMoeBlock.__call__

        def wrapped(self, x, target_verify=False):
            out = orig_call(self, x, target_verify)
            captured.setdefault(self.layer_idx, []).append(out.dtype)
            return out

        Qwen3_5MoeSparseMoeBlock.__call__ = wrapped
        try:
            baseline_cache = model.make_cache()
            model.set_moe_expansion(None)
            baseline = model(prompt, cache=baseline_cache).logits
            mx.eval(baseline)

            expanded_cache = model.make_cache()
            model.set_moe_expansion(MoeExpansion(layers=(0, 3), n=6, t=0.5, d=0.5))
            out = model(prompt, cache=expanded_cache).logits
            mx.eval(out)
        finally:
            Qwen3_5MoeSparseMoeBlock.__call__ = orig_call

        assert baseline.dtype == mx.bfloat16
        assert out.dtype == baseline.dtype
        for dtypes in captured.values():
            for dt in dtypes:
                assert dt == mx.bfloat16

        # The full-attention layer's KVCache must also stay bf16 -- if the
        # routing weight promoted the residual stream to float32, the K/V
        # projections (and every cache built from them) would silently
        # double in size.
        kv_dtypes = [
            c.keys.dtype
            for c in expanded_cache
            if isinstance(c, KVCache) and c.keys is not None
        ]
        assert kv_dtypes, "expected at least one populated KVCache"
        assert all(dt == mx.bfloat16 for dt in kv_dtypes)

    def test_nemotron_expanded_forward_stays_bf16(self):
        from mlx_vlm.models import nemotron_h
        from mlx_vlm.models.cache import KVCache

        config = _nemotron_config()
        model = _to_bf16(nemotron_h.Model(config))
        prompt = mx.array([[1, 2, 3, 4, 5]])

        captured = {}
        orig_call = NemotronHMoE.__call__

        def wrapped(self, x):
            out = orig_call(self, x)
            captured.setdefault(self.layer_idx, []).append(out.dtype)
            return out

        NemotronHMoE.__call__ = wrapped
        try:
            baseline_cache = model.make_cache()
            model.language_model.set_moe_expansion(None)
            baseline = model(prompt, cache=baseline_cache).logits
            mx.eval(baseline)

            expanded_cache = model.make_cache()
            model.language_model.set_moe_expansion(
                MoeExpansion(layers=(3, 4), n=6, t=0.5, d=0.5)
            )
            out = model(prompt, cache=expanded_cache).logits
            mx.eval(out)
        finally:
            NemotronHMoE.__call__ = orig_call

        assert baseline.dtype == mx.bfloat16
        assert out.dtype == baseline.dtype
        for dtypes in captured.values():
            for dt in dtypes:
                assert dt == mx.bfloat16

        kv_dtypes = [
            c.keys.dtype
            for c in expanded_cache
            if isinstance(c, KVCache) and c.keys is not None
        ]
        assert kv_dtypes, "expected at least one populated KVCache"
        assert all(dt == mx.bfloat16 for dt in kv_dtypes)


class TestFix2NoAmbientEnvLeak:
    """FIX-2 (blocker, project rule): mlx-serve spawns the worker without
    `env=`, inheriting the router's full environment. cli.py must always
    write `MLX_VLM_MOE_EXPAND` (clearing it to "" when the flag is omitted),
    never merely set-when-given, or a stale export from an earlier invocation
    silently enables expansion while the cmdline/fingerprint say native.
    """

    def test_a_stale_ambient_export_is_cleared_when_the_flag_is_omitted(
        self, monkeypatch
    ):
        from mlx_vlm.server.cli import _configure_moe_expand

        monkeypatch.setenv("MLX_VLM_MOE_EXPAND", "27-39:20:0.8:0.5")
        _configure_moe_expand(None)  # --moe-expand not passed on THIS invocation
        assert os.environ["MLX_VLM_MOE_EXPAND"] == ""

    def test_an_empty_string_flag_also_clears_it(self, monkeypatch):
        from mlx_vlm.server.cli import _configure_moe_expand

        monkeypatch.setenv("MLX_VLM_MOE_EXPAND", "27-39:20:0.8:0.5")
        _configure_moe_expand("")
        assert os.environ["MLX_VLM_MOE_EXPAND"] == ""

    def test_a_given_value_is_forwarded(self, monkeypatch):
        from mlx_vlm.server.cli import _configure_moe_expand

        monkeypatch.delenv("MLX_VLM_MOE_EXPAND", raising=False)
        _configure_moe_expand("27-39:20:0.8:0.5")
        assert os.environ["MLX_VLM_MOE_EXPAND"] == "27-39:20:0.8:0.5"

    def test_a_malformed_value_raises_before_touching_the_env(self, monkeypatch):
        from mlx_vlm.server.cli import _configure_moe_expand

        monkeypatch.delenv("MLX_VLM_MOE_EXPAND", raising=False)
        with pytest.raises(ValueError):
            _configure_moe_expand("not-a-valid-spec")

    def test_generation_py_treats_the_cleared_env_as_off(self, monkeypatch):
        # Mirrors the exact read site in server/generation.py's
        # _initialize_model: `if moe_expand_spec:` must skip on "".
        monkeypatch.setenv("MLX_VLM_MOE_EXPAND", "")
        moe_expand_spec = os.environ.get("MLX_VLM_MOE_EXPAND")
        assert not moe_expand_spec


def test_fix3_apply_moe_expand_and_log_writes_to_stderr(capsys):
    """FIX-3: mlx-serve sends the worker's stdout to DEVNULL -- only stderr
    reaches $TMPDIR/mlx-manager-logs -- so the startup line must also be
    printed there directly, not just via `logger.info` (which this fork
    routes to stdout)."""
    from mlx_vlm.models.qwen3_5_moe.language import LanguageModel
    from mlx_vlm.server.generation import _apply_moe_expand_and_log

    config = _qwen_config()
    model = LanguageModel(config, config=_qwen_model_config())

    n_layers = _apply_moe_expand_and_log(model, "2-3:6:0.5:0.5")

    captured = capsys.readouterr()
    assert n_layers == 2
    assert "moe_expand=2-3:6:0.5:0.5 layers=2" in captured.err


class TestFix4NoOpCounting:
    """FIX-4: `set_moe_expansion` counted layers by `in_range` alone, so
    `n <= k` reported `layers=N` while every one of them was a silent
    native-path no-op. Counting now additionally requires `exp.n >
    that layer's top_k`; `apply_moe_expansion` (the CLI/chat entry point)
    raises when the count comes back 0 but the range does contain MoE
    blocks. Direct `set_moe_expansion` calls stay non-raising (`strict=False`
    default) so the spec-required N==K native-passthrough tests above keep
    working."""

    def test_qwen_set_moe_expansion_does_not_count_n_equals_k_layers(self):
        from mlx_vlm.models.qwen3_5_moe.language import LanguageModel

        config = _qwen_config()  # num_experts_per_tok=2
        model = LanguageModel(config, config=_qwen_model_config())
        count = model.set_moe_expansion(MoeExpansion(layers=(0, 3), n=2, t=0.5, d=0.5))
        assert count == 0

    def test_qwen_set_moe_expansion_direct_call_does_not_raise_for_n_equals_k(self):
        from mlx_vlm.models.qwen3_5_moe.language import LanguageModel

        config = _qwen_config()
        model = LanguageModel(config, config=_qwen_model_config())
        # strict=False (the default) must NOT raise -- N == K is the
        # spec-required native-passthrough probe, not a misconfiguration.
        count = model.set_moe_expansion(MoeExpansion(layers=(0, 3), n=2, t=0.5, d=0.5))
        assert count == 0

    def test_apply_moe_expansion_raises_when_n_never_exceeds_top_k_in_range(self):
        from mlx_vlm.models.qwen3_5_moe.language import LanguageModel

        config = _qwen_config()
        model = LanguageModel(config, config=_qwen_model_config())
        with pytest.raises(ValueError, match="does not exceed native top_k"):
            apply_moe_expansion(model, "0-3:2:0.5:0.5")

    def test_apply_moe_expansion_does_not_raise_when_some_layers_do_expand(self):
        from mlx_vlm.models.qwen3_5_moe.language import LanguageModel

        config = _qwen_config()
        model = LanguageModel(config, config=_qwen_model_config())
        count = apply_moe_expansion(model, "2-3:6:0.5:0.5")
        assert count == 2

    def test_nemotron_set_moe_expansion_does_not_count_n_equals_k_layers(self):
        from mlx_vlm.models import nemotron_h

        config = _nemotron_config()  # num_experts_per_tok=2
        model = nemotron_h.Model(config)
        count = model.language_model.set_moe_expansion(
            MoeExpansion(layers=(3, 4), n=2, t=0.5, d=0.5)
        )
        assert count == 0

    def test_apply_moe_expansion_raises_for_nemotron_n_equals_k(self):
        from mlx_vlm.models import nemotron_h

        config = _nemotron_config()
        model = nemotron_h.Model(config)
        with pytest.raises(ValueError, match="does not exceed native top_k"):
            apply_moe_expansion(model, "3-4:2:0.5:0.5")


class TestFix5NumExpertsValidation:
    """FIX-5: `N > num_experts` used to fail deep inside the first request
    (an out-of-bounds gather) instead of at startup/apply time."""

    def test_apply_moe_expansion_raises_when_n_exceeds_num_experts_qwen(self):
        from mlx_vlm.models.qwen3_5_moe.language import LanguageModel

        config = _qwen_config()  # num_experts=8
        model = LanguageModel(config, config=_qwen_model_config())
        with pytest.raises(ValueError, match="exceeds this model's total expert count"):
            apply_moe_expansion(model, "0-3:9:0.5:0.5")

    def test_apply_moe_expansion_raises_when_n_exceeds_num_experts_nemotron(self):
        from mlx_vlm.models import nemotron_h

        config = _nemotron_config()  # n_routed_experts=8
        model = nemotron_h.Model(config)
        with pytest.raises(ValueError, match="exceeds this model's total expert count"):
            apply_moe_expansion(model, "3-4:9:0.5:0.5")

    def test_apply_moe_expansion_allows_n_equal_to_num_experts(self):
        from mlx_vlm.models.qwen3_5_moe.language import LanguageModel

        config = _qwen_config()  # num_experts=8
        model = LanguageModel(config, config=_qwen_model_config())
        count = apply_moe_expansion(model, "0-3:8:0.5:0.5")
        assert count == 4
