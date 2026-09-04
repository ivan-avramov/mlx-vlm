from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, KVCache
from ..moe_expand import MoeExpansion, expand_route_with_weight_base
from ..recurrent_rollback import (
    RecurrentStateRollbackMixin,
)  # Fork: MTP speculative-verify rollback contract (see recurrent_rollback.py docstring)
from ..ssm import ssm_update, ssm_update_with_states
from ..switch_layers import SwitchMLP
from .config import ModelConfig


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float, group_size: int):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)
        self.group_size = group_size

    def __call__(self, x: mx.array, gate: mx.array = None) -> mx.array:
        if gate is not None:
            x = swiglu(gate, x)
        x = mx.unflatten(x, axis=-1, shape=(-1, self.group_size))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * x.flatten(-2)


class NemotronHMamba2Mixer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.num_heads = args.mamba_num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.ssm_state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.mamba_num_heads * args.mamba_head_dim
        self.n_groups = args.n_groups
        self.head_dim = args.mamba_head_dim
        self.time_step_limit = args.time_step_limit
        self.heads_per_group = self.num_heads // self.n_groups

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.conv_kernel,
            padding=0,
            groups=self.conv_dim,
            bias=args.use_conv_bias,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size, projection_size, bias=args.mamba_proj_bias
        )

        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)

        group_size = self.intermediate_size // self.n_groups
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=args.layer_norm_epsilon,
            group_size=group_size,
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.mamba_proj_bias
        )

    def _split_projected_states(self, projected: mx.array):
        # Nemotron-H checkpoints may tensor-core-pad the projection with two
        # unused ``d_mlp`` branches. The reference derives their width from the
        # loaded weight rather than the config and discards them before gate.
        base_size = self.intermediate_size + self.conv_dim + self.num_heads
        extra_size = projected.shape[-1] - base_size
        if extra_size < 0 or extra_size % 2:
            raise ValueError(
                "invalid Nemotron-H Mamba projection width: "
                f"got {projected.shape[-1]}, expected {base_size} plus an even padding"
            )
        d_mlp = extra_size // 2
        gate_start = 2 * d_mlp
        conv_start = gate_start + self.intermediate_size
        dt_start = conv_start + self.conv_dim
        return (
            projected[..., gate_start:conv_start],
            projected[..., conv_start:dt_start],
            projected[..., dt_start:],
        )

    def _conv(
        self,
        conv_input: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
        capture_states: bool = False,
    ) -> Tuple[mx.array, Optional[list]]:
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)

        conv_states = None
        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                    dtype=conv_input.dtype,
                )
            else:
                conv_state = cache[0]
            padded_input = mx.concatenate([conv_state, conv_input], axis=1)
            n_keep = self.conv_kernel_size - 1
            if capture_states:
                # Fork: MTP-verify capture (recurrent_sink/gdn_states, see
                # NemotronHModel.__call__ and recurrent_rollback.py). Pure
                # slicing, no compute: the conv state after position t is
                # the n_keep-wide window ending at t in the padded input --
                # at t == T-1 this is `padded_input[:, -n_keep:, :]`,
                # identical to the non-capture branch below. Left-padded
                # batches (`cache.lengths` set) use the take_along_axis
                # per-row-offset trick below instead, which this slicing
                # doesn't account for -- out of scope (the rollback contract
                # this feeds already requires uniform batch acceptance).
                if cache.lengths is not None:
                    raise NotImplementedError(
                        "NemotronHMamba2Mixer._conv: capture_states with "
                        "cache.lengths set (left-padded batch) is not "
                        "supported -- MTP-verify capture assumes a "
                        "uniform, non-left-padded batch."
                    )
                conv_states = [
                    padded_input[:, t + 1 : t + 1 + n_keep, :]
                    for t in range(conv_input.shape[1])
                ]
                cache[0] = conv_states[-1]
            elif cache.lengths is not None:
                t = padded_input.shape[1]
                ends = mx.clip(cache.lengths, 0, t - n_keep)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(padded_input, positions, axis=1)
            else:
                cache[0] = padded_input[:, -n_keep:, :]
        else:
            padded_input = mx.pad(
                conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)]
            )

        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output), conv_states

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
        capture_states: bool = False,
    ) -> Tuple[mx.array, Optional[list]]:
        batch_size, seq_len, _ = hidden_states.shape

        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        if cache:
            state = cache[1]
        else:
            state = None

        ssm_states = None
        if capture_states:
            y, state, states = ssm_update_with_states(
                hidden_states,
                self.A_log,
                B,
                C,
                self.D.astype(hidden_states.dtype),
                dt,
                self.dt_bias,
                state,
                self.time_step_limit,
                mask,
            )
            ssm_states = [states[:, t] for t in range(states.shape[1])]
        else:
            y, state = ssm_update(
                hidden_states,
                self.A_log,
                B,
                C,
                self.D.astype(hidden_states.dtype),
                dt,
                self.dt_bias,
                state,
                self.time_step_limit,
                mask,
            )
        if cache:
            cache[1] = state

        return y.reshape(batch_size, seq_len, self.intermediate_size), ssm_states

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array],
        cache: Optional[ArraysCache] = None,
        capture_sink: Optional[list] = None,
    ) -> mx.array:

        if capture_sink is not None and cache is None:
            raise ValueError("capture_sink requires a cache")

        projected = self.in_proj(hidden_states)

        gate, conv_input, dt = self._split_projected_states(projected)
        capture_states = capture_sink is not None
        conv_output, conv_states = self._conv(
            conv_input, cache, mask, capture_states=capture_states
        )
        hidden_states_ssm, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        y, ssm_states = self._ssm(
            hidden_states_ssm, B, C, dt, cache, mask, capture_states=capture_states
        )
        if cache:
            cache.advance(y.shape[1])
        if capture_states:
            # Fork: MTP-verify capture -- one snapshot per input position,
            # aligned index-for-index with conv_states (see NemotronHModel
            # .__call__'s recurrent_sink plumbing and
            # recurrent_rollback.py's rollback contract).
            capture_sink.extend(zip(conv_states, ssm_states))
        y = self.norm(y, gate)
        return self.out_proj(y)


class NemotronHAttention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = (
            args.head_dim
            if args.head_dim is not None
            else (args.hidden_size // args.num_attention_heads)
        )
        self.num_key_value_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = (
            self.k_proj(x)
            .reshape(B, L, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(B, L, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class NemotronHMLP(nn.Module):
    def __init__(self, args: ModelConfig, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or args.intermediate_size

        self.up_proj = nn.Linear(
            args.hidden_size, intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = nn.Linear(
            intermediate_size, args.hidden_size, bias=args.mlp_bias
        )

    def __call__(self, x):
        return self.down_proj(nn.relu2(self.up_proj(x)))


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
    expand_n=0,
    expand_t=0.0,
    expand_d=1.0,
):
    # M34: `expand_n <= top_k` (the default, 0) is the ORIGINAL code path --
    # unchanged below, not re-implemented. `expand_n > top_k` is the
    # layer-scoped expert-budget expansion (see ../moe_expand.py): selection
    # still runs on the bias-corrected, group-masked score (`scores`), but the
    # weight numerator is the plain sigmoid score (`orig_scores`), matching
    # native's `orig_scores`/`scores` split.

    orig_scores = scores = mx.sigmoid(gates.astype(mx.float32))
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    if expand_n > top_k:
        inds, out_scores = expand_route_with_weight_base(
            scores,
            orig_scores,
            k,
            expand_n,
            expand_t,
            expand_d,
            normalize=(top_k > 1 and norm_topk_prob),
        )
    else:
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        out_scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        if top_k > 1 and norm_topk_prob:
            denominator = out_scores.sum(axis=-1, keepdims=True)
            out_scores = out_scores / (denominator + 1e-20)
    out_scores = out_scores * routed_scaling_factor

    return inds, out_scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(
        self,
        x,
        expand_n: int = 0,
        expand_t: float = 0.0,
        expand_d: float = 1.0,
    ):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            expand_n,
            expand_t,
            expand_d,
        )


class NemotronHMoE(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_latent_size = config.moe_latent_size
        self.layer_idx = layer_idx
        # M34: layer-scoped expert-budget expansion. None == native top-K
        # everywhere (byte-identical to upstream); set via
        # `LanguageModel.set_moe_expansion`, never touched otherwise.
        self.moe_expand: Optional[MoeExpansion] = None

        expert_input_dim = (
            config.moe_latent_size
            if config.moe_latent_size is not None
            else config.hidden_size
        )
        self.switch_mlp = SwitchMLP(
            expert_input_dim,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=nn.ReLU2(),
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_shared_expert_intermediate_size
            self.shared_experts = NemotronHMLP(
                config, intermediate_size=intermediate_size
            )

        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(
                config.hidden_size, config.moe_latent_size, bias=config.mlp_bias
            )
            self.fc2_latent_proj = nn.Linear(
                config.moe_latent_size, config.hidden_size, bias=config.mlp_bias
            )

    def __call__(self, x):
        residuals = x
        exp = self.moe_expand
        if (
            exp is not None
            and exp.n > self.num_experts_per_tok
            and exp.in_range(self.layer_idx)
        ):
            inds, scores = self.gate(x, exp.n, exp.t, exp.d)
        else:
            inds, scores = self.gate(x)

        if self.moe_latent_size is not None:
            x = self.fc1_latent_proj(x)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        if self.moe_latent_size is not None:
            y = self.fc2_latent_proj(y)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(residuals)

        return y


class NemotronHBlock(nn.Module):
    def __init__(self, args: ModelConfig, block_type: str, layer_idx: int = 0):
        super().__init__()
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

        self.block_type = block_type

        if self.block_type == "M":
            self.mixer = NemotronHMamba2Mixer(args)
        elif self.block_type == "*":
            self.mixer = NemotronHAttention(args)
        elif self.block_type == "-":
            self.mixer = NemotronHMLP(args)
        elif self.block_type == "E":
            self.mixer = NemotronHMoE(args, layer_idx)

    def __call__(
        self,
        x,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        capture_sink: Optional[list] = None,
    ):
        hidden_states = self.norm(x)
        if self.block_type == "M":
            hidden_states = self.mixer(
                hidden_states, mask=mask, cache=cache, capture_sink=capture_sink
            )
        elif self.block_type == "*":
            hidden_states = self.mixer(hidden_states, mask=mask, cache=cache)
        else:
            hidden_states = self.mixer(hidden_states)

        return x + hidden_states


class NemotronHModel(nn.Module):
    def __init__(self, args: ModelConfig, with_embeddings: bool = True):
        super().__init__()
        self.with_embeddings = with_embeddings
        if with_embeddings:
            self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            NemotronHBlock(args, block_type, layer_idx=i)
            for i, block_type in enumerate(args.hybrid_override_pattern)
        ]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.fa_idx = 0
        self.ssm_idx = 0
        for b in args.hybrid_override_pattern:
            if b == "*":
                break
            elif b == "M":
                self.fa_idx += 1
        for b in args.hybrid_override_pattern:
            if b == "*":
                self.ssm_idx += 1
            elif b == "M":
                break

    def __call__(
        self,
        inputs=None,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        hidden_sink: Optional[list] = None,
        skip_final_norm: bool = False,
        recurrent_sink: Optional[list] = None,
    ):
        # Fork: upstream rejects both-supplied as well as neither-supplied
        # (`if (inputs is None) == (inputs_embeds is None)`). Only neither is
        # unserviceable -- the branch immediately below already handles
        # both-supplied by preferring inputs_embeds. Rejecting it broke the
        # shared AR loop, which passes both: generate/ar.py calls
        # `model.language_model(y, inputs_embeds=inputs_embeds, ...)` because
        # for a VLM inputs_embeds is the vision-merged embedding while the
        # token ids are still carried alongside. That made `mlx_vlm.generate`
        # raise on every nemotron_h model while the server path (which passes
        # inputs_embeds only) worked, so the AR loop is right and this guard
        # was the outlier.
        if inputs is None and inputs_embeds is None:
            raise ValueError("Provide inputs or inputs_embeds")
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif self.with_embeddings:
            hidden_states = self.embeddings(inputs)
        else:
            raise ValueError("This Nemotron-H backbone has no token embedding table")

        if cache is None:
            cache = [None] * len(self.layers)

        # Fork: shared tail for every exit path below (`hidden_sink`/
        # `skip_final_norm` support the MTP speculative drafter
        # (nemotron_h_mtp), which needs the pre-final-norm hidden state --
        # the target's `norm_f` is a SEPARATE parameter from the MTP head's
        # own `final_layernorm`, so capturing post-norm hidden here would
        # double-normalize whatever the drafter feeds forward. Mirrors the
        # `hidden_sink`/`skip_final_norm` contract other MTP-capable
        # backbones (deepseek_v4, qwen3_5) already expose.
        def _finish(hidden_states):
            if hidden_sink is not None:
                hidden_sink.append(hidden_states)
            if skip_final_norm:
                return hidden_states
            return self.norm_f(hidden_states)

        # Fork: `recurrent_sink` drives the MTP speculative-verify rollback
        # contract (mlx_vlm/models/recurrent_rollback.py). Each mamba2
        # layer's `_conv`/`_ssm` (models/ssm.py's `ssm_update_with_states`)
        # now emits the per-position `(conv_state, ssm_state)` snapshot
        # directly from ONE forward -- one Metal launch per layer, not one
        # per verify-block position -- via the `capture_sink` list threaded
        # through the loop below (see NemotronHMamba2Mixer.__call__).
        # Exactness comes from the kernel/ops-twin applying the SAME
        # single-step recurrence, in the SAME order, that a normal
        # one-token-at-a-time decode would; see
        # `mlx_vlm/models/ssm.py:ssm_update_with_states`'s docstring and
        # `test_ssm_with_states.py` for the kernel-vs-single-step-loop check
        # that pins this down. See recurrent_rollback.py's module docstring
        # for the full rationale.
        has_attention = any(layer.block_type == "*" for layer in self.layers)
        has_mamba = any(layer.block_type == "M" for layer in self.layers)
        attn_cache = cache[self.fa_idx] if has_attention else None
        ssm_cache = cache[self.ssm_idx] if has_mamba else None
        attn_mask = create_attention_mask(hidden_states, attn_cache)
        ssm_mask = create_ssm_mask(hidden_states, ssm_cache)

        cache_counter = 0
        for layer in self.layers:
            if layer.block_type == "M" or layer.block_type == "*":
                c = cache[cache_counter]
                cache_counter += 1
            else:
                c = None

            if layer.block_type == "*":
                mask = attn_mask
            else:
                mask = ssm_mask

            capture_sink_for_layer = None
            if recurrent_sink is not None and layer.block_type == "M":
                capture_sink_for_layer = []
            hidden_states = layer(
                hidden_states, mask=mask, cache=c, capture_sink=capture_sink_for_layer
            )
            if capture_sink_for_layer is not None:
                recurrent_sink[cache_counter - 1] = capture_sink_for_layer

        return _finish(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.model_type = args.model_type

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.backbone(inputs, cache=cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.block_type == "M":
                caches.append(ArraysCache(size=2))
            elif layer.block_type == "*":
                caches.append(KVCache())
        return caches

    def sanitize(self, weights):
        weights = {k: v for (k, v) in weights.items() if not k.startswith("mtp.")}
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"backbone.layers.{layer_idx}.mixer"
            for m, n in [("down_proj", "fc2"), ("up_proj", "fc1")]:
                if f"{prefix}.experts.0.{m}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.experts.{e}.{m}.weight")
                        for e in range(self.args.n_routed_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

        return weights

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k and "A_log" not in k

        return predicate


# Fork: `RecurrentStateRollbackMixin` supplies the MTP speculative-verify
# rollback contract (rollback_speculative_cache / speculative_verify_logits /
# speculative_verify_hidden / speculative_logits_from_hidden /
# speculative_argmax_from_hidden); see recurrent_rollback.py's module
# docstring for why that logic lives outside this file.
class LanguageModel(RecurrentStateRollbackMixin, nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        # Fork: `return_hidden`/`return_shared_kv`/`skip_logits`/
        # `skip_final_norm` are the generic MTP-verify contract
        # (speculative/mtp.py's `_mtp_verify_target` fallback calls
        # `lm(verify_input, cache=prompt_cache, return_hidden=True,
        # return_shared_kv=True)` and reads `out.hidden_states[-1]`) that the
        # nemotron_h_mtp drafter needs. nemotron_h has no MLA/GQA cache to
        # share across layers the way deepseek_v4 does, so
        # `shared_kv_states` is always an empty dict rather than populated --
        # the drafter's `set_shared_kv` ignores its contents regardless.
        if inputs is None:
            inputs = kwargs.get("input_ids")
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        skip_logits = kwargs.pop("skip_logits", False)
        skip_final_norm = kwargs.pop("skip_final_norm", False)
        hidden_sink = kwargs.pop("hidden_sink", None)
        if return_hidden and hidden_sink is None:
            hidden_sink = []

        # Fork: `capture_recurrent_states` is the other half of the MTP
        # speculative-verify rollback contract
        # (models/recurrent_rollback.py's RecurrentStateRollbackMixin,
        # mixed into this class above). It asks the backbone to record every
        # mamba2 layer's (conv_state, ssm_state) after each position of this
        # call, aligned index-for-index with `cache`, so
        # `rollback_speculative_cache` can restore any accepted position
        # exactly.
        capture_recurrent_states = kwargs.pop("capture_recurrent_states", False)
        recurrent_sink = None
        if capture_recurrent_states:
            if cache is None:
                raise ValueError(
                    "capture_recurrent_states requires an existing prompt_cache"
                )
            recurrent_sink = [None] * len(cache)

        out = self.backbone(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            hidden_sink=hidden_sink,
            skip_final_norm=skip_final_norm,
            recurrent_sink=recurrent_sink,
        )
        logits = None if skip_logits else self.lm_head(out)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            shared_kv_states={} if return_shared_kv else None,
            gdn_states=recurrent_sink,
        )

    # Fork: RecurrentStateRollbackMixin's speculative_logits_from_hidden
    # calls this before lm_head. hidden_sink (above) deliberately captures
    # PRE-norm_f hidden -- the MTP drafter needs it -- but the model's real
    # logits are lm_head(norm_f(hidden)) (see Model.__call__ above). Without
    # this override, speculative_logits_from_hidden/_argmax_from_hidden
    # would silently use un-normed hidden, diverging from the real forward
    # and corrupting MTP's target-token sampling and acceptance comparison.
    def speculative_final_norm(self, hidden: mx.array) -> mx.array:
        return self.backbone.norm_f(hidden)

    def sanitize(self, weights):
        return Model.sanitize(self, weights)

    @property
    def cast_predicate(self):
        return Model.cast_predicate.fget(self)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        return Model.make_cache(self)

    def set_moe_expansion(
        self, exp: Optional[MoeExpansion], strict: bool = False
    ) -> int:
        """Set (or clear, with `None`) M34 expert-budget expansion on every
        `NemotronHMoE` block ("E" layers). Returns the number of MoE layers
        that ACTUALLY get expanded -- absolute index inside `exp`'s layer
        range AND `exp.n > that layer's num_experts_per_tok` (0 if `exp` is
        None). A layer in range with `exp.n <= num_experts_per_tok` is a
        native-path no-op (see `NemotronHMoE.__call__`) and is not counted.
        Only ever touches `self`'s own layers -- a bound MTP drafter is a
        separate `nn.Module`, never reachable from here.

        `strict=True` (used by `apply_moe_expansion`, the CLI/chat entry
        point) raises `ValueError` if the range contains MoE blocks but none
        of them get expanded -- every one would be a silent no-op. Direct
        callers (e.g. tests exercising the deliberate `N == K` native
        passthrough) default to `strict=False` and never raise for this."""
        count = 0
        in_range_total = 0
        top_k_seen = None
        for layer in self.backbone.layers:
            if layer.block_type != "E":
                continue
            layer.mixer.moe_expand = exp
            if exp is not None and exp.in_range(layer.mixer.layer_idx):
                in_range_total += 1
                top_k_seen = layer.mixer.num_experts_per_tok
                if exp.n > layer.mixer.num_experts_per_tok:
                    count += 1
        if strict and exp is not None and in_range_total > 0 and count == 0:
            raise ValueError(
                f"moe_expand n={exp.n} does not exceed native top_k="
                f"{top_k_seen} on any layer in range "
                f"{exp.layers[0]}-{exp.layers[1]}"
            )
        return count
