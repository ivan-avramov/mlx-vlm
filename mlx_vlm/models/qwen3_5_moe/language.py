from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..moe_expand import MoeExpansion, expand_route  # Fork: M34 opt-in routing
from ..qwen3_5.language import LanguageModel as Qwen3_5LanguageModel
from ..qwen3_5.language import Qwen3_5Attention as Qwen3_5MoeAttention
from ..qwen3_5.language import Qwen3_5GatedDeltaNet as Qwen3_5MoeGatedDeltaNet
from ..qwen3_5.language import Qwen3_5MLP as Qwen3_5MoeMLP
from ..qwen3_5.language import Qwen3_5Model
from ..switch_layers import SwitchGLU
from .config import ModelConfig, TextConfig


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    # Fork: M34 layer-scoped expert expansion; default route remains native.
    def __init__(self, args: TextConfig, layer_idx: int = 0):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        shared_expert_intermediate_size = args.shared_expert_intermediate_size

        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.layer_idx = layer_idx
        # M34: layer-scoped expert-budget expansion. None == native top-K
        # everywhere (byte-identical to upstream); set via
        # `LanguageModel.set_moe_expansion`, never touched otherwise.
        self.moe_expand: Optional[MoeExpansion] = None

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

        self.shared_expert = Qwen3_5MoeMLP(dim, shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def _shared_expert_scale(self, x: mx.array) -> mx.array:
        return mx.sigmoid(self.shared_expert_gate(x))

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        exp = self.moe_expand
        if exp is not None and exp.n > k and exp.in_range(self.layer_idx):
            inds, scores = expand_route(gates, k, exp.n, exp.t, exp.d)
        else:
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = self._shared_expert_scale(x) * shared_y

        return y + shared_y


class Qwen3_5MoeDecoderLayer(nn.Module):
    # Fork: M34 layer-scoped expert expansion; default route remains native.
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(args)
        else:
            self.self_attn = Qwen3_5MoeAttention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = Qwen3_5MoeSparseMoeBlock(args, layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(
                self.input_layernorm(x),
                mask,
                cache,
            )
        else:
            r = self.self_attn(
                self.input_layernorm(x),
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5MoeModel(Qwen3_5Model):

    def __init__(self, args: TextConfig):
        nn.Module.__init__(self)
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3_5MoeDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1


class LanguageModel(Qwen3_5LanguageModel):
    # Fork: M34 layer-scoped expert expansion; default route remains native.

    def __init__(self, args: TextConfig, config: ModelConfig = None):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen3_5MoeModel(args)
        self._rope_deltas = None
        self._position_ids = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def set_moe_expansion(
        self, exp: Optional[MoeExpansion], strict: bool = False
    ) -> int:
        """Set (or clear, with `None`) M34 expert-budget expansion on every MoE
        block. Returns the number of layers that ACTUALLY get expanded --
        absolute index inside `exp`'s layer range AND `exp.n > that layer's
        top_k` (0 if `exp` is None). A layer in range with `exp.n <= top_k` is
        a native-path no-op (see `Qwen3_5MoeSparseMoeBlock.__call__`) and is
        not counted. Only ever touches `self`'s own layers -- a bound MTP
        drafter is a separate `nn.Module`, never reachable from here.

        `strict=True` (used by `apply_moe_expansion`, the CLI/chat entry
        point) raises `ValueError` if the range contains MoE blocks but none
        of them get expanded -- every one would be a silent no-op. Direct
        callers (e.g. tests exercising the deliberate `N == K` native
        passthrough) default to `strict=False` and never raise for this."""
        count = 0
        in_range_total = 0
        top_k_seen = None
        for layer in self.model.layers:
            layer.mlp.moe_expand = exp
            if exp is not None and exp.in_range(layer.mlp.layer_idx):
                in_range_total += 1
                top_k_seen = layer.mlp.top_k
                if exp.n > layer.mlp.top_k:
                    count += 1
        if strict and exp is not None and in_range_total > 0 and count == 0:
            raise ValueError(
                f"moe_expand n={exp.n} does not exceed native top_k="
                f"{top_k_seen} on any layer in range "
                f"{exp.layers[0]}-{exp.layers[1]}"
            )
        return count
