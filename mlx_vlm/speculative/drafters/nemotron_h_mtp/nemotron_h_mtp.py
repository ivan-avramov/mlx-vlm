from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ....models.base import create_attention_mask
from ....models.cache import KVCache
from ....models.nemotron_h.language import NemotronHBlock
from ... import mtp_profile
from .config import NemotronHMTPConfig


class NemotronHMTPDraftModel(nn.Module):
    """Native MTP drafter for the `nemotron_h` (hybrid Mamba/attention/MoE) family.

    Nemotron-H's MTP head is DeepSeek-style (`enorm`/`hnorm`/`eh_proj` feeding a
    small decoder stack that ends in its own final norm, then the TARGET's
    shared `lm_head`) but, unlike DeepSeek-V4 or Qwen3.5, its decoder stack is
    not one homogeneous layer repeated `mtp_num_hidden_layers` times -- the
    checkpoint carries `mtp.layers.<i>.*` where each index has its own
    `layers_block_type`-style block type. The only inventory observed so far
    (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`) is
    `mtp_layers_block_type: ["attention", "moe"]` -- one full-attention layer
    (no RoPE: matches the target's own `NemotronHAttention`, which applies
    none either) followed by one MoE layer with a shared expert, exactly
    mirroring the target backbone's own "*"/"E" block sub-layers. This class
    builds that stack out of the SAME `NemotronHBlock` the target backbone
    uses, so the mixer implementations (and their weight-name contracts) are
    shared by construction rather than re-derived.
    """

    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True

    def __init__(self, config: NemotronHMTPConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("NemotronHMTPConfig.text_config must be set")

        self.args = text_config
        hidden_size = text_config.hidden_size
        eps = text_config.layer_norm_epsilon

        self.enorm = nn.RMSNorm(hidden_size, eps=eps)
        self.hnorm = nn.RMSNorm(hidden_size, eps=eps)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self._block_kinds = config.block_type_chars()
        if "M" in self._block_kinds:
            raise NotImplementedError(
                "nemotron_h_mtp: a mamba sub-layer in the MTP head is not "
                "supported. The only checkpoint inventoried so far carries "
                "['attention', 'moe'], and a mamba sub-layer would need "
                "ArraysCache + create_ssm_mask plumbing this class does not "
                "have (untested against any real weights)."
            )
        self.layers = [NemotronHBlock(text_config, kind) for kind in self._block_kinds]
        self.final_layernorm = nn.RMSNorm(hidden_size, eps=eps)

        self._input_embed = None
        self._lm_head_fn = None
        self._cache: List[KVCache] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._next_position: Any = 0
        self._round_appended = 0
        self._kv_valid_len: Any = 0
        self._position: Any = 0
        self._draft_round = 0

        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model) -> "NemotronHMTPDraftModel":
        inner = None
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model

        if inner is not None:
            self._input_embed = inner.embed_tokens
        else:
            # nemotron_h has no `embed_tokens` anywhere in the chain above --
            # its backbone names the embedding table `embeddings` (see
            # NemotronHModel in models/nemotron_h/language.py).
            lm = getattr(target_model, "language_model", target_model)
            backbone = getattr(lm, "backbone", None)
            embeddings = getattr(backbone, "embeddings", None)
            if embeddings is None:
                raise AttributeError(
                    f"Cannot find embed_tokens or backbone.embeddings in "
                    f"{type(target_model).__name__}"
                )
            self._input_embed = embeddings

        lm = getattr(target_model, "language_model", target_model)
        self._lm_head_fn = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self._input_embed.as_linear
        )
        return self

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for kind in self._block_kinds if kind == "*"]

    def reset(self, target_model) -> List[KVCache]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self._draft_round = 0
        self._cache = self.make_cache()
        self._seed_token = None
        self._seed_hidden = None
        self._next_position = 0
        self._round_appended = 0
        return self._cache

    def draft_eval_state(self):
        state = [self._seed_token, self._seed_hidden]
        for cache in self._cache:
            state.append(cache.state)
        return state

    def set_shared_kv(
        self,
        shared_kv_states: dict,
        kv_offset,
        position=None,
        kv_valid_len=None,
        left_padding=None,
    ) -> None:
        del shared_kv_states, left_padding
        if kv_valid_len is None:
            kv_valid_len = kv_offset
        if position is None:
            position = kv_valid_len
        self._kv_valid_len = kv_valid_len
        self._position = position
        if not self._cache or self._cache[0].offset == 0:
            self._next_position = kv_valid_len

    def _forward_hidden(
        self,
        token_embed: mx.array,
        hidden: mx.array,
        cache: Optional[List[KVCache]],
    ) -> mx.array:
        h = self.eh_proj(
            mx.concatenate([self.enorm(token_embed), self.hnorm(hidden)], axis=-1)
        )

        cache = cache or []
        cache_ptr = 0
        mask = None
        for layer, kind in zip(self.layers, self._block_kinds):
            if kind == "*":
                layer_cache = cache[cache_ptr] if cache_ptr < len(cache) else None
                cache_ptr += 1
                if mask is None:
                    mask = (
                        create_attention_mask(h, layer_cache)
                        if layer_cache is not None
                        else ("causal" if h.shape[1] > 1 else None)
                    )
                h = layer(h, mask=mask, cache=layer_cache)
            else:
                h = layer(h)
        return self.final_layernorm(h)

    def _forward_tokens(
        self,
        tokens: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
    ) -> mx.array:
        token_embed = self._input_embed(tokens.astype(token_dtype))
        h = self._forward_hidden(token_embed, hidden[:, : tokens.shape[1], :], self._cache)
        steps = int(tokens.shape[1])
        self._next_position = (
            self._next_position + steps
            if isinstance(self._next_position, int)
            else self._next_position + steps
        )
        return h

    def _forward_token(
        self,
        tok: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
    ) -> mx.array:
        return self._forward_tokens(tok, hidden, token_dtype)

    def _set_seed_from_hidden(self, hidden: mx.array, sampler, greedy: bool) -> None:
        logits = self._lm_head_fn(hidden)
        self._seed_token = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
        self._seed_hidden = hidden

    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus_token,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        if input_ids.shape[1] == 0:
            return
        if isinstance(bonus_token, int):
            bonus = mx.array([[bonus_token]], dtype=token_dtype)
        else:
            bonus = bonus_token[:, None].astype(token_dtype)

        shifted = mx.concatenate([input_ids[:, 1:].astype(token_dtype), bonus], axis=1)
        self._next_position = 0
        h = self._forward_tokens(
            shifted,
            hidden[:, : shifted.shape[1], :],
            token_dtype,
        )
        self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)

    def accept_verified_tokens(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: int,
        new_tokens: List[int],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        keep_appended = min(int(accepted), self._round_appended)
        trim = self._round_appended - keep_appended
        if trim > 0:
            for cache in self._cache:
                cache.trim(trim)
            self._next_position = (
                self._next_position - trim
                if isinstance(self._next_position, int)
                else self._next_position - trim
            )

        token_chunks = []
        hidden_chunks = []
        for draft_idx in range(keep_appended, int(accepted)):
            token_chunks.append(draft_tokens[:, draft_idx : draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx : draft_idx + 1, :])

        if new_tokens:
            token_chunks.append(mx.array([[int(new_tokens[-1])]], dtype=token_dtype))
            hidden_chunks.append(verify_hidden[:, int(accepted) : int(accepted) + 1, :])

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            h = self._forward_tokens(tokens, hiddens, token_dtype)
            self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)
        self._round_appended = 0

    def accept_verified_tokens_batch(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: List[int],
        new_tokens: List[List[int]],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        """Extend the Nemotron-H MTP drafter cache after batched verify."""
        if len(accepted) <= 1:
            self.accept_verified_tokens(
                verify_hidden,
                draft_tokens,
                int(accepted[0]),
                new_tokens[0],
                sampler,
                token_dtype,
                greedy,
            )
            return

        accepted_set = {int(a) for a in accepted}
        if len(accepted_set) != 1:
            raise ValueError(
                "Nemotron-H MTP batched cache update requires uniform acceptance."
            )
        accepted_i = accepted_set.pop()

        keep_appended = min(accepted_i, self._round_appended)
        trim = self._round_appended - keep_appended
        if trim > 0:
            for cache in self._cache:
                cache.trim(trim)
            self._next_position = (
                self._next_position - trim
                if isinstance(self._next_position, int)
                else self._next_position - trim
            )

        token_chunks = []
        hidden_chunks = []
        for draft_idx in range(keep_appended, accepted_i):
            token_chunks.append(draft_tokens[:, draft_idx : draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx : draft_idx + 1, :])

        if all(new_tokens):
            bonus = mx.array(
                [[int(row_tokens[-1])] for row_tokens in new_tokens],
                dtype=token_dtype,
            )
            token_chunks.append(bonus)
            hidden_chunks.append(verify_hidden[:, accepted_i : accepted_i + 1, :])

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            h = self._forward_tokens(tokens, hiddens, token_dtype)
            self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)
        self._round_appended = 0

    def filter_batch(self, keep) -> None:
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)

        for cache in self._cache:
            if cache.keys is not None:
                cache.keys = cache.keys[keep]
                cache.values = cache.values[keep]

        if self._seed_token is not None:
            self._seed_token = self._seed_token[keep]
        if self._seed_hidden is not None:
            self._seed_hidden = self._seed_hidden[keep]

        for attr in ("_next_position", "_kv_valid_len", "_position"):
            value = getattr(self, attr)
            if isinstance(value, mx.array) and value.ndim > 0 and value.size > 1:
                setattr(self, attr, value[keep])

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> mx.array:
        del cache
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block() "
                "so the drafter can use the target embeddings and LM head."
            )

        # M29 H1: env-gated per-draft-token profiler (MLX_VLM_MTP_PROFILE_HEAD).
        # ``hp`` is None unless the env var is set -- zero extra calls when
        # unset. draft_block has no end-of-generation signal to report from,
        # so the profiler is a module singleton the round loop's final report
        # flushes (mtp_profile.MTPRoundProfiler._after_report).
        hp = mtp_profile.head_profiler_from_env()
        if hp is not None:
            hp.begin()

        if isinstance(last_bonus, int):
            tok = mx.array([[last_bonus]], dtype=token_dtype)
        else:
            tok = last_bonus[:, None].astype(token_dtype)

        h_prev = hidden
        tokens: List[mx.array] = []
        self._round_appended = 0

        if self._seed_token is not None and self._seed_hidden is not None:
            tok = self._seed_token.astype(token_dtype)
            h_prev = self._seed_hidden
            tokens.append(tok)
            self._seed_token = None
            self._seed_hidden = None

        while len(tokens) < block_size - 1:
            h_prev = self._forward_token(tok, h_prev, token_dtype)
            if hp is not None:
                hp.mark("proj_layers", h_prev)
            self._round_appended += 1
            logits = self._lm_head_fn(h_prev)
            if hp is not None:
                hp.mark("lm_head", logits)
            tok = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
            if hp is not None:
                hp.mark("sampler")
            tokens.append(tok)
            if hp is not None:
                hp.mark("eval", tok)
                hp.end_unit()

        self._draft_round += 1
        return mx.concatenate(tokens, axis=1)

    def sanitize(self, weights: dict) -> dict:
        """Map the checkpoint's `mtp.layers.<i>.*` layout onto this module's.

        Idempotent: called both on the raw HF `mtp.*` tensors (via
        `split.py`, before quantization) and on this drafter's own saved
        sidecar (via the normal `load_model` path, where the keys are
        already in final form and every branch below is a no-op).
        """
        weights = dict(weights)

        stripped = {}
        for key, value in weights.items():
            if key.startswith("mtp."):
                key = key[len("mtp.") :]
            stripped[key] = value
        weights = stripped

        # eh_proj/enorm/hnorm live under the first mtp layer's HF namespace
        # but are top-level modules here -- they run once, before the block
        # stack, not per-layer.
        for name in ("eh_proj.weight", "enorm.weight", "hnorm.weight"):
            old_key = f"layers.0.{name}"
            if old_key in weights:
                weights[name] = weights.pop(old_key)

        # The MTP head's own output norm is attached to whichever layer
        # index HF marks last (observed: `layers.1.final_layernorm`), but it
        # runs once at the end of the block stack, not inside that block.
        for key in list(weights):
            if key.endswith(".final_layernorm.weight"):
                weights["final_layernorm.weight"] = weights.pop(key)

        n_routed_experts = getattr(self.args, "n_routed_experts", None)
        layer_indices = sorted(
            {
                int(key.split(".")[1])
                for key in weights
                if key.startswith("layers.") and ".mixer.experts." in key
            }
        )
        for layer_idx in layer_indices:
            prefix = f"layers.{layer_idx}.mixer"
            if f"{prefix}.experts.0.up_proj.weight" not in weights:
                continue
            n_experts = n_routed_experts
            if n_experts is None:
                n_experts = (
                    max(
                        int(key.split(".")[3])
                        for key in weights
                        if key.startswith(f"{prefix}.experts.")
                    )
                    + 1
                )
            for src, dst in (("up_proj", "fc1"), ("down_proj", "fc2")):
                expert_keys = [
                    f"{prefix}.experts.{e}.{src}.weight" for e in range(n_experts)
                ]
                missing = [key for key in expert_keys if key not in weights]
                if missing:
                    raise ValueError(
                        f"nemotron_h_mtp sanitize: missing expert tensors "
                        f"{missing[:3]}: n_routed_experts={n_experts} expects "
                        f"experts 0..{n_experts - 1} for {src}."
                    )
                weights[f"{prefix}.switch_mlp.{dst}.weight"] = mx.stack(
                    [weights.pop(key) for key in expert_keys]
                )

        return weights
