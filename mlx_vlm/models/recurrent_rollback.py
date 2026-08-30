"""Generic MTP speculative-verify rollback for hybrid recurrent language models.

Fork-only module -- upstream has no MTP speculative decoding at all, so there is
no upstream counterpart to diverge from or stay in sync with.

Why this lives here and not in ``models/nemotron_h/language.py``
------------------------------------------------------------------
``mlx_vlm/speculative/mtp.py`` defines a contract every MTP-capable target
``LanguageModel`` must implement: ``speculative_verify_logits``,
``speculative_verify_hidden``, ``speculative_logits_from_hidden``,
``speculative_argmax_from_hidden`` and ``rollback_speculative_cache`` (see
``mlx_vlm/models/qwen3_5/language.py`` for the reference GatedDeltaNet
implementation). For a HYBRID RECURRENT model -- fixed-size recurrent state per
layer (mamba2's ``(conv_state, ssm_state)`` here; qwen3_5's GDN state there)
interleaved with plain attention KV layers -- most of this contract is
architecture-agnostic: how you slice ``accepted``/``block_size``, how you trim
a trimmable KV cache, how you fail loud when a batch can't be rolled back
exactly. Only *producing* the per-position recurrent-state snapshots requires
reaching into the model's own forward pass, and only re-applying the model's
final norm (see ``speculative_final_norm`` below) is architecture-specific.

Putting the reusable half in this mixin -- rather than duplicating it inside
``mlx_vlm/models/nemotron_h/language.py`` -- means a future upstream sync of
that file only has to reconcile the couple of small ``# Fork:`` hunks that
call into this module, not a few hundred lines of rollback logic. It also
means the next hybrid-recurrent model (mamba2 or otherwise) that wants this
contract can mix this class in directly instead of re-deriving it. It lives
under ``mlx_vlm/models/`` (an otherwise-empty package, so no import-cycle risk)
rather than under ``mlx_vlm/speculative/`` because ``speculative/__init__.py``
pulls in the drafter registry, and a drafter (``nemotron_h_mtp``) imports
*from* ``models/nemotron_h/language.py`` -- so a model file importing anything
under ``speculative/`` risks a cycle the moment that registry starts eagerly
importing drafters.

Why replaying the verify block one token at a time is exact
-------------------------------------------------------------
Unlike qwen3_5's GatedDeltaNet, nemotron_h's mamba2 mixer has no dedicated
verify-time kernel that exposes intermediate per-position states. Instead,
``models/ssm.py:ssm_update`` already dispatches, per call, either an exact
chunked scan (``ssm_attn``, used whenever ``seq_len > 1``, or off-GPU, or with
no prior state) or -- for a single new token against existing state, on GPU
with Metal available -- a dedicated single-step ``ssm_update_kernel``
(``models/ssm.py`` ~211-232). Both of those are exactly what ordinary
one-token-at-a-time decoding already uses; nothing is special-cased for
verify. So replaying a (tiny, <=~4-token) verify block one position at a time
through the model's own ``__call__`` against the same evolving cache runs the
IDENTICAL code path -- and, on GPU, the identical kernel -- that a normal
decode step would, rather than some alternate "replay" algorithm that merely
happens to agree with it. THAT structural identity, not chunk-size invariance
per se, is why the captured ``(conv_state, ssm_state)`` after each replayed
position is exact: ``ssm_attn``'s chunked scan is in fact chunk-invariant too,
but that isn't load-bearing here, because the replay never runs it at
width > 1. The model side of this
(``mlx_vlm/models/nemotron_h/language.py``'s ``recurrent_sink`` plumbing)
just drives the replay and hands the snapshots back here.

Note: this fork's CPU-pinned test suite
(``mlx_vlm/tests/test_nemotron_h_rollback.py``) only exercises the
``ssm_attn`` chunked-scan branch -- CPU never dispatches
``ssm_update_kernel`` (see the ``seq_len``/device guard in
``models/ssm.py:ssm_update``). GPU-kernel-path exactness for this replay is
therefore unverified by that suite and would need a GPU-side check to
confirm.
"""

from typing import Any, List, Optional, Sequence, Union

import mlx.core as mx

__all__ = ["RecurrentStateRollbackMixin"]


def _to_accepted_list(accepted: Union[int, Sequence[int], mx.array]) -> List[int]:
    if isinstance(accepted, int):
        return [accepted]
    if isinstance(accepted, mx.array):
        return [int(x) for x in accepted.reshape(-1).tolist()]
    return [int(x) for x in accepted]


def _is_recurrent_cache(cache_entry: Any) -> bool:
    """True for a fixed-size-recurrent-state cache (e.g. mamba2's ArraysCache).

    Mirrors qwen3_5's ``_is_ssm_cache`` predicate: a cache that is neither
    trimmable (plain/rotating KV caches are) nor row-zeroable (some batched KV
    caches expose ``zero_row_tail``) is, by elimination, a recurrent-state
    cache whose ``[0]``/``[1]`` slots hold the per-layer state arrays.
    """
    return not cache_entry.is_trimmable() and not hasattr(cache_entry, "zero_row_tail")


class RecurrentStateRollbackMixin:
    """Mixin implementing the MTP speculative-verify contract for a hybrid
    recurrent-attention ``LanguageModel``.

    Assumes the host class:
      * is callable as ``self(inputs, cache=..., return_hidden=True,
        return_shared_kv=True, skip_logits=..., capture_recurrent_states=True)``
        and returns a ``LanguageModelOutput`` whose ``gdn_states`` is a list
        aligned index-for-index with the ``cache`` list passed in -- ``None``
        at non-recurrent-layer positions, and at each recurrent-layer position
        a list of ``(conv_state, ssm_state)`` snapshots, one per input
        position, captured under ``capture_recurrent_states=True``;
      * returns a *pre-final-norm* hidden state from that call (nemotron_h's
        ``hidden_sink`` does, deliberately -- the MTP drafter needs pre-norm
        hidden too), and exposes a plain ``self.lm_head`` linear/callable, so
        this mixin's own ``speculative_logits_from_hidden`` must re-apply the
        final norm itself via ``speculative_final_norm`` (identity by
        default) before calling ``lm_head`` -- override it when the host's
        hidden_sink is pre-norm, matching the model's real
        ``lm_head(final_norm(hidden))`` logits.
    """

    def speculative_final_norm(self, hidden: mx.array) -> mx.array:
        """Applied to a pre-final-norm hidden before ``lm_head`` below.

        Identity by default (matches a host whose ``hidden_sink`` is already
        post-norm, e.g. qwen3_5). A host that captures PRE-norm hidden (e.g.
        nemotron_h, whose drafter needs the pre-norm state) must override
        this to apply its own final norm -- otherwise
        ``speculative_logits_from_hidden``/``speculative_argmax_from_hidden``
        silently diverge from the model's real ``lm_head(final_norm(hidden))``
        logits, corrupting MTP's target-token sampling and acceptance
        comparison.
        """
        return hidden

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self.lm_head(self.speculative_final_norm(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, inputs: mx.array, cache: List[Any]):
        out = self(
            inputs,
            cache=cache,
            return_hidden=True,
            return_shared_kv=True,
            skip_logits=True,
            capture_recurrent_states=True,
        )
        return out.hidden_states[-1], out.shared_kv_states, out.gdn_states

    def speculative_verify_logits(self, inputs: mx.array, cache: List[Any], sampler):
        out = self(
            inputs,
            cache=cache,
            return_hidden=True,
            return_shared_kv=True,
            capture_recurrent_states=True,
        )
        return (
            out.hidden_states[-1],
            out.shared_kv_states,
            out.gdn_states,
            sampler(out.logits),
        )

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        states: Optional[List[Any]],
        accepted: Union[int, Sequence[int], mx.array],
        block_size: int,
    ) -> int:
        """Restore ``caches`` to the state after the last ACCEPTED token.

        ``accepted``/``block_size`` semantics match qwen3_5's/gemma4's hook:
        ``accepted`` is the 0-indexed position of the last accepted draft
        token *within* the just-verified ``block_size``-token block (so
        ``n = accepted + 1`` tokens of the block are kept). For a batch,
        ``accepted`` is one value per row.

        Recurrent-state (mamba2) caches are restored by selecting, per row,
        the snapshot captured after that row's ``accepted`` position --
        ``states`` must be the ``gdn_states`` list returned by this mixin's
        own ``speculative_verify_hidden``/``speculative_verify_logits``
        (aligned with ``caches``; see the class docstring). Trimmable KV
        caches (attention layers) are simply trimmed by
        ``block_size - n``.

        Only ``batch_size == 1`` or a batch that uniformly accepted the same
        count is supported: nemotron_h's attention layers use a plain
        ``KVCache`` with a single scalar ``offset`` shared by every row, so a
        batch with differing per-row acceptance has no way to represent each
        row's true valid length without corrupting rejected-token K/V into
        positions later attention would still attend to. Rather than produce
        silently-wrong state, that case raises.
        """
        accepted_list = _to_accepted_list(accepted)
        if len(set(accepted_list)) > 1:
            raise NotImplementedError(
                "rollback_speculative_cache: nemotron_h's plain KVCache has no "
                "per-row valid-length tracking (no prepare()/right_padding/"
                "zero_row_tail), so a batch whose rows accepted different "
                f"counts (accepted={accepted_list}) cannot be rolled back "
                "exactly. Only batch_size==1 or uniform acceptance across the "
                "batch is supported."
            )
        max_a = accepted_list[0]
        n = max_a + 1
        if n > block_size:
            raise ValueError(
                f"accepted={max_a} implies {n} accepted tokens, exceeding "
                f"block_size={block_size}"
            )
        trim = block_size - n
        batch_size = len(accepted_list)

        # Fork: fail EARLY and by NAME when the caller never captured
        # recurrent state at all, rather than letting the loop below raise a
        # per-cache-index "missing snapshot" error on whichever recurrent
        # layer happens to be enumerated first. `states is None` means verify
        # ran without `capture_recurrent_states` -- which currently only
        # happens for a non-mtp draft_kind (dflash/eagle3/suffix), none of
        # which route through this mixin's `speculative_verify_hidden`/
        # `_logits`. `hasattr(lm, "rollback_speculative_cache")` alone can't
        # tell mtp.py's callers that up front, so this has to be the backstop.
        if states is None and any(
            c is not None and _is_recurrent_cache(c) for c in caches
        ):
            raise RuntimeError(
                "rollback_speculative_cache: called with states=None, but "
                "this target has recurrent (mamba2) cache layers that need "
                "per-position state to roll back. verify ran without "
                "capture_recurrent_states -- this target only supports "
                "draft_kind=mtp (whose verify calls "
                "speculative_verify_hidden/_logits with "
                "capture_recurrent_states=True); other draft kinds "
                "(dflash/eagle3/suffix) cannot use nemotron_h as an "
                "MTP-style rollback target."
            )

        for idx, cache_entry in enumerate(caches):
            if cache_entry is None:
                continue
            if _is_recurrent_cache(cache_entry):
                snapshots = states[idx] if states is not None else None
                if not snapshots or max_a >= len(snapshots):
                    have = 0 if not snapshots else len(snapshots)
                    raise RuntimeError(
                        "rollback_speculative_cache: missing recurrent-state "
                        f"snapshot for cache index {idx} (have {have} "
                        f"position(s), need index {max_a}). "
                        "speculative_verify_hidden/speculative_verify_logits "
                        "must be called first so this mixin's own verify "
                        "forward populates it via capture_recurrent_states."
                    )
                # Fork: per-row gather mirrors qwen3_5's per-row handling
                # (rollback_speculative_cache there also selects a snapshot
                # per accepted index). It is written generally even though
                # the uniform-acceptance gate above currently makes every
                # row pick the same snapshot index.
                conv_rows = [
                    snapshots[accepted_list[row]][0][row] for row in range(batch_size)
                ]
                ssm_rows = [
                    snapshots[accepted_list[row]][1][row] for row in range(batch_size)
                ]
                cache_entry[0] = mx.stack(conv_rows, axis=0)
                cache_entry[1] = mx.stack(ssm_rows, axis=0)
            elif cache_entry.is_trimmable():
                if trim > 0:
                    cache_entry.trim(trim)
            else:
                raise RuntimeError(
                    "rollback_speculative_cache: cache type "
                    f"{type(cache_entry).__name__} at index {idx} is neither "
                    "a recurrent (ArraysCache) nor a trimmable KV cache; this "
                    "mixin doesn't know how to roll it back."
                )
        return max_a
