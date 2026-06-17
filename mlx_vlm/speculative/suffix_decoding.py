"""Drafter-free SuffixDecoding (n-gram / prompt-lookup) speculative decoding.

A *model-free* speculative decoder: no draft model, no trained head, no extra
weights, no extra GPU memory. It proposes draft tokens from an n-gram index over
the prompt + tokens generated so far, then plugs into the **existing**
speculative verify + accept + rollback machinery (``_speculative_walk`` and the
target model's ``rollback_speculative_cache``).

v1 is "prompt lookup decoding" (PLD): a k-gram dict keyed by the
``min_match``-token anchor, with a longest-context-match / most-recent tie-break.
The proposer is a plain Python object (not an ``nn.Module``) and is passed *as*
``draft_model`` so the dispatch in ``generate/ar.py`` fires for free.

Correctness: under greedy sampling, output is token-identical with and without
suffix decoding — the verify pass corrects any wrong draft. Under temperature
> 0, acceptance routes through ``_speculative_walk`` (sample target, accept iff
it equals the draft, else take the target's token). Because suffix proposals are
**deterministic** (draft probability 1), accept-iff-equal is exactly speculative
rejection sampling and preserves the target distribution — it is not a "naive
equality check" that would bias a stochastic drafter.
"""

from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .common import _record_speculative_round, _speculative_walk, generation_stream

# Default maximum draft length when neither the proposer nor the caller pins one
# (i.e. no ``--draft-block-size``). HF's PromptLookup default is ~10; suffix
# matches on echo-heavy text are often long, so we allow a bit more.
DEFAULT_MAX_DRAFT = 16

# Miss-path decode chunk: how many plain AR tokens to generate as one pure-async
# device chain before re-checking for an n-gram match. Larger amortises the
# host readback better (faster on novel text) but defers match detection longer
# (a few lost speculation opportunities when echoing resumes).
MISS_CHUNK = 4

# Cooldown window bounds (tokens). After the proposer's ``cooldown`` threshold of
# consecutive 0-accept verify rounds, proposing is suppressed for ``window``
# tokens — starting at BASE and doubling on each re-trigger up to MAX — so a long
# novel passage settles into plain (near-baseline) decode with only rare probes.
COOLDOWN_BASE_WINDOW = 16
COOLDOWN_MAX_WINDOW = 256


class SuffixDecodingProposer:
    """Drafter-free proposer.

    Indexes ``prompt + generated tokens`` and proposes the continuation that
    most-recently followed the longest match of the current suffix.

    Args:
        min_match: shortest suffix (anchor) length that may match. A suffix
            shorter than this never proposes (returns ``[]``).
        max_match: longest suffix length considered when ranking matches by
            context length.
        max_draft: default cap on proposal length; falls back to the caller's
            ``draft_block_size`` (then :data:`DEFAULT_MAX_DRAFT`) at call time.
        cooldown: after this many consecutive 0-accept verify rounds, suppress
            proposing for a growing back-off window (then probe). ``None``
            disables it. Trades a tiny novel-text bookkeeping cost for no wasted
            verify forwards when speculation is hopeless.
    """

    def __init__(
        self,
        *,
        min_match: int = 2,
        max_match: int = 8,
        max_draft: Optional[int] = None,
        cooldown: Optional[int] = None,
    ):
        if min_match < 1:
            raise ValueError("min_match must be >= 1")
        if cooldown is not None and cooldown < 1:
            raise ValueError("cooldown must be >= 1 (or None to disable)")
        self.min_match = int(min_match)
        self.max_match = max(int(max_match), self.min_match)
        self.max_draft = max_draft
        # After this many consecutive 0-accept verify rounds, stop proposing for
        # a back-off window. None disables the cooldown (always propose).
        self.cooldown = int(cooldown) if cooldown is not None else None
        self._tokens: List[int] = []
        # anchor (min_match-gram) -> list of start positions, ascending.
        self._index: dict = {}
        # Speculative-stats hooks (read by _format_speculative_stats).
        self.accept_lens: List[float] = []
        self.draft_lens: List[int] = []

    # -- corpus / index management ---------------------------------------- #
    def reset(self, prompt_token_ids: List[int]) -> None:
        """(Re)build the index from the full prompt token ids.

        Must be fed the *full* prompt (including any cached prefix), not just
        the uncached suffix, so matches span the whole context.
        """
        self._tokens = [int(t) for t in prompt_token_ids]
        self._index = {}
        # Per-request stats; cleared here so a reused proposer (e.g. the server's
        # single shared instance) doesn't carry acceptance history across turns.
        self.accept_lens = []
        self.draft_lens = []
        k = self.min_match
        for pos in range(len(self._tokens) - k + 1):
            self._index.setdefault(tuple(self._tokens[pos : pos + k]), []).append(pos)

    def observe(self, emitted: List[int]) -> None:
        """Append accepted tokens so later proposals can match fresh output."""
        if not emitted:
            return
        k = self.min_match
        old_len = len(self._tokens)
        self._tokens.extend(int(t) for t in emitted)
        # New k-grams are those overlapping at least one new token.
        start = max(0, old_len - k + 1)
        for pos in range(start, len(self._tokens) - k + 1):
            self._index.setdefault(tuple(self._tokens[pos : pos + k]), []).append(pos)

    @property
    def tokens(self) -> List[int]:
        return self._tokens

    # -- proposal --------------------------------------------------------- #
    def propose(self, context_suffix: List[int], max_draft: int) -> List[int]:
        """Return up to ``max_draft`` candidate token ids (possibly empty).

        Anchors on the last ``min_match`` tokens of ``context_suffix``, then
        among earlier occurrences picks the one with the longest backward
        context match (ties broken toward the most recent), and returns the
        tokens that followed it.
        """
        k = self.min_match
        if max_draft <= 0 or len(context_suffix) < k:
            return []

        anchor = tuple(int(t) for t in context_suffix[-k:])
        starts = self._index.get(anchor)
        if not starts:
            return []

        corpus = self._tokens
        n = len(corpus)
        best_pos = -1
        best_match = -1
        for pos in starts:
            cont_start = pos + k
            if cont_start >= n:
                # No continuation to propose (includes the trailing self-match).
                continue
            # Extend the match backward for as long as the suffix agrees.
            match_len = k
            si = len(context_suffix) - k - 1
            ci = pos - 1
            while (
                match_len < self.max_match
                and si >= 0
                and ci >= 0
                and int(context_suffix[si]) == corpus[ci]
            ):
                match_len += 1
                si -= 1
                ci -= 1
            # Prefer a longer context match; tie-break toward most recent.
            if match_len > best_match or (match_len == best_match and pos > best_pos):
                best_match = match_len
                best_pos = pos

        if best_pos < 0:
            return []
        cont_start = best_pos + k
        return corpus[cont_start : cont_start + max_draft]


def _adaptive_max_draft(
    accept_lens: List[float],
    draft_lens: List[int],
    *,
    ceiling: int,
    remaining: int,
) -> int:
    """Pick the next draft-length cap from recent acceptance.

    Linear ramp **up** by 1 when the last proposal was fully accepted, geometric
    backoff (halve) when recent acceptance is weak, otherwise hold. Bounded to
    ``[1, min(ceiling, remaining)]``. Like dflash's block-size controller, but
    floored at 1 so misses on novel text stay as cheap as a single decode.
    """
    cap = min(int(ceiling), int(remaining))
    if cap <= 1:
        return max(0, cap)

    recent = [
        (float(a), int(d))
        for a, d in zip(accept_lens[-8:], draft_lens[-8:])
        if int(d) > 0
    ]
    if not recent:
        return cap

    last_a, last_d = recent[-1]
    drafted = sum(d for _, d in recent)
    accepted = sum(a for a, _ in recent)
    accept_rate = accepted / drafted if drafted else 0.0

    if accept_rate < 0.5:
        nxt = max(1, last_d // 2)  # geometric backoff
    elif last_a >= last_d:
        nxt = last_d + 1  # linear ramp up on full acceptance
    else:
        nxt = last_d  # hold

    return max(1, min(cap, nxt))


def run_suffix_decoding_rounds(
    model: nn.Module,
    proposer: SuffixDecodingProposer,
    prompt_cache: List,
    prompt_token_ids: List[int],
    *,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    thinking_budget_criteria: Optional[Any] = None,
) -> Generator[Tuple[int, None], None, None]:
    """Drafter-free speculative-decoding round loop (single sequence, B == 1).

    Mirrors ``_dflash_rounds`` but a token-only proposer needs none of the
    model internals: no ``return_hidden``, no ``return_shared_kv``, no shared-KV
    slicing, no drafter cache. ``generate_step`` is responsible for prefill,
    sampling the first bonus token, and yielding it; this loop yields every
    subsequent token.

    Each round:
        propose -> (empty? emit one normal token) -> verify (logits only) ->
        ``_speculative_walk`` -> ``rollback_speculative_cache`` (trim KV to
        accepted+1) -> ``proposer.observe`` -> advance by accepted+1.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache. "
            "Suffix decoding supports models with a speculative rollback hook "
            "(dense gemma4, hybrid GatedDeltaNet qwen3_5)."
        )

    # Per-target capture hook: dense KV-only models (gemma4) need nothing; hybrid
    # GatedDeltaNet models (qwen3_5) return {"capture_layer_ids": []} so the verify
    # forward snapshots GDN state for rollback. Detected via the hook, never by
    # model_type; defaults to no extra kwargs (e.g. fake LMs in tests).
    _vk_hook = getattr(lm, "suffix_verify_kwargs", None)
    verify_kwargs = _vk_hook() if callable(_vk_hook) else {}

    proposer.reset(prompt_token_ids)
    proposer.observe([first_bonus])  # corpus tail must include the latest token

    ceiling = int(proposer.max_draft or draft_block_size or DEFAULT_MAX_DRAFT)

    # Cooldown state: suppress proposing while ``emitted < cold_until`` after a
    # run of 0-accept verify rounds (see SuffixDecodingProposer.cooldown).
    cooldown = getattr(proposer, "cooldown", None)
    miss_run = 0
    cold_until = 0
    window = COOLDOWN_BASE_WINDOW

    b = first_bonus
    emitted = 1  # the first bonus has already been yielded by the caller

    while emitted < max_tokens:
        # Thinking budget: stream_generate drives the criteria's __call__ on each
        # yielded token, keeping ``budget_exceeded`` current. If thinking ran over
        # the cap, commit the pending token and force the model's real end-of-
        # thinking token so generation leaves the thinking block and answers.
        # Block-granularity (caught at the next round top, <= one draft block of
        # overshoot); never fires when no criteria / no thinking is configured.
        if thinking_budget_criteria is not None and getattr(
            thinking_budget_criteria, "budget_exceeded", False
        ):
            with mx.stream(generation_stream):
                lm(mx.array([[b]], dtype=token_dtype), cache=prompt_cache)
            end_id = int(thinking_budget_criteria.thinking_end_token_id)
            yield end_id, None
            emitted += 1
            proposer.observe([end_id])
            b = end_id
            continue

        budget = max_tokens - emitted
        cap = _adaptive_max_draft(
            proposer.accept_lens,
            proposer.draft_lens,
            ceiling=ceiling,
            remaining=budget,
        )
        suppressed = cooldown is not None and emitted < cold_until
        draft = [] if suppressed else proposer.propose(proposer.tokens, cap)

        if not draft:
            # Miss: fall back to plain autoregressive decode. A per-token loop
            # would regress vs the baseline because the n-gram proposer needs each
            # token's value on the host (forcing a device->host sync that stalls
            # the GPU pipeline). Instead run a short *pure-async device chain* — a
            # single-token forward per step, identical to baseline decode so greedy
            # output is bit-identical — then materialise the whole chunk with one
            # sync, observe it, and re-check for a match. Match *detection* is
            # deferred by at most one chunk (a performance, not correctness, knob).
            while emitted < max_tokens:
                steps = min(MISS_CHUNK, max_tokens - emitted)
                chunk = []
                with mx.stream(generation_stream):
                    y = mx.array([[b]], dtype=token_dtype)
                    for _ in range(steps):
                        out = lm(y, cache=prompt_cache)
                        y = (
                            sampler(out.logits[:, -1:, :])
                            .reshape(1, 1)
                            .astype(token_dtype)
                        )
                        chunk.append(y)
                mx.eval(chunk[-1])  # one sync materialises the whole chain
                toks = [int(t.reshape(-1).item()) for t in chunk]
                for tok in toks:
                    yield tok, None
                    emitted += 1
                proposer.observe(toks)
                b = toks[-1]
                if emitted % 256 == 0:
                    mx.clear_cache()
                # Thinking budget may have tripped mid-chunk; leave the miss loop
                # so the round top forces the end-of-thinking token.
                if thinking_budget_criteria is not None and getattr(
                    thinking_budget_criteria, "budget_exceeded", False
                ):
                    break
                # While cooling down, keep decoding plainly instead of probing.
                if emitted < max_tokens and not (
                    cooldown is not None and emitted < cold_until
                ):
                    recheck_cap = _adaptive_max_draft(
                        proposer.accept_lens,
                        proposer.draft_lens,
                        ceiling=ceiling,
                        remaining=max_tokens - emitted,
                    )
                    if proposer.propose(proposer.tokens, recheck_cap):
                        break  # a match reappeared -> back to the verify path
            continue

        draft_tokens = mx.array([draft], dtype=token_dtype)
        with mx.stream(generation_stream):
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=token_dtype), draft_tokens], axis=1
            )
            verify_out = lm(verify_input, cache=prompt_cache, **verify_kwargs)
            target_tokens = sampler(verify_out.logits)
        mx.async_eval(target_tokens)

        accepted, new_tokens = _speculative_walk(draft_tokens, target_tokens, budget)
        _record_speculative_round(proposer, accepted, len(draft))

        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return

        proposer.observe(new_tokens)
        b = new_tokens[-1] if new_tokens else b

        # Trim the verify KV back to accepted+1 positions (the accepted drafts
        # plus the seed token). gdn_states is None for dense models.
        if accepted < len(draft):
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache,
                    getattr(verify_out, "gdn_states", None),
                    accepted,
                    len(draft) + 1,
                )

        # Cooldown bookkeeping: a run of fully-rejected rounds suppresses further
        # proposing for a growing window; any acceptance resets it.
        if cooldown is not None:
            if accepted == 0:
                miss_run += 1
                if miss_run >= cooldown:
                    cold_until = emitted + window
                    window = min(window * 2, COOLDOWN_MAX_WINDOW)
                    miss_run = 0
            else:
                miss_run = 0
                window = COOLDOWN_BASE_WINDOW

        if emitted % 256 == 0:
            mx.clear_cache()
