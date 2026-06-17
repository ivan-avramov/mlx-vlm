"""Drafter-free SuffixDecoding (n-gram / prompt-lookup) speculative decoding.

Mirrors the conventions in ``test_speculative.py``:
  * proposer unit tests are pure-data (no model);
  * round-loop tests drive a fake LM whose argmax follows a known script;
  * the integration "gate" asserts greedy output is token-identical with and
    without suffix decoding, against a real (tiny, random-weight) gemma4.
"""

from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.speculative.suffix_decoding import (
    SuffixDecodingProposer,
    _adaptive_max_draft,
    run_suffix_decoding_rounds,
)


# --------------------------------------------------------------------------- #
# Test doubles for the round-loop tests
# --------------------------------------------------------------------------- #
class _MarkovLM:
    """Fake target LM whose greedy argmax follows an order-1 Markov script.

    ``nxt[x]`` is the greedy next token after token ``x``. The forward returns
    one-hot logits per position so ``mx.argmax`` reproduces the script exactly,
    making greedy generation fully deterministic.
    """

    def __init__(self, nxt, vocab=128):
        self.nxt = nxt
        self.vocab = vocab
        self.calls = 0
        self.rollback_calls = []

    def __call__(self, inputs, cache=None, **kwargs):
        self.calls += 1
        ids = [int(x) for x in inputs.reshape(-1).tolist()]
        rows = []
        for x in ids:
            row = [0.0] * self.vocab
            row[self.nxt[x]] = 10.0
            rows.append(row)
        return SimpleNamespace(logits=mx.array([rows]), gdn_states=None)

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        self.rollback_calls.append((caches, gdn_states, accepted, block_size))


class _ScriptedProposer:
    """Proposer double that emits a fixed sequence of drafts (then misses)."""

    def __init__(self, drafts):
        self._drafts = [list(d) for d in drafts]
        self._i = 0
        self.accept_lens = []
        self.draft_lens = []
        self.max_draft = None
        self.observed = []
        self.reset_with = None
        self._tokens = []

    def reset(self, ids):
        self.reset_with = list(ids)
        self._tokens = list(ids)

    def observe(self, toks):
        self.observed.append(list(toks))
        self._tokens.extend(toks)

    @property
    def tokens(self):
        return self._tokens

    def propose(self, suffix, max_draft):
        if self._i < len(self._drafts):
            d = self._drafts[self._i]
            self._i += 1
            return d[:max_draft]
        return []


_ARGMAX = lambda logits: mx.argmax(logits, axis=-1)


def _drive(model, proposer, *, first_bonus, max_tokens, draft_block_size=8):
    return [
        tok
        for tok, _ in run_suffix_decoding_rounds(
            model,
            proposer,
            prompt_token_ids=[],
            prompt_cache=[SimpleNamespace(offset=0)],
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=_ARGMAX,
            draft_block_size=draft_block_size,
        )
    ]


# --------------------------------------------------------------------------- #
# Unit — proposer (pure data)
# --------------------------------------------------------------------------- #
def test_suffix_propose_exact_repeat():
    # corpus contains a unique [2,3,4,5] run; querying suffix [2,3] proposes
    # the tokens that followed it.
    p = SuffixDecodingProposer(min_match=2)
    p.reset([7, 8, 2, 3, 4, 5, 6])
    assert p.propose([9, 9, 2, 3], max_draft=3) == [4, 5, 6]


def test_suffix_propose_truncates_to_max_draft():
    p = SuffixDecodingProposer(min_match=2)
    p.reset([7, 8, 2, 3, 4, 5, 6])
    assert p.propose([2, 3], max_draft=2) == [4, 5]


def test_suffix_propose_no_match_returns_empty():
    p = SuffixDecodingProposer(min_match=2)
    p.reset([7, 8, 2, 3, 4, 5, 6])
    assert p.propose([100, 101], max_draft=4) == []


def test_suffix_propose_min_match_threshold():
    # suffix shorter than min_match can never match -> [].
    p = SuffixDecodingProposer(min_match=2)
    p.reset([7, 8, 2, 3, 4, 5, 6])
    assert p.propose([3], max_draft=4) == []


def test_suffix_propose_excludes_trailing_self_match():
    # When the suffix IS the corpus tail (the usual round-loop case), the
    # trailing occurrence has no continuation and must be skipped; an earlier
    # occurrence supplies the proposal.
    p = SuffixDecodingProposer(min_match=2)
    p.reset([1, 2, 3, 9, 9, 1, 2])  # [1,2] at idx 0 (->3) and idx 5 (tail)
    assert p.propose([1, 2], max_draft=3) == [3, 9, 9]


def test_suffix_propose_prefers_longest_context_match():
    # Two occurrences of the min-match anchor [2,3]; the one with a longer
    # backward-matching context (… 1,2,3) wins over the bare [2,3].
    p = SuffixDecodingProposer(min_match=2, max_match=8)
    p.reset([2, 3, 50, 51, 1, 2, 3, 60, 61])
    # anchor [2,3] at idx 0 (cont 50,51) and idx 5 (cont 60,61).
    # suffix ends ...1,2,3 -> idx5 extends back through 1 -> longer match.
    assert p.propose([0, 1, 2, 3], max_draft=2) == [60, 61]


def test_suffix_observe_extends_index():
    # Tokens learned via observe() are matchable on later proposes.
    p = SuffixDecodingProposer(min_match=2)
    p.reset([7, 8])
    assert p.propose([4, 5], max_draft=2) == []
    p.observe([4, 5, 6, 7])  # corpus -> [7,8,4,5,6,7]
    assert p.propose([9, 4, 5], max_draft=2) == [6, 7]


def test_suffix_reset_clears_prior_corpus():
    p = SuffixDecodingProposer(min_match=2)
    p.reset([4, 5, 6, 7])
    assert p.propose([4, 5], max_draft=2) == [6, 7]
    p.reset([1, 2, 3])
    assert p.propose([4, 5], max_draft=2) == []


# --------------------------------------------------------------------------- #
# Unit — rounds against a fake LM (mirror test_speculative.py walk tests)
# --------------------------------------------------------------------------- #
def test_rounds_partial_accept_trims_cache_to_accepted_plus_one():
    # Greedy script: 0->1->2->3->4 ; the drafted 3rd token (7) is wrong.
    nxt = {0: 1, 1: 2, 2: 3, 3: 4, 7: 50}
    lm = _MarkovLM(nxt)
    model = SimpleNamespace(language_model=lm)
    proposer = _ScriptedProposer([[1, 2, 7]])  # round 1 draft; then misses

    out = _drive(model, proposer, first_bonus=0, max_tokens=5)

    # Round 1 accepts [1,2] then the target's correction 3; round 2 misses -> 4.
    assert out == [1, 2, 3, 4]
    assert proposer.accept_lens == [2]
    assert proposer.draft_lens == [3]

    # Rollback trims the verify chunk to accepted+1: accepted=2, block=len+1=4.
    assert len(lm.rollback_calls) == 1
    caches, gdn, accepted, block = lm.rollback_calls[0]
    assert gdn is None
    assert accepted == 2
    assert block == 4

    # Two forward passes produced four tokens (one verify + one miss step).
    assert lm.calls == 2
    assert lm.calls < len(out)

    # The first bonus is observed before any proposal so the corpus tail is live.
    assert proposer.observed[0] == [0]


def test_rounds_full_accept_skips_rollback_and_speeds_up():
    # A strict 4-cycle the real proposer can echo perfectly from the prompt.
    nxt = {10: 11, 11: 12, 12: 13, 13: 10}
    lm = _MarkovLM(nxt)
    model = SimpleNamespace(language_model=lm)
    proposer = SuffixDecodingProposer(min_match=2)

    prompt = [10, 11, 12, 13, 10, 11, 12, 13]
    out = [
        tok
        for tok, _ in run_suffix_decoding_rounds(
            model,
            proposer,
            prompt_token_ids=prompt,
            prompt_cache=[SimpleNamespace(offset=0)],
            first_bonus=10,  # greedy after the prompt's trailing 13
            max_tokens=9,
            sampler=_ARGMAX,
            draft_block_size=8,
        )
    ]

    # Token-identical to the true greedy continuation of the cycle.
    assert out == [11, 12, 13, 10, 11, 12, 13, 10]
    # Every round drafted and accepted at least one token.
    assert proposer.accept_lens
    assert all(a > 0 for a in proposer.accept_lens)
    # Full prefix acceptance never needs a rollback.
    assert lm.rollback_calls == []
    # The win: far fewer forward passes than tokens emitted.
    assert lm.calls < len(out)


def test_rounds_pure_miss_matches_plain_greedy():
    # No matches ever -> every round is a single normal AR step (no regression,
    # output still exactly greedy).
    nxt = {0: 1, 1: 2, 2: 3, 3: 4}
    lm = _MarkovLM(nxt)
    model = SimpleNamespace(language_model=lm)
    proposer = _ScriptedProposer([])  # always misses

    out = _drive(model, proposer, first_bonus=0, max_tokens=5)

    assert out == [1, 2, 3, 4]
    assert lm.calls == 4  # one forward per token, like plain AR
    assert lm.rollback_calls == []


import pytest  # noqa: E402

# --------------------------------------------------------------------------- #
# Wiring — dispatch branches in speculative/utils.py
# --------------------------------------------------------------------------- #
import mlx_vlm.speculative.utils as spec_utils  # noqa: E402


def test_run_speculative_rounds_dispatches_suffix(monkeypatch):
    captured = {}

    def fake_rounds(
        model, proposer, prompt_cache, prompt_token_ids, *, first_bonus, **kwargs
    ):
        captured["prompt_token_ids"] = prompt_token_ids
        captured["first_bonus"] = first_bonus
        yield 7, None

    monkeypatch.setattr(spec_utils, "run_suffix_decoding_rounds", fake_rounds)

    out = list(
        spec_utils.run_speculative_rounds(
            SimpleNamespace(language_model=SimpleNamespace()),
            object(),  # proposer passed as draft_model
            [],
            mx.array([[1, 2, 3]], dtype=mx.int32),
            mx.array([5], dtype=mx.int32),  # first_token
            mx.zeros((1, 4)),  # logprobs
            None,  # last_outputs (unused by suffix)
            draft_kind="suffix",
            max_tokens=10,
            sampler=_ARGMAX,
            prompt_token_ids=[1, 2, 3],
        )
    )

    assert out[0][0] == 5  # first bonus yielded by the dispatcher
    assert out[1] == (7, None)  # then delegated to the suffix rounds
    assert captured["first_bonus"] == 5
    assert captured["prompt_token_ids"] == [1, 2, 3]


def test_run_speculative_rounds_suffix_falls_back_to_input_ids(monkeypatch):
    captured = {}

    def fake_rounds(model, proposer, prompt_cache, prompt_token_ids, **kwargs):
        captured["prompt_token_ids"] = prompt_token_ids
        yield 7, None

    monkeypatch.setattr(spec_utils, "run_suffix_decoding_rounds", fake_rounds)

    list(
        spec_utils.run_speculative_rounds(
            SimpleNamespace(language_model=SimpleNamespace()),
            object(),
            [],
            mx.array([[1, 2, 3]], dtype=mx.int32),
            mx.array([5], dtype=mx.int32),
            mx.zeros((1, 4)),
            None,
            draft_kind="suffix",
            max_tokens=10,
            sampler=_ARGMAX,
        )
    )
    # No explicit prompt_token_ids -> derive from input_ids.
    assert captured["prompt_token_ids"] == [1, 2, 3]


def test_run_speculative_server_rounds_suffix_singleton(monkeypatch):
    captured = {}

    def fake_rounds(
        model, proposer, prompt_cache, prompt_token_ids, *, first_bonus, **kw
    ):
        captured["first_bonus"] = first_bonus
        captured["ptids"] = prompt_token_ids
        yield 9, None

    monkeypatch.setattr(spec_utils, "run_suffix_decoding_rounds", fake_rounds)

    out = list(
        spec_utils.run_speculative_server_rounds(
            SimpleNamespace(language_model=SimpleNamespace()),
            object(),
            prompt_cache=[],
            hidden=None,
            draft_kind="suffix",
            first_bonus=mx.array([4], dtype=mx.int32),
            max_tokens=5,
            sampler=_ARGMAX,
            prompt_tokens=mx.array([[1, 2]], dtype=mx.int32),
        )
    )

    assert captured["first_bonus"] == 4
    assert captured["ptids"] == [1, 2]
    assert out[0] == ([4], None)  # server wraps tokens in a list
    assert out[1] == ([9], None)


def test_run_speculative_server_rounds_suffix_batch_raises():
    with pytest.raises(NotImplementedError):
        list(
            spec_utils.run_speculative_server_rounds(
                SimpleNamespace(language_model=SimpleNamespace()),
                object(),
                prompt_cache=[],
                hidden=None,
                draft_kind="suffix",
                first_bonus=mx.array([4, 5], dtype=mx.int32),
                max_tokens=5,
                sampler=_ARGMAX,
            )
        )


def test_get_speculative_rounds_batch_rejects_suffix():
    with pytest.raises(NotImplementedError):
        spec_utils.get_speculative_rounds_batch("suffix")


# --------------------------------------------------------------------------- #
# Integration — equivalence gate against a real (tiny, random-weight) gemma4
# --------------------------------------------------------------------------- #
from mlx_vlm.models import cache as cache_mod  # noqa: E402
from mlx_vlm.models.gemma4.config import TextConfig  # noqa: E402
from mlx_vlm.models.gemma4.language import LanguageModel  # noqa: E402


def _tiny_gemma4(seed=0):
    mx.random.seed(seed)
    cfg = TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        head_dim=16,
        global_head_dim=16,
        vocab_size=48,
        vocab_size_per_layer_input=48,
        num_key_value_heads=1,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=16,
        sliding_window=4096,
        sliding_window_pattern=2,
        final_logit_softcapping=None,
        use_double_wide_mlp=False,
    )
    lm = LanguageModel(cfg)
    lm.eval()
    return lm


def _reference_greedy(lm, prompt, n_tokens):
    c = cache_mod.make_prompt_cache(lm)
    out = lm(mx.array([prompt]), cache=c)
    tok = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    toks = [tok]
    for _ in range(n_tokens - 1):
        o = lm(mx.array([[tok]]), cache=c)
        tok = int(mx.argmax(o.logits[:, -1, :], axis=-1).item())
        toks.append(tok)
    return toks


def test_suffix_decoding_matches_greedy_on_real_tiny_gemma4():
    # THE GATE: greedy output must be token-identical with and without suffix
    # decoding. Exercises the real verify forward, real KV cache, and the real
    # gemma4 rollback_speculative_cache.
    lm = _tiny_gemma4(seed=0)
    model = SimpleNamespace(language_model=lm)
    # A repetitive prompt maximises the chance of (verified) proposals.
    prompt = [3, 4, 5, 6, 7, 8] * 4
    n = 24

    ref = _reference_greedy(lm, prompt, n)

    proposer = SuffixDecodingProposer(min_match=2)
    out = lm(mx.array([prompt]), cache=(spec_cache := cache_mod.make_prompt_cache(lm)))
    first_bonus = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    spec = [first_bonus] + [
        tok
        for tok, _ in run_suffix_decoding_rounds(
            model,
            proposer,
            spec_cache,
            prompt,
            first_bonus=first_bonus,
            max_tokens=n,
            sampler=_ARGMAX,
            draft_block_size=8,
        )
    ]

    assert spec == ref
    assert len(spec) == n


def test_suffix_decoding_no_regression_when_proposals_all_miss():
    # On effectively-novel output (random weights rarely echo), every round is a
    # single AR step: graceful degradation, never more forwards than tokens.
    lm = _tiny_gemma4(seed=1)
    model = SimpleNamespace(language_model=lm)
    prompt = [11, 12, 13, 14, 15]
    n = 16

    ref = _reference_greedy(lm, prompt, n)

    proposer = SuffixDecodingProposer(min_match=2)
    out = lm(mx.array([prompt]), cache=(spec_cache := cache_mod.make_prompt_cache(lm)))
    first_bonus = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    spec = [first_bonus] + [
        tok
        for tok, _ in run_suffix_decoding_rounds(
            model,
            proposer,
            spec_cache,
            prompt,
            first_bonus=first_bonus,
            max_tokens=n,
            sampler=_ARGMAX,
            draft_block_size=8,
        )
    ]
    assert spec == ref


class _ForcingProposer:
    """Always proposes a fixed (usually-wrong) draft, forcing the verify +
    rollback path to fire on every round so a cache-trim bug is observable."""

    def __init__(self, draft):
        self._draft = list(draft)
        self.accept_lens = []
        self.draft_lens = []
        self.max_draft = None
        self._tokens = []

    def reset(self, ids):
        self._tokens = list(ids)

    def observe(self, toks):
        self._tokens.extend(toks)

    @property
    def tokens(self):
        return self._tokens

    def propose(self, suffix, max_draft):
        return self._draft[: max(0, max_draft)]


def _cache_offset(cache):
    return next(int(c.offset) for c in cache if hasattr(c, "offset"))


def test_suffix_decoding_rollback_preserves_greedy_on_real_tiny_gemma4():
    # Forces a verified draft every round so the real gemma4
    # rollback_speculative_cache is exercised on every round. Output must stay
    # exactly greedy AND the spec KV cache must end the same length as a clean
    # autoregressive cache — a disabled/incorrect rollback leaves the rejected
    # draft tokens in the cache, growing it by the draft length each round.
    lm = _tiny_gemma4(seed=2)
    model = SimpleNamespace(language_model=lm)
    prompt = [3, 4, 5, 6, 7, 8, 9, 10]
    n = 20

    ref = _reference_greedy(lm, prompt, n)
    ref_cache = cache_mod.make_prompt_cache(lm)
    rout = lm(mx.array([prompt]), cache=ref_cache)
    rtok = int(mx.argmax(rout.logits[:, -1, :], axis=-1).item())
    for _ in range(n - 1):
        ro = lm(mx.array([[rtok]]), cache=ref_cache)
        rtok = int(mx.argmax(ro.logits[:, -1, :], axis=-1).item())

    proposer = _ForcingProposer([3, 4, 5])  # unlikely to match random greedy
    spec_cache = cache_mod.make_prompt_cache(lm)
    out = lm(mx.array([prompt]), cache=spec_cache)
    first_bonus = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    spec = [first_bonus] + [
        tok
        for tok, _ in run_suffix_decoding_rounds(
            model,
            proposer,
            spec_cache,
            prompt,
            first_bonus=first_bonus,
            max_tokens=n,
            sampler=_ARGMAX,
            draft_block_size=8,
        )
    ]

    assert spec == ref
    # The forcing draft was verified (and rejected) every round.
    assert proposer.accept_lens
    # Rollback trimmed each verify chunk back to accepted+1, so the spec cache
    # tracks the clean AR cache to within a single (final, un-rolled-back) draft
    # block. Without rollback it accumulates draft_block_size garbage per round.
    assert _cache_offset(spec_cache) - _cache_offset(ref_cache) <= 8


# --------------------------------------------------------------------------- #
# Unit — adaptive draft sizing (linear ramp up, geometric backoff)
# --------------------------------------------------------------------------- #
def test_adaptive_max_draft_no_history_uses_ceiling():
    assert _adaptive_max_draft([], [], ceiling=16, remaining=100) == 16


def test_adaptive_max_draft_bounded_by_remaining_budget():
    assert _adaptive_max_draft([], [], ceiling=16, remaining=3) == 3


def test_adaptive_max_draft_ramps_up_linearly_on_full_accept():
    # Last round fully accepted (a == d) -> grow by exactly 1.
    assert _adaptive_max_draft([8.0], [8], ceiling=16, remaining=100) == 9


def test_adaptive_max_draft_does_not_exceed_ceiling():
    assert _adaptive_max_draft([16.0], [16], ceiling=16, remaining=100) == 16


def test_adaptive_max_draft_geometric_backoff_on_low_acceptance():
    # Low acceptance rate -> halve the previous draft length.
    assert _adaptive_max_draft([0.0], [8], ceiling=16, remaining=100) == 4
    assert _adaptive_max_draft([0.0], [4], ceiling=16, remaining=100) == 2
    assert _adaptive_max_draft([0.0], [2], ceiling=16, remaining=100) == 1


def test_adaptive_max_draft_floor_is_one():
    assert _adaptive_max_draft([0.0], [1], ceiling=16, remaining=100) == 1


def test_adaptive_max_draft_holds_on_partial_accept():
    # Decent acceptance but not full -> hold steady (neither grow nor shrink).
    assert _adaptive_max_draft([6.0], [8], ceiling=16, remaining=100) == 8


# --------------------------------------------------------------------------- #
# Unit — cooldown (suppress proposing after sustained rejections)
# --------------------------------------------------------------------------- #
class _ConstLM:
    """Fake LM whose greedy argmax is a constant, so any draft is rejected."""

    def __init__(self, const, vocab=128):
        self.const = const
        self.vocab = vocab
        self.calls = 0
        self.rollback_calls = []

    def __call__(self, inputs, cache=None, **kwargs):
        self.calls += 1
        n = inputs.reshape(-1).shape[0]
        row = [0.0] * self.vocab
        row[self.const] = 10.0
        return SimpleNamespace(logits=mx.array([[row] * n]), gdn_states=None)

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        self.rollback_calls.append((caches, gdn_states, accepted, block_size))


class _AlwaysProposer:
    """Proposer double that always proposes the same (wrong) draft."""

    def __init__(self, draft, *, cooldown=None):
        self._draft = list(draft)
        self.cooldown = cooldown
        self.max_draft = None
        self.accept_lens = []
        self.draft_lens = []
        self._tokens = []

    def reset(self, ids):
        self._tokens = list(ids)

    def observe(self, toks):
        self._tokens.extend(toks)

    @property
    def tokens(self):
        return self._tokens

    def propose(self, suffix, max_draft):
        return self._draft[: max(0, max_draft)]


def test_cooldown_suppresses_proposing_after_consecutive_rejections():
    lm = _ConstLM(const=7)
    model = SimpleNamespace(language_model=lm)
    proposer = _AlwaysProposer(draft=[3, 3], cooldown=2)

    out = _drive(model, proposer, first_bonus=0, max_tokens=30)

    assert all(t == 7 for t in out)  # output is still exact greedy
    # After 2 rejected rounds the cooldown suppresses proposing, so only a
    # handful of (probe) verify rounds happen instead of one per token.
    assert len(lm.rollback_calls) <= 6


def test_no_cooldown_verifies_every_round():
    lm = _ConstLM(const=7)
    model = SimpleNamespace(language_model=lm)
    proposer = _AlwaysProposer(draft=[3, 3], cooldown=None)

    out = _drive(model, proposer, first_bonus=0, max_tokens=30)

    assert all(t == 7 for t in out)
    # No suppression -> essentially every token comes from a rejected verify.
    assert len(lm.rollback_calls) >= 20


def test_cooldown_does_not_trigger_while_accepting():
    nxt = {10: 11, 11: 12, 12: 13, 13: 10}
    lm = _MarkovLM(nxt)
    model = SimpleNamespace(language_model=lm)
    proposer = SuffixDecodingProposer(min_match=2, cooldown=2)
    prompt = [10, 11, 12, 13, 10, 11, 12, 13]

    out = [
        tok
        for tok, _ in run_suffix_decoding_rounds(
            model,
            proposer,
            [SimpleNamespace(offset=0)],
            prompt,
            first_bonus=10,
            max_tokens=12,
            sampler=_ARGMAX,
            draft_block_size=8,
        )
    ]

    # Full acceptance the whole way -> cooldown never fires, output is the cycle.
    assert out == [11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13]
    assert all(a > 0 for a in proposer.accept_lens)


def test_proposer_cooldown_defaults_to_none():
    assert SuffixDecodingProposer(min_match=2).cooldown is None
    assert SuffixDecodingProposer(min_match=2, cooldown=3).cooldown == 3


# --------------------------------------------------------------------------- #
# Unit — suffix honors thinking_budget (round-granularity forced close)
# --------------------------------------------------------------------------- #
class _FakeBudgetCriteria:
    """Minimal stand-in for ThinkingBudgetCriteria: stream_generate drives
    __call__ per emitted token; budget_exceeded flips True after `budget`
    thinking tokens and resets when the end token is seen."""

    def __init__(self, budget, end_token_id):
        self.thinking_budget = budget
        self.thinking_end_token_id = end_token_id
        self.enable_thinking = True
        self.in_thinking = True
        self.thinking_token_count = 0
        self.budget_exceeded = False

    def __call__(self, token_id):
        if token_id == self.thinking_end_token_id:
            self.in_thinking = False
            self.budget_exceeded = False
            return None
        if self.in_thinking:
            self.thinking_token_count += 1
            if self.thinking_token_count > self.thinking_budget:
                self.budget_exceeded = True
        return None


def test_suffix_forces_thinking_end_token_on_budget():
    # Greedy chain never emits the end token (99); the only way 99 appears is the
    # forced close. After the force, generation continues from 99 -> 50, 51, ...
    nxt = {i: i + 1 for i in range(60)}
    nxt[99] = 50
    lm = _MarkovLM(nxt, vocab=128)
    model = SimpleNamespace(language_model=lm)
    proposer = _ScriptedProposer([])  # always miss -> plain decode chain
    criteria = _FakeBudgetCriteria(budget=4, end_token_id=99)

    out = []
    for tok, _ in run_suffix_decoding_rounds(
        model,
        proposer,
        [SimpleNamespace(offset=0)],
        [],
        first_bonus=0,
        max_tokens=20,
        sampler=_ARGMAX,
        draft_block_size=8,
        thinking_budget_criteria=criteria,
    ):
        out.append(tok)
        criteria(tok)  # mimic stream_generate driving the criteria

    assert 99 in out  # the end token was forced (model never emits it itself)
    # forced within budget + one block of overshoot
    assert out.index(99) <= 4 + 8
    # generation continues after the forced close (post-thinking answer)
    assert out[out.index(99) + 1] == 50


def test_suffix_no_criteria_is_unaffected():
    # Without a criteria, behaviour is the plain greedy chain (no forced token).
    nxt = {i: i + 1 for i in range(60)}
    lm = _MarkovLM(nxt, vocab=128)
    model = SimpleNamespace(language_model=lm)
    proposer = _ScriptedProposer([])
    out = [
        tok
        for tok, _ in run_suffix_decoding_rounds(
            model,
            proposer,
            [SimpleNamespace(offset=0)],
            [],
            first_bonus=0,
            max_tokens=6,
            sampler=_ARGMAX,
            draft_block_size=8,
            thinking_budget_criteria=None,
        )
    ]
    assert out == [1, 2, 3, 4, 5]


# --------------------------------------------------------------------------- #
# Unit — CLI/generate_step structured-output fallback
# --------------------------------------------------------------------------- #
from mlx_vlm.generate.ar import _suffix_structured_fallback  # noqa: E402


def test_suffix_structured_fallback():
    # suffix + a structured grammar -> fall back to plain decode
    assert _suffix_structured_fallback("suffix", [object()]) is True
    # no grammar -> speculate
    assert _suffix_structured_fallback("suffix", None) is False
    assert _suffix_structured_fallback("suffix", []) is False
    # other drafter kinds are not gated here
    assert _suffix_structured_fallback("dflash", [object()]) is False
    assert _suffix_structured_fallback("mtp", [object()]) is False
    assert _suffix_structured_fallback(None, [object()]) is False


# --------------------------------------------------------------------------- #
# Integration — hybrid GatedDeltaNet (qwen3_5) GDN-state rollback
# --------------------------------------------------------------------------- #
import mlx_vlm.models.qwen3_5.language as qwen_language  # noqa: E402
from mlx_vlm.models.cache import ArraysCache  # noqa: E402
from mlx_vlm.turboquant import BatchTurboQuantKVCache  # noqa: E402


def _tiny_qwen3_5(seed=0):
    # Dims chosen so the GatedDeltaNet decode-step Metal kernel builds: 4/4-dim
    # configs hit "zero-length arrays (float state[n_per_t])". num_hidden_layers=2
    # + full_attention_interval=2 => layer 0 is GDN (ArraysCache), layer 1 is full
    # attention (KVCache), so the recurrent-state rollback path is actually exercised.
    mx.random.seed(seed)
    cfg = qwen_language.TextConfig(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        full_attention_interval=2,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        head_dim=16,
        rope_parameters={
            "type": "default",
            "mrope_section": [1, 0, 0],
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )
    lm = qwen_language.LanguageModel(cfg)
    # get_rope_index reads self.config.vision_config.spatial_merge_size even for
    # text-only input, so attach a minimal ModelConfig (mirrors test_speculative.py).
    lm.config = qwen_language.ModelConfig(
        text_config=cfg,
        vision_config=SimpleNamespace(spatial_merge_size=2),
        model_type="qwen3_5",
        image_token_id=101,
        video_token_id=102,
        image_token_index=101,
        video_token_index=102,
        vision_start_token_id=100,
        vision_end_token_id=103,
        vocab_size=32,
    )
    lm.eval()
    return lm


def _qwen_reference_greedy(lm, prompt, n_tokens):
    c = cache_mod.make_prompt_cache(lm)
    out = lm(mx.array([prompt]), cache=c)
    tok = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    toks = [tok]
    for _ in range(n_tokens - 1):
        o = lm(mx.array([[tok]]), cache=c)
        tok = int(mx.argmax(o.logits[:, -1, :], axis=-1).item())
        toks.append(tok)
    return toks, c


def _gdn_state(cache):
    """Flattened GDN recurrent + conv arrays from the ArraysCache slots."""
    arrs = []
    for c in cache:
        if isinstance(c, ArraysCache):
            arrs.extend(s for s in c.state if s is not None)
    return arrs


def test_suffix_decoding_rollback_preserves_gdn_on_real_tiny_qwen3_5():
    # Forces a verified draft every round so the GDN recurrent state advances
    # through rejected draft positions and MUST be restored on rollback. Without
    # capture (gdn_states=None) the qwen rollback cannot restore it -> divergence
    # (in fact a TypeError on current code). With capture, output stays exactly
    # greedy AND the GDN state matches a clean autoregressive decode.
    lm = _tiny_qwen3_5(seed=2)
    model = SimpleNamespace(language_model=lm)
    prompt = [3, 4, 5, 6, 7, 8, 9, 10]
    n = 20

    ref, ref_cache = _qwen_reference_greedy(lm, prompt, n)

    proposer = _ForcingProposer([3, 4, 5])  # unlikely to match random greedy
    spec_cache = cache_mod.make_prompt_cache(lm)
    out = lm(mx.array([prompt]), cache=spec_cache)
    first_bonus = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    spec = [first_bonus] + [
        tok
        for tok, _ in run_suffix_decoding_rounds(
            model,
            proposer,
            spec_cache,
            prompt,
            first_bonus=first_bonus,
            max_tokens=n,
            sampler=_ARGMAX,
            draft_block_size=8,
        )
    ]

    # Token-identical greedy output IS the end-to-end proof that the GDN state was
    # restored on every rejection: a disabled/None capture crashes (TypeError) or
    # diverges. Element-wise GDN restoration is asserted at a controlled rollback
    # boundary in test_qwen_gdn_state_restored_to_clean_after_rejected_verify
    # (the run's final verify block is intentionally NOT rolled back -- the loop
    # returns on max_tokens before rollback -- so end-of-run cache state carries an
    # un-rolled-back tail, which is why the gemma4 gate checks only the offset bound).
    assert spec == ref
    assert proposer.accept_lens  # the forcing draft was verified every round
    # KV cache tracks the clean AR cache to within one (final, un-rolled-back) block.
    assert _cache_offset(spec_cache) - _cache_offset(ref_cache) <= 8


def test_qwen_gdn_state_restored_to_clean_after_rejected_verify():
    # Targeted, deterministic proof that rollback restores the GatedDeltaNet
    # recurrent + conv state to exactly what a clean single-token decode produces.
    # Process the prompt, commit one greedy token t0, snapshot the GDN state; then
    # on a fresh cache run a verify block [t0, wrong, wrong, wrong] WITH capture and
    # rollback(accepted=0) -> the GDN state must match the clean "after t0" snapshot.
    lm = _tiny_qwen3_5(seed=3)
    prompt = [3, 4, 5, 6, 7, 8]

    cref = cache_mod.make_prompt_cache(lm)
    o = lm(mx.array([prompt]), cache=cref)
    t0 = int(mx.argmax(o.logits[:, -1, :], axis=-1).item())
    lm(mx.array([[t0]]), cache=cref)  # commit t0 via the single-token decode path
    clean_gdn = _gdn_state(cref)

    cspec = cache_mod.make_prompt_cache(lm)
    lm(mx.array([prompt]), cache=cspec)
    drafts = [d for d in (30, 31, 29) if d != t0]  # ensure a genuine rejection
    verify_in = mx.array([[t0, *drafts]])
    vout = lm(verify_in, cache=cspec, **lm.suffix_verify_kwargs())
    mx.eval(vout.logits)
    assert vout.gdn_states is not None  # capture populated the rollback state
    lm.rollback_speculative_cache(cspec, vout.gdn_states, 0, verify_in.shape[1])
    restored_gdn = _gdn_state(cspec)

    assert len(restored_gdn) == len(clean_gdn) and restored_gdn
    for a, b in zip(restored_gdn, clean_gdn):
        assert bool(mx.allclose(a, b, atol=1e-4).item())


def test_suffix_verify_kwargs_hook_per_model():
    # gemma4: KV-only -> no capture kwargs.
    assert _tiny_gemma4(seed=0).suffix_verify_kwargs() == {}
    # qwen3_5: GDN -> capture_layer_ids=[] (gdn_sink without hidden capture).
    assert _tiny_qwen3_5(seed=0).suffix_verify_kwargs() == {"capture_layer_ids": []}


class _KwargRecordingLM:
    """Fake LM exposing a suffix_verify_kwargs hook; records what the verify
    forward actually received so we can prove the kwargs are threaded through."""

    def __init__(self, nxt, kwargs, vocab=128):
        self._nxt = nxt
        self._kwargs = dict(kwargs)
        self.vocab = vocab
        self.verify_kwargs_seen = None

    def suffix_verify_kwargs(self):
        return dict(self._kwargs)

    def __call__(self, inputs, cache=None, **kwargs):
        ids = [int(x) for x in inputs.reshape(-1).tolist()]
        if len(ids) > 1:  # the multi-token verify forward (not the 1-token miss path)
            self.verify_kwargs_seen = kwargs
        rows = []
        for x in ids:
            row = [0.0] * self.vocab
            row[self._nxt[x]] = 10.0
            rows.append(row)
        return SimpleNamespace(logits=mx.array([rows]), gdn_states=None)

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        pass


def test_suffix_verify_forward_receives_hook_kwargs():
    nxt = {0: 1, 1: 2, 2: 3, 3: 4, 7: 50}
    lm = _KwargRecordingLM(nxt, {"capture_layer_ids": []})
    model = SimpleNamespace(language_model=lm)
    _drive(model, _ScriptedProposer([[1, 2, 7]]), first_bonus=0, max_tokens=5)
    assert lm.verify_kwargs_seen == {"capture_layer_ids": []}


def test_suffix_no_hook_passes_no_extra_kwargs():
    # A fake LM without the hook (the v1 fakes) must see an unchanged verify call.
    nxt = {0: 1, 1: 2, 2: 3, 3: 4, 7: 50}

    class _NoHookLM(_KwargRecordingLM):
        suffix_verify_kwargs = None  # attribute is not callable -> default {}

    lm = _NoHookLM(nxt, {})
    model = SimpleNamespace(language_model=lm)
    _drive(model, _ScriptedProposer([[1, 2, 7]]), first_bonus=0, max_tokens=5)
    assert lm.verify_kwargs_seen == {}


def test_qwen_capture_layer_ids_is_load_bearing_for_gdn_states():
    # Direct proof that the hook's kwarg — not the model by itself — is what makes
    # gdn_states non-None. A disabled/None capture leaves rollback nothing to restore.
    lm = _tiny_qwen3_5(seed=0)
    c = cache_mod.make_prompt_cache(lm)
    lm(mx.array([[3, 4, 5]]), cache=c)

    plain = lm(mx.array([[6, 7, 8]]), cache=c)
    assert plain.gdn_states is None  # without capture: nothing to roll back

    c2 = cache_mod.make_prompt_cache(lm)
    lm(mx.array([[3, 4, 5]]), cache=c2)
    captured = lm(mx.array([[6, 7, 8]]), cache=c2, **lm.suffix_verify_kwargs())
    assert captured.gdn_states is not None and len(captured.gdn_states) >= 1
    assert captured.hidden_states == []  # capture_layer_ids=[] adds no hidden overhead


def test_suffix_decoding_matches_greedy_on_real_tiny_qwen3_5():
    # THE GATE on hybrid GDN: greedy output token-identical with/without suffix,
    # using the real SuffixDecodingProposer (mixed accept/reject) on a repetitive
    # prompt, exercising the real verify, KV + GDN caches, and rollback.
    lm = _tiny_qwen3_5(seed=0)
    model = SimpleNamespace(language_model=lm)
    prompt = [3, 4, 5, 6, 7, 8] * 4
    n = 24

    ref, _ = _qwen_reference_greedy(lm, prompt, n)

    proposer = SuffixDecodingProposer(min_match=2)
    spec_cache = cache_mod.make_prompt_cache(lm)
    out = lm(mx.array([prompt]), cache=spec_cache)
    first_bonus = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    spec = [first_bonus] + [
        tok
        for tok, _ in run_suffix_decoding_rounds(
            model,
            proposer,
            spec_cache,
            prompt,
            first_bonus=first_bonus,
            max_tokens=n,
            sampler=_ARGMAX,
            draft_block_size=8,
        )
    ]

    assert spec == ref
    assert len(spec) == n


def _quant_kv_cache():
    # tiny qwen3_5: layer 0 = GDN (ArraysCache), layer 1 = full attn (quantized KV).
    # Mirrors Qwen3.6's kv_quant_scheme=turboquant runtime cache.
    arr = ArraysCache(size=2)
    arr.left_padding = mx.array([0], dtype=mx.int32)
    return [arr, BatchTurboQuantKVCache([0], bits=3.5)]


def test_qwen_suffix_verify_supports_quantized_kv_cache():
    # Regression: suffix verify (target_verify, via capture_layer_ids=[]) must not
    # crash on a turboquant KV cache — Qwen3.6 runs kv_quant_scheme=turboquant. The
    # target_verify manual attention loop slices cache keys directly, but quantized
    # caches return a non-subscriptable _QuantizedStateProxy; the fix falls back to
    # the cache-aware SDPA path. The verify must still select the same tokens a plain
    # (non-target-verify) forward would, and still capture GDN state for rollback.
    lm = _tiny_qwen3_5(seed=0)
    prompt = [3, 4, 5, 6, 7, 8]
    block = [9, 10, 11, 12]

    cplain = _quant_kv_cache()
    lm(mx.array([prompt]), cache=cplain)
    plain = lm(mx.array([block]), cache=cplain)  # plain forward (reference)

    cver = _quant_kv_cache()
    lm(mx.array([prompt]), cache=cver)
    ver = lm(mx.array([block]), cache=cver, **lm.suffix_verify_kwargs())
    mx.eval(plain.logits, ver.logits)

    assert ver.gdn_states is not None and len(ver.gdn_states) >= 1
    assert ver.logits.shape == plain.logits.shape
    # The verify path selects the same per-position tokens a plain forward would.
    assert mx.array_equal(
        mx.argmax(plain.logits, axis=-1), mx.argmax(ver.logits, axis=-1)
    ).item()
