"""Env-gated MTP round/head profiler (M29 H1, 2026-08-31; fix round 2026-09-01).

CPU-pinned (no GPU, no real model): the round-loop tests use the same fakes
as ``test_mtp_thinking_budget.py`` (patched ``_mtp_verify_target`` /
``_mtp_acceptance_walk``); the head-loop tests reuse
``test_nemotron_h_mtp.py``'s tiny synthetic drafter/target helpers.

Fix round (verifier verdict FIX-FIRST on commit 7a1e6d48):
F1 two new phases (``yield``, ``accept``) so consumer/attribution time
    stops leaking into ``rollback``/``other``.
F2 the early-return path (yield loop hits ``max_tokens`` mid-round) now
    marks ``yield`` and calls ``end_unit`` before returning, so that
    round's own draft/verify/walk time is no longer counted in totals
    without a matching unit in the per-round means.
F3 ``round_profiler_from_env`` resets the head singleton so per-request
    head lines are not cumulative across a multi-request server session.
F4 this file gained: an exact-count synchronize assertion (a), a direct
    fence-order unit test of ``_PhaseTimer.mark`` (b), accepted/rd and
    n_draft/rd + yield/accept-field assertions (c), a two-generation
    head-not-cumulative test (d), and a rollback-mark-actually-fires
    comparison against an accept-all run (e).
"""

import re
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

import mlx_vlm.speculative.mtp as mtp_utils
from mlx_vlm.speculative import mtp_profile
from mlx_vlm.speculative.mtp_profile import (
    MTPHeadProfiler,
    MTPRoundProfiler,
    _PhaseTimer,
    cache_state_arrays,
    reset_head_profiler,
)
from mlx_vlm.speculative.utils import _mtp_rounds
from mlx_vlm.tests.test_nemotron_h_mtp import (
    HIDDEN_SIZE,
    NemotronHMTPDraftModel,
    _drafter_config,
    _target_model,
)

ROUND_LINE_RE = re.compile(
    r"^\[mtp_profile\] rounds=(?P<rounds>\d+) draft=(?P<draft>[-\d.]+) "
    r"verify=(?P<verify>[-\d.]+) walk=(?P<walk>[-\d.]+) "
    r"yield=(?P<yield_>[-\d.]+) accept=(?P<accept>[-\d.]+) "
    r"rollback=(?P<rollback>[-\d.]+) other=(?P<other>[-\d.]+) "
    r"emitted/rd=(?P<emitted_rd>[-\d.]+) accepted/rd=(?P<accepted_rd>[-\d.]+) "
    r"n_draft/rd=(?P<n_draft_rd>[-\d.]+) round_ms=(?P<round_ms>[-\d.]+) "
    r"final=(?P<final>[01])$"
)
HEAD_LINE_RE = re.compile(
    r"^\[mtp_profile_head\] draft_tokens=(?P<draft_tokens>\d+) "
    r"proj_layers=(?P<proj_layers>[-\d.]+) lm_head=(?P<lm_head>[-\d.]+) "
    r"sampler=(?P<sampler>[-\d.]+) eval=(?P<eval>[-\d.]+) "
    r"final=(?P<final>[01])$"
)

# 3 rounds, all clean: with the fixed (1, [9, 10]) walk fake and block_size=3
# (bs=3, n_draft=2 every round), max_tokens=7 makes the loop's own
# max-tokens check trip exactly on round 3's SECOND (last) yielded token --
# i.e. the "early return" round is, numerically, a round that already
# yielded its full 2 tokens. So all 3 rounds end up with emitted=2,
# accepted=1, n_draft=2 -> emitted/rd=2.0, accepted/rd=1.0, n_draft/rd=2.0
# exactly. (At the old ROUNDS_MAX_TOKENS=8, round 4 starts with a smaller
# bs=2 block and both end_unit's counters and "rounds" itself would differ
# from the old test's "rounds=3" -- 7 was chosen specifically so the F2 fix
# reproduces the pre-fix "rounds=3" line shape while now genuinely covering
# 3 end_unit calls instead of 2.)
ROUNDS_MAX_TOKENS = 7


class _CacheEntry:
    """Fake prompt-cache entry: ``.state`` is a tuple of mx arrays, matching
    the real cache classes' ``state`` property shape."""

    def __init__(self):
        self.offset = 0
        self.state = (mx.zeros((1, 1)), mx.zeros((1, 1)))


class _Draft:
    def __init__(self):
        self.config = SimpleNamespace(block_size=3)
        self.accept_lens = []
        self.draft_lens = []
        self.draft_block_calls = 0

    def set_shared_kv(self, *args, **kwargs):
        pass

    def reset(self, model):
        pass

    def draft_block(self, *args, **kwargs):
        self.draft_block_calls += 1
        return mx.array([[7, 8]], dtype=mx.int32)


class _LM:
    def rollback_speculative_cache(self, *args):
        pass


def _verify(width):
    return mtp_utils._MTPVerifyResult(
        hidden=mx.zeros((1, width, 2), dtype=mx.float32),
        shared_kv_states={},
        gdn_states=None,
    )


def _run_rounds(
    max_tokens=ROUNDS_MAX_TOKENS, prompt_cache=None, walk_return=(1, [9, 10])
):
    draft = _Draft()
    model = SimpleNamespace(language_model=_LM())
    cache = prompt_cache if prompt_cache is not None else [_CacheEntry()]
    with (
        patch.object(
            mtp_utils, "_mtp_verify_target", side_effect=lambda *a, **k: _verify(3)
        ),
        patch.object(mtp_utils, "_mtp_acceptance_walk", return_value=walk_return),
    ):
        toks = [
            t
            for t, _ in _mtp_rounds(
                model,
                draft,
                cache,
                mx.zeros((1, 1, 2), dtype=mx.float32),
                {},
                first_bonus=1,
                max_tokens=max_tokens,
                sampler=lambda logits: mx.argmax(logits, axis=-1),
                draft_block_size=3,
                token_dtype=mx.int32,
                greedy_sampling=True,
            )
        ]
    return toks, draft


@pytest.fixture(autouse=True)
def _reset_head():
    reset_head_profiler()
    yield
    reset_head_profiler()


class TestCollect:
    def test_cache_state_arrays_walks_tuple_state(self):
        entries = [_CacheEntry(), SimpleNamespace(state=None), SimpleNamespace()]
        arrays = cache_state_arrays(entries)
        assert len(arrays) == 2
        assert all(isinstance(a, mx.array) for a in arrays)


class TestMarkFenceOrder:
    """F4b: mark() must eval the phase's arrays BEFORE synchronizing -- MLX is
    lazy, so a sync with no preceding eval measures nothing meaningful for a
    marked-with-arrays phase."""

    def _recorders(self, monkeypatch):
        order = []
        real_eval, real_sync = mx.eval, mx.synchronize

        def rec_eval(*args, **kwargs):
            order.append("eval")
            return real_eval(*args, **kwargs)

        def rec_sync(*args, **kwargs):
            order.append("sync")
            return real_sync(*args, **kwargs)

        monkeypatch.setattr(mx, "eval", rec_eval)
        monkeypatch.setattr(mx, "synchronize", rec_sync)
        return order

    def test_mark_with_arrays_evals_immediately_before_synchronizing(self, monkeypatch):
        order = self._recorders(monkeypatch)
        timer = _PhaseTimer("t", ["p"], "unit")
        timer.begin()
        order.clear()
        timer.mark("p", mx.zeros((2, 2)))
        assert order == ["eval", "sync"]

    def test_mark_without_arrays_synchronizes_but_never_evals(self, monkeypatch):
        order = self._recorders(monkeypatch)
        timer = _PhaseTimer("t", ["p"], "unit")
        timer.begin()
        order.clear()
        timer.mark("p")
        assert order == ["sync"]


class TestRoundProfilerEnv:
    def test_env_set_emits_summary_line_with_all_fields(self, monkeypatch, capsys):
        monkeypatch.setenv(mtp_profile.ENV_ROUNDS, "1")
        toks, draft = _run_rounds()
        assert len(toks) == 6  # 3 rounds x 2 emitted tokens each (see comment above)

        err = capsys.readouterr().err
        lines = [l for l in err.splitlines() if l.startswith("[mtp_profile] ")]
        assert lines, f"no [mtp_profile] line in stderr: {err!r}"
        final_lines = [l for l in lines if l.endswith("final=1")]
        assert len(final_lines) == 1
        m = ROUND_LINE_RE.match(final_lines[0])
        assert m is not None, final_lines[0]
        fields = m.groupdict()

        assert fields["rounds"] == "3"
        assert fields["final"] == "1"
        for key in (
            "draft",
            "verify",
            "walk",
            "yield_",
            "accept",
            "rollback",
            "other",
            "emitted_rd",
            "accepted_rd",
            "n_draft_rd",
            "round_ms",
        ):
            value = float(fields[key])
            assert value == value  # not NaN
            assert value >= 0.0
            assert value != float("inf")
        # F4c: clean rates at max_tokens=7 (every round emitted 2, accepted 1,
        # drafted 2 -- see the ROUNDS_MAX_TOKENS comment).
        assert float(fields["emitted_rd"]) == 2.0
        assert float(fields["accepted_rd"]) == 1.0
        assert float(fields["n_draft_rd"]) == 2.0

    def test_env_unset_zero_extra_synchronize_calls(self, monkeypatch, capsys):
        monkeypatch.delenv(mtp_profile.ENV_ROUNDS, raising=False)
        real_sync = mx.synchronize
        calls = {"n": 0}

        def counting_sync(*args, **kwargs):
            calls["n"] += 1
            return real_sync(*args, **kwargs)

        monkeypatch.setattr(mx, "synchronize", counting_sync)
        _run_rounds()
        assert calls["n"] == 0
        err = capsys.readouterr().err
        assert "[mtp_profile]" not in err

    def test_env_set_exact_synchronize_count(self, monkeypatch, capsys):
        """F4a: pin the EXACT sync count so a neutered mark() (or one that
        drops its sync) cannot pass silently.

        Arithmetic for ROUNDS_MAX_TOKENS=7, block_size=3 (bs=3, n_draft=2
        every round), walk fake fixed at (1, [9, 10]):
          begin()                                          -> 1 sync
          round 1 (full):   other,draft,verify,walk,
                             yield,accept,rollback           -> 7 marks -> 7 syncs
          round 2 (full):   same 7 marks                     -> 7 syncs
          round 3 (early return at the 2nd yielded token,
                   i.e. i=1): other,draft,verify,walk,
                   yield (early-return branch)                -> 5 marks -> 5 syncs
                   (no accept/rollback -- the early return
                   happens before that code)
          report(final=True): no sync of its own
          total = 1 + 7 + 7 + 5 = 20
        """
        monkeypatch.setenv(mtp_profile.ENV_ROUNDS, "1")
        real_sync = mx.synchronize
        calls = {"n": 0}

        def counting_sync(*args, **kwargs):
            calls["n"] += 1
            return real_sync(*args, **kwargs)

        monkeypatch.setattr(mx, "synchronize", counting_sync)
        _run_rounds()
        assert calls["n"] == 20

    def test_every_2_emits_a_non_final_line_before_the_final_line(
        self, monkeypatch, capsys
    ):
        monkeypatch.setenv(mtp_profile.ENV_ROUNDS, "1")
        monkeypatch.setattr(MTPRoundProfiler, "every", 2)
        _run_rounds()

        err = capsys.readouterr().err
        lines = [l for l in err.splitlines() if l.startswith("[mtp_profile] ")]
        assert len(lines) >= 2
        non_final = [l for l in lines if l.endswith("final=0")]
        final = [l for l in lines if l.endswith("final=1")]
        assert non_final, lines
        assert final, lines
        assert err.index(non_final[0]) < err.index(final[0])
        m = ROUND_LINE_RE.match(non_final[0])
        assert m is not None
        assert m.group("rounds") == "2"
        m_final = ROUND_LINE_RE.match(final[0])
        assert m_final.group("rounds") == "3"

    def test_rollback_path_exercised_and_cache_state_seen(self, monkeypatch, capsys):
        monkeypatch.setenv(mtp_profile.ENV_ROUNDS, "1")
        cache = [_CacheEntry()]
        # bs=3 (block_size) and accepted=1 (fixed by the walk fake) => accepted
        # < bs - 1 on every FULL round (1 and 2; round 3 early-returns before
        # reaching the rollback check), so rollback_speculative_cache is
        # reached and cache_state_arrays(prompt_cache) sees the fake state.
        calls = {"n": 0}
        real_rollback = _LM.rollback_speculative_cache

        def counting_rollback(self, *args):
            calls["n"] += 1
            return real_rollback(self, *args)

        with patch.object(_LM, "rollback_speculative_cache", counting_rollback):
            _run_rounds(prompt_cache=cache)
        assert calls["n"] == 2

        err = capsys.readouterr().err
        final_line = [
            l
            for l in err.splitlines()
            if l.startswith("[mtp_profile] ") and l.endswith("final=1")
        ][0]
        m = ROUND_LINE_RE.match(final_line)
        assert float(m.group("rollback")) >= 0.0

    def test_rollback_mark_fires_vs_accept_all_run(self, monkeypatch):
        """F4e: rollback only fires when accepted < bs - 1. Compare a real
        rollback-path run (calls ``cache_state_arrays``, accepted=1 < bs-1=2)
        against an accept-all run (accepted=2 == bs-1=2, rollback never
        called) -- the accept-all run must call ``cache_state_arrays`` zero
        times, and the rollback-path run must call it exactly on the 2 full
        rounds (see the exact-count test above)."""
        monkeypatch.setenv(mtp_profile.ENV_ROUNDS, "1")
        calls = {"n": 0}
        real_collect = mtp_profile.cache_state_arrays

        def counting_collect(prompt_cache):
            calls["n"] += 1
            return real_collect(prompt_cache)

        with patch.object(mtp_profile, "cache_state_arrays", counting_collect):
            _run_rounds(walk_return=(1, [9, 10]))  # accepted=1 < bs-1=2
        assert calls["n"] == 2

        calls["n"] = 0
        with patch.object(mtp_profile, "cache_state_arrays", counting_collect):
            _run_rounds(walk_return=(2, [9, 10]))  # accepted=2 == bs-1=2: accept-all
        assert calls["n"] == 0


class TestHeadProfilerEnv:
    def _draft_block_smoke(self):
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
        return tokens

    def test_env_set_emits_head_summary_line(self, monkeypatch, capsys):
        monkeypatch.setenv(mtp_profile.ENV_HEAD, "1")
        self._draft_block_smoke()
        mtp_profile.head_profiler_from_env().report(final=True)

        err = capsys.readouterr().err
        lines = [l for l in err.splitlines() if l.startswith("[mtp_profile_head] ")]
        assert lines, f"no [mtp_profile_head] line in stderr: {err!r}"
        m = HEAD_LINE_RE.match(lines[-1])
        assert m is not None, lines[-1]
        fields = m.groupdict()
        assert fields["draft_tokens"] == "2"
        for key in ("proj_layers", "lm_head", "sampler", "eval"):
            value = float(fields[key])
            assert value == value
            assert value >= 0.0

    def test_env_unset_zero_extra_synchronize_calls(self, monkeypatch, capsys):
        monkeypatch.delenv(mtp_profile.ENV_HEAD, raising=False)
        real_sync = mx.synchronize
        calls = {"n": 0}

        def counting_sync(*args, **kwargs):
            calls["n"] += 1
            return real_sync(*args, **kwargs)

        monkeypatch.setattr(mx, "synchronize", counting_sync)
        self._draft_block_smoke()
        assert calls["n"] == 0
        assert mtp_profile.head_profiler_from_env() is None


class TestRoundFinalFlushesHead:
    def test_head_singleton_created_gets_final_flush_from_round_report(
        self, monkeypatch, capsys
    ):
        monkeypatch.setenv(mtp_profile.ENV_ROUNDS, "1")
        monkeypatch.setenv(mtp_profile.ENV_HEAD, "1")

        # round_profiler_from_env() resets the head singleton (F3), so the
        # head singleton must be created AFTER it, the way draft_block
        # creates it mid-round in the real flow.
        prof = mtp_profile.round_profiler_from_env()
        prof.begin()
        head = mtp_profile.head_profiler_from_env()
        head.begin()
        head.mark("eval", mx.zeros((1,)))
        head.end_unit()

        prof.report(final=True)

        err = capsys.readouterr().err
        round_final = [
            l
            for l in err.splitlines()
            if l.startswith("[mtp_profile] ") and l.endswith("final=1")
        ]
        head_final = [
            l
            for l in err.splitlines()
            if l.startswith("[mtp_profile_head] ") and l.endswith("final=1")
        ]
        assert round_final and head_final
        assert err.index(round_final[0]) < err.index(head_final[0])

    def test_no_head_singleton_does_not_crash(self, monkeypatch, capsys):
        monkeypatch.setenv(mtp_profile.ENV_ROUNDS, "1")
        monkeypatch.delenv(mtp_profile.ENV_HEAD, raising=False)
        reset_head_profiler()
        _run_rounds()  # must not raise
        err = capsys.readouterr().err
        assert "[mtp_profile_head]" not in err


class TestHeadNotCumulativeAcrossGenerations:
    def test_second_generation_head_line_counts_only_its_own_draft_tokens(
        self, monkeypatch, capsys
    ):
        """F3/F4d: two consecutive fake generations under both env vars --
        the second head final line must start from draft_tokens counted for
        that generation only, not carry generation 1's count forward."""
        monkeypatch.setenv(mtp_profile.ENV_ROUNDS, "1")
        monkeypatch.setenv(mtp_profile.ENV_HEAD, "1")

        # Generation 1: 3 simulated draft tokens.
        round1 = mtp_profile.round_profiler_from_env()
        round1.begin()
        head1 = mtp_profile.head_profiler_from_env()
        for _ in range(3):
            head1.begin()
            head1.mark("eval", mx.zeros((1,)))
            head1.end_unit()
        round1.report(final=True)
        capsys.readouterr()  # discard generation 1's lines

        # Generation 2: 2 simulated draft tokens. round_profiler_from_env()
        # must give generation 2 a fresh head singleton.
        round2 = mtp_profile.round_profiler_from_env()
        round2.begin()
        head2 = mtp_profile.head_profiler_from_env()
        assert head2 is not head1
        for _ in range(2):
            head2.begin()
            head2.mark("eval", mx.zeros((1,)))
            head2.end_unit()
        round2.report(final=True)

        err = capsys.readouterr().err
        head_final = [
            l
            for l in err.splitlines()
            if l.startswith("[mtp_profile_head] ") and l.endswith("final=1")
        ]
        assert len(head_final) == 1
        m = HEAD_LINE_RE.match(head_final[0])
        assert m is not None, head_final[0]
        assert m.group("draft_tokens") == "2"
