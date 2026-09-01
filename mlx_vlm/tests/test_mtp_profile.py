"""Env-gated MTP round/head profiler (M29 H1, 2026-08-31).

CPU-pinned (no GPU, no real model): the round-loop tests use the same fakes
as ``test_mtp_thinking_budget.py`` (patched ``_mtp_verify_target`` /
``_mtp_acceptance_walk``); the head-loop tests reuse
``test_nemotron_h_mtp.py``'s tiny synthetic drafter/target helpers.
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

# 3 full rounds (2 emitted tokens each, from the fixed (1, [9, 10]) walk fake)
# plus a 4th partial round whose yield loop hits max_tokens and returns early
# (skipping that round's rollback + end_unit -- see mtp.py's _mtp_rounds).
ROUNDS_MAX_TOKENS = 8


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


def _run_rounds(max_tokens=ROUNDS_MAX_TOKENS, prompt_cache=None):
    draft = _Draft()
    model = SimpleNamespace(language_model=_LM())
    cache = prompt_cache if prompt_cache is not None else [_CacheEntry()]
    with (
        patch.object(
            mtp_utils, "_mtp_verify_target", side_effect=lambda *a, **k: _verify(3)
        ),
        patch.object(mtp_utils, "_mtp_acceptance_walk", return_value=(1, [9, 10])),
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


class TestRoundProfilerEnv:
    def test_env_set_emits_summary_line_with_all_fields(self, monkeypatch, capsys):
        monkeypatch.setenv(mtp_profile.ENV_ROUNDS, "1")
        toks, draft = _run_rounds()
        assert len(toks) == 7  # 3 full rounds x 2 + 1 partial round x 1

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
        assert float(fields["emitted_rd"]) == 2.0

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
        # < bs - 1 on every full round, so rollback_speculative_cache is
        # reached and cache_state_arrays(prompt_cache) sees the fake state.
        calls = {"n": 0}
        real_rollback = _LM.rollback_speculative_cache

        def counting_rollback(self, *args):
            calls["n"] += 1
            return real_rollback(self, *args)

        with patch.object(_LM, "rollback_speculative_cache", counting_rollback):
            _run_rounds(prompt_cache=cache)
        assert calls["n"] >= 1

        err = capsys.readouterr().err
        final_line = [
            l
            for l in err.splitlines()
            if l.startswith("[mtp_profile] ") and l.endswith("final=1")
        ][0]
        m = ROUND_LINE_RE.match(final_line)
        assert float(m.group("rollback")) >= 0.0


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
        # Create the head singleton the way draft_block would.
        head = mtp_profile.head_profiler_from_env()
        head.begin()
        head.mark("eval", mx.zeros((1,)))
        head.end_unit()

        _run_rounds()

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
