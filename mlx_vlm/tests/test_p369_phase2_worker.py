"""P369 phase 2: the worker's CPU-testable parts and the verifier key-length recorder.

Fork-only file. The worker boot (the real CLI with ``uvicorn.run`` intercepted, a real model) is exercised by
the GPU smoke; here: exact prompt sizing by dry-run counts, the row / snapshot artifacts, frozen verify-scan
counters around probes, the served-state and source checks, atomic results, and ``KeyLengths`` (every
verifier block's key-length span, from which straddles of each MLX threshold are proven).
"""

import json
import sys
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from test_mtp_exactness import (
    PICKS,
    PROMPT,
    _quantized_pair,
    deployed_sampling,
    fake_model,
    scripted,
)
from test_mtp_stop_state import turn

from mlx_vlm.generate.common import PromptCacheState

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import compare, observe, worker  # noqa: E402

metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="exact kernels are Metal"
)


# ------------------------------------------------------------------ prompt sizing
def test_size_prompt_hits_an_exact_count_with_nudges():
    def count(text):  # two tokens per word, one per trailing nudge char
        words = text.split()
        return sum(1 if len(w) == 1 else 2 for w in words) + 7

    text, tokens = worker.size_prompt(count, 1001, prefix="P", suffix="S")
    assert tokens == 1001 and count(text) == 1001


def test_size_prompt_reports_an_unreachable_target():
    count = lambda text: 3 * len(text.split())  # noqa: E731 only multiples of 3
    text, closest = worker.size_prompt(count, 1000, prefix="P", suffix="S")
    assert text is None and abs(closest - 1000) <= 2


# ------------------------------------------------------------------ artifacts
def test_rows_round_trip_with_their_exact_bits(tmp_path):
    rows = [
        (p, compare.as_record(mx.random.normal((7,)).astype(mx.bfloat16)))
        for p in (0, 1, 1)
    ]
    worker.save_rows(tmp_path / "r.npz", rows)
    back = worker.load_rows(tmp_path / "r.npz")
    assert [p for p, _ in back] == [0, 1, 1]
    for (_, a), (_, b) in zip(rows, back):
        assert a["dtype"] == b["dtype"] == "bfloat16"
        assert np.array_equal(a["bits"], b["bits"])


def test_rows_refuse_mixed_dtypes(tmp_path):
    rows = [
        (0, compare.as_record(mx.zeros((3,), dtype=mx.bfloat16))),
        (1, compare.as_record(mx.zeros((3,), dtype=mx.float32))),
    ]
    with pytest.raises(RuntimeError, match="mixed row dtypes"):
        worker.save_rows(tmp_path / "r.npz", rows)


def test_snapshots_round_trip(tmp_path):
    probe = compare.as_record(mx.arange(5).astype(mx.float32))
    snaps = [
        dict(offset=8, sha="a" * 64, neutral=True, kind="prompt_end", probe=probe),
        dict(offset=11, sha="b" * 64, neutral=False, kind="block", probe=probe),
    ]
    worker.save_snapshots(tmp_path / "s.npz", snaps)
    back = worker.load_snapshots(tmp_path / "s.npz")
    for a, b in zip(snaps, back):
        assert (a["offset"], a["sha"], a["neutral"], a["kind"]) == (
            b["offset"],
            b["sha"],
            b["neutral"],
            b["kind"],
        )
        assert np.array_equal(a["probe"]["bits"], b["probe"]["bits"])


def test_probe_counters_frozen_restores_production_counts():
    policy = types.SimpleNamespace(
        _counts={"verify_blocks_len1": 4, "verify_blocks_straddle": 1}
    )
    with worker.probe_counters_frozen(policy):
        policy._counts["verify_blocks_len1"] += 1
    assert policy._counts == {"verify_blocks_len1": 4, "verify_blocks_straddle": 1}
    with worker.probe_counters_frozen(None):
        pass


# ------------------------------------------------------------------ served state / sources / atomic
GOOD = dict(
    drafter_loaded=True,
    draft_kind="mtp",
    verify_policy="joint_v1",
    apc_manager=False,
    apc_env=None,
    session_max=1,
    session_max_env="1",
)
EXPECT = dict(drafter=True, verify_policy="joint_v1")


def test_assert_served_accepts_the_expected_state():
    worker.assert_served(GOOD, EXPECT)


@pytest.mark.parametrize(
    "change",
    [
        dict(drafter_loaded=False),
        dict(draft_kind="suffix"),
        dict(verify_policy=None),
        dict(apc_manager=True),
        dict(apc_env="1"),
        dict(session_max=2),
        dict(session_max_env="2"),
    ],
)
def test_assert_served_refuses_any_mismatch(change):
    with pytest.raises(RuntimeError, match="served state mismatch"):
        worker.assert_served(dict(GOOD, **change), EXPECT)


def test_check_sources_refuses_a_module_outside_the_reviewed_tree(
    tmp_path, monkeypatch
):
    root = Path(worker.__file__).resolve().parents[2]
    worker.check_sources(root, DEV)
    stray = types.ModuleType("mlx_vlm._p369_stray")
    stray.__file__ = str(tmp_path / "elsewhere.py")
    monkeypatch.setitem(sys.modules, "mlx_vlm._p369_stray", stray)
    with pytest.raises(RuntimeError, match="outside the reviewed sources"):
        worker.check_sources(root, DEV)


def test_atomic_write_leaves_only_the_final_file(tmp_path):
    worker.atomic_write(tmp_path / "result.json", {"a": 1})
    assert json.loads((tmp_path / "result.json").read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


# ------------------------------------------------------------------ KeyLengths
@metal
def test_key_lengths_records_every_verifier_block_span():
    lm, drafter = _quantized_pair(PICKS[1])
    model, params = fake_model(lm), deployed_sampling(PICKS[1])
    plain, _ = turn(model, PROMPT, PromptCacheState(), n=9, seed=5, params=params)
    plain = list(plain)
    scripted(
        drafter, lambda p: plain[p] if p < len(plain) else None, high=True, vocab=64
    )
    with observe.KeyLengths() as kl:
        turn(
            model,
            PROMPT,
            PromptCacheState(),
            drafter=drafter,
            n=9,
            seed=5,
            params=params,
        )
    assert kl.blocks and all(b["width"] > 1 for b in kl.blocks)
    first = kl.blocks[0]
    assert (
        first["first_key"] == len(PROMPT) + 1
        and first["last_key"] == len(PROMPT) + first["width"]
    )
    t = first["first_key"]  # a threshold the first block spans
    assert observe.straddles(kl.blocks, t) or first["width"] == 1
    assert not observe.straddles(kl.blocks, 10_000)


# ------------------------------------------------------------------ smoke positives (Q32)
@metal
@pytest.mark.parametrize("pick", PICKS)
def test_smoke_positives_fire_on_live_data(pick):
    lm, drafter = _quantized_pair(pick)
    model, params = fake_model(lm), deployed_sampling(pick)
    plain, _ = turn(model, PROMPT, PromptCacheState(), n=9, seed=5, params=params)
    plain = list(plain)
    scripted(
        drafter, lambda p: plain[p] if p < len(plain) else None, high=True, vocab=64
    )
    with observe.Recorder(mode="on", lm=lm, probe_token=5) as on:
        emitted, _ = turn(
            model,
            PROMPT,
            PromptCacheState(),
            drafter=drafter,
            n=9,
            seed=5,
            params=params,
        )
    offsets = [on.prompt_end["offset"]] + [c["offset"] for c in on.commits]
    with observe.Recorder(
        mode="off",
        force_tokens=list(emitted),
        hash_offsets=offsets,
        lm=lm,
        probe_token=5,
    ) as off:
        turn(model, PROMPT, PromptCacheState(), n=len(emitted), seed=5, params=params)
    got = worker.smoke_positives(on, off, len(emitted), params)
    assert got["baseline_pass"], got
    assert all(got["fired"].values()), got
    assert {"raw_ulp", "in_support", "old_prefix_k", "recurrent", "probe_logit"} <= set(
        got["fired"]
    )


# ------------------------------------------------------------------ same-instance pairing (P391)
def test_drafter_detached_restores_the_same_objects():
    drafter = object()
    rg = types.SimpleNamespace(draft_model=drafter, draft_kind="mtp")
    with worker.drafter_detached(rg):
        assert rg.draft_model is None and rg.draft_kind is None
    assert rg.draft_model is drafter and rg.draft_kind == "mtp"
    with pytest.raises(ZeroDivisionError):
        with worker.drafter_detached(rg):
            1 / 0
    assert rg.draft_model is drafter and rg.draft_kind == "mtp"


def test_snapshot_holder_splits_prompt_end_and_commits(tmp_path):
    probe = compare.as_record(mx.arange(3).astype(mx.float32))
    snaps = [
        dict(offset=8, sha="a" * 64, neutral=True, kind="prompt_end", probe=probe),
        dict(offset=11, sha="b" * 64, neutral=True, kind="block", probe=probe),
    ]
    worker.save_snapshots(tmp_path / "s.npz", snaps)
    holder = worker.snapshot_holder(worker.load_snapshots(tmp_path / "s.npz"), rows=[])
    assert holder.prompt_end["offset"] == 8 and [
        c["offset"] for c in holder.commits
    ] == [11]
    plain = [dict(s, kind="plain") for s in snaps]
    worker.save_snapshots(tmp_path / "p.npz", plain)
    holder = worker.snapshot_holder(worker.load_snapshots(tmp_path / "p.npz"))
    assert holder.prompt_end is None and [c["offset"] for c in holder.commits] == [
        8,
        11,
    ]


# ------------------------------------------------------------------ round 5 (Q35 / Q37 / Q38)
def test_wait_idle_waits_for_the_queue_and_in_flight_and_times_out():
    calls = {"n": 0}

    class Q:
        def qsize(self):
            calls["n"] += 1
            return 0 if calls["n"] > 2 else 1

    metrics = types.SimpleNamespace(_in_flight=0)
    worker.wait_idle(
        types.SimpleNamespace(requests=Q()), metrics, timeout_s=2, poll_s=0.001
    )
    assert calls["n"] >= 3
    busy = types.SimpleNamespace(_in_flight=1)
    with pytest.raises(RuntimeError, match="did not go idle"):
        worker.wait_idle(
            types.SimpleNamespace(requests=types.SimpleNamespace(qsize=lambda: 0)),
            busy,
            timeout_s=0.01,
            poll_s=0.001,
        )


def test_detached_guard_refuses_draft_kwargs_or_speculative_prefill():
    from mlx_vlm.generate import ar
    from mlx_vlm.server import generation

    with worker.DetachedGuard() as guard:
        assert generation._inline_draft_kwargs(None, None, None) == {}
    assert guard.problems == []
    with worker.DetachedGuard() as guard:
        ar.speculative_prefill_kwargs("mtp", None)
    assert any("speculative prefill" in p for p in guard.problems)


def _part(tokens, finish="length", cached=0):
    return dict(
        final=dict(token_ids=list(tokens), finish_reason=finish, cached_tokens=cached)
    )


EXPECTED = [8, 11, 13]


def test_c1_acceptance():
    ok = dict(rows={"pass": True}, states={"pass": True})
    on, offs = _part([1, 2, 3]), [_part([1, 2, 3]), _part([1, 2, 3])]
    full = [EXPECTED, EXPECTED]
    assert worker.c1_acceptance(
        on, offs, ok, retired=True, expected_offsets=EXPECTED, replay_offsets=full
    )["pass"]
    cases = [
        (on, [_part([1, 2]), _part([1, 2, 3])], ok, True, full, "forced tokens"),
        (
            on,
            [_part([1, 2, 3], finish="stop"), _part([1, 2, 3])],
            ok,
            True,
            full,
            "finish",
        ),
        (on, [_part([1, 2, 3], cached=4), _part([1, 2, 3])], ok, True, full, "reused"),
        (on, offs, dict(ok, states={"pass": False}), True, full, "states"),
        (on, offs, ok, False, full, "not retired"),
        # code round 1 H4: both replays missing the same ON offset is a C1 failure (ABORT), not an E1c FAIL
        (on, offs, ok, True, [[8, 13], [8, 13]], "OFF1 offsets"),
        (on, offs, ok, True, [EXPECTED, [8, 11]], "OFF2 offsets"),
        (on, offs, ok, True, [EXPECTED + [15], EXPECTED], "OFF1 offsets"),
    ]
    for on_, offs_, cmp_, ids, replay, why in cases:
        got = worker.c1_acceptance(
            on_,
            offs_,
            cmp_,
            retired=ids,
            expected_offsets=EXPECTED,
            replay_offsets=replay,
        )
        assert not got["pass"] and any(why in p for p in got["problems"]), (why, got)


# ------------------------------------------------------------------ round 6 (Q35 / Q38)
def test_capture_acknowledges_every_cached_request_exit(monkeypatch):
    from mlx_vlm.server import generation

    calls = []

    def fake_process(rg, *args, **kwargs):
        calls.append(1)
        if kwargs.get("boom"):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        generation.ResponseGenerator, "_process_cached_request", fake_process
    )
    with worker.Capture() as capture:
        generation.ResponseGenerator._process_cached_request(object())
        with pytest.raises(RuntimeError):
            generation.ResponseGenerator._process_cached_request(object(), boom=True)
        assert capture.completed == 2
    assert generation.ResponseGenerator._process_cached_request is fake_process


def test_wait_ack_waits_for_the_next_exit():
    import threading

    capture = types.SimpleNamespace(completed=3)
    threading.Timer(0.05, lambda: setattr(capture, "completed", 4)).start()
    worker.wait_ack(capture, 3, timeout_s=2)
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        worker.wait_ack(capture, 4, timeout_s=0.05)


def _production_session(chat_id):
    """A session entry exactly as the server makes it (``{"state", "last_used", "turn_hashes"}``)."""
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.server import session_manager

    state = session_manager.get_or_create_prompt_cache_state(chat_id)
    state.cache = [KVCache(), KVCache()]
    state.token_ids = [1, 2]
    return state


def test_release_trajectory_retires_the_cache_tree(monkeypatch):
    from mlx_vlm.server import session_manager

    monkeypatch.setattr(session_manager, "_session_cache_max", 4)
    monkeypatch.setattr(
        session_manager, "_session_caches", type(session_manager._session_caches)()
    )
    cleared = []
    real_clear = PromptCacheState.clear
    monkeypatch.setattr(
        PromptCacheState, "clear", lambda self: (cleared.append(self), real_clear(self))
    )
    state = _production_session("p369-x")
    assert isinstance(session_manager._session_caches["p369-x"], dict)
    rec = types.SimpleNamespace(_cache=state.cache)
    worker.release_trajectory(rec, "p369-x")
    assert "p369-x" not in session_manager._session_caches
    assert cleared == [
        state
    ]  # code round 1 H13: the PromptCacheState itself is cleared
    assert rec._cache is None and state.cache is None and state.token_ids is None
    held = _production_session("p369-y")
    lingering = held.cache  # someone else still holds the tree
    rec = types.SimpleNamespace(_cache=held.cache)
    with pytest.raises(RuntimeError, match="not retired"):
        worker.release_trajectory(rec, "p369-y")
    del lingering


def test_release_session_clears_the_production_entry(monkeypatch):
    from mlx_vlm.server import session_manager

    monkeypatch.setattr(session_manager, "_session_cache_max", 4)
    monkeypatch.setattr(
        session_manager, "_session_caches", type(session_manager._session_caches)()
    )
    state = _production_session("p369-z")
    worker.release_session("p369-z")
    assert "p369-z" not in session_manager._session_caches
    assert state.cache is None and state.token_ids is None


def test_release_refuses_an_unexpected_session_entry_shape(monkeypatch):
    from mlx_vlm.server import session_manager

    monkeypatch.setattr(
        session_manager, "_session_caches", type(session_manager._session_caches)()
    )
    session_manager._session_caches["p369-odd"] = (
        PromptCacheState()
    )  # not the production shape
    with pytest.raises(RuntimeError, match="unexpected session entry"):
        worker.release_session("p369-odd")


# ------------------------------------------------------------------ code round 1 H8: the request heartbeat
def test_progress_records_the_request_and_its_deadline_atomically(tmp_path):
    import time

    before = time.time()
    worker.write_progress(tmp_path, index=3, name="e7a/x", deadline_s=120.0)
    got = json.loads((tmp_path / "progress.json").read_text())
    assert (got["index"], got["name"], got["deadline_s"]) == (3, "e7a/x", 120.0)
    assert before <= got["started"] <= time.time()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["progress.json"]


# ------------------------------------------------------------------ code round 1 H6 H7: prompt digests, E7b continuity
def test_token_digest_is_exact_over_order_values_and_length():
    d = worker.token_digest
    assert d([1, 2, 3]) == d(np.array([[1, 2, 3]])) == d(mx.array([1, 2, 3]))
    assert len({d([1, 2, 3]), d([1, 2, 4]), d([1, 2]), d([2, 1, 3])}) == 4


def test_capture_records_each_requests_prompt_digest(monkeypatch):
    from mlx_vlm.server import generation

    monkeypatch.setattr(
        generation.ResponseGenerator,
        "_preprocess_request",
        lambda rg, prompt, *a, **k: dict(input_ids=np.array([[5, 6, 7]])),
    )
    with worker.Capture() as capture:
        generation.ResponseGenerator._preprocess_request(object(), "p")
        assert capture.prompt == dict(
            tokens=3, sha256=worker.token_digest([5, 6, 7]), ids=[5, 6, 7]
        )
        capture.reset()
        assert capture.prompt is None


def test_assistant_echo_returns_the_complete_turn():
    full = dict(role="assistant", content="A", reasoning_content="R", tool_calls=None)
    assert worker.assistant_echo(full) == dict(
        role="assistant", content="A", reasoning_content="R"
    )
    assert worker.assistant_echo(dict(content=None)) == dict(
        role="assistant", content=""
    )
    calls = [dict(id="c1", type="function", function=dict(name="f", arguments="{}"))]
    assert (
        worker.assistant_echo(dict(content="", tool_calls=calls))["tool_calls"] == calls
    )


def test_turn_continuity():
    prev = [1, 2, 3, 4, 5, 6]  # prompt 1..4, generated 5, 6
    hashes = ["h0", "h1"]
    ok = worker.turn_continuity(
        prev_ids=prev,
        prev_hashes=hashes,
        prev_prompt_tokens=4,
        new_ids=prev + [7, 8],
        new_hashes=hashes + ["h2", "h3"],
        cached_tokens=5,
    )
    assert ok["continuity"] and (ok["lcp"], ok["boundary"], ok["generated_prev"]) == (
        6,
        6,
        2,
    )
    cases = [
        (dict(new_ids=[1, 2, 3, 4, 9, 6, 7]), "does not reproduce"),
        (dict(new_hashes=["h0", "hX", "h2"]), "hash chain"),
        (dict(cached_tokens=3), "reuse stops"),
        (dict(prev_ids=[1, 2, 3, 4], new_ids=[1, 2, 3, 4, 7]), "no generated tokens"),
    ]
    base = dict(
        prev_ids=prev,
        prev_hashes=hashes,
        prev_prompt_tokens=4,
        new_ids=prev + [7, 8],
        new_hashes=hashes + ["h2", "h3"],
        cached_tokens=5,
    )
    for change, why in cases:
        got = worker.turn_continuity(**dict(base, **change))
        assert not got["continuity"] and why in got["reason"], (change, got)


# ------------------------------------------------------------------ code round 2 H9: the snapshots the worker loaded
def test_check_snapshots_compares_the_resolved_revision_of_each_model_flag(tmp_path):
    snap = tmp_path / "models--o--m" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    resolve = lambda repo: snap  # noqa: E731
    argv = ["--model", "o/m", "--draft-model", "o/m", "--port", "1"]
    got = worker.check_snapshots(
        argv, {"--model": "rev1", "--draft-model": "rev1"}, resolve
    )
    assert got == {"--model": "rev1", "--draft-model": "rev1"}
    with pytest.raises(RuntimeError, match="snapshot"):
        worker.check_snapshots(
            argv, {"--model": "rev0", "--draft-model": "rev1"}, resolve
        )
    with pytest.raises(RuntimeError, match="snapshot"):
        worker.check_snapshots(
            ["--model", "o/m"], {"--model": "rev1", "--draft-model": "rev1"}, resolve
        )


# ------------------------------------------------------------------ code round 2 H8: the observation lands before cleanup
def test_finish_request_writes_the_entry_and_the_after_phase_before_cleanup(tmp_path):
    seen = []

    def cleanup():
        progress = json.loads((tmp_path / "progress.json").read_text())
        seen.append(((tmp_path / "req-2.json").exists(), progress["phase"]))

    worker.finish_request(tmp_path, 2, dict(name="r2", x=1), "r2", [cleanup, cleanup])
    assert seen == [(True, "after"), (True, "after")]
    assert json.loads((tmp_path / "req-2.json").read_text())["x"] == 1


def test_check_snapshots_rehashes_the_loaded_files(tmp_path):
    import hashlib

    snap = tmp_path / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"weights")
    (snap / "config.json").write_bytes(b"{}")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.iterdir()}
    argv = ["--model", "o/m"]
    want = {"--model": dict(revision="rev1", files=files)}
    assert worker.check_snapshots(argv, want, lambda repo: snap) == {"--model": "rev1"}
    (snap / "model.safetensors").write_bytes(b"mutated, same revision")
    with pytest.raises(RuntimeError, match="model.safetensors"):
        worker.check_snapshots(argv, want, lambda repo: snap)
