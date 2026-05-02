"""Tests for the rewind-guard helpers in mlx_vlm.generate.

Covers memory.md #13 (SWA RotatingKVCache rewind safety), #13a (hybrid-cache
rewind guard + DeltaNet state restore), and the PromptCacheState divergence
walker that invalidates stale snapshots when the cached token sequence is
overwritten mid-prefix.

These guard silent-correctness invariants — the failure modes are wrong
generation outputs (no exception), so unit tests are the only way to catch
regressions before they reach inference.
"""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from mlx_vlm.generate import (
    PromptCacheState,
    _adjust_chunk_for_snapshot_landing,
    _anchor_within_loop_range,
    _capture_anchor_state,
    _capture_arrays_layers_for_snapshot,
    _capture_rotating_layers_for_snapshot,
    _classify_snapshot_action,
    _compute_anchor_before_latest_user_offset,
    _first_kv_offset,
    _has_non_trimmable,
    _is_rotating_kv_layer,
    _restore_arrays_layers_from_snapshots,
    _restore_deltanet_state,
    _restore_rotating_layers_from_snapshots,
    _rotating_post_gen_trim_safe,
    _rotating_rewind_safe,
    _should_capture_anchor_pre_prefill,
    _trim_cache,
)
from mlx_vlm.snapshot import DeltaNetSnapshotRing, capture_rotating


def _arrays_cache(num_arrays: int = 2, fill: float = 0.0) -> ArraysCache:
    c = ArraysCache(num_arrays)
    c.state = [mx.full((1, 4), fill) for _ in range(num_arrays)]
    return c


def _rotating(offset: int, max_size: int) -> RotatingKVCache:
    """Construct a RotatingKVCache with explicit offset/max_size for the
    rewind-safety check. The real cache populates offset via update_and_fetch;
    setting it directly is sufficient because the guard only reads the two
    attributes.
    """
    r = RotatingKVCache(max_size=max_size)
    r.offset = offset
    return r


class _Container:
    """Fake nested cache container exposing a ``caches`` attribute, mimicking
    mlx_lm composite caches (e.g. ChunkedKVCache wrappers) that the rewind
    helpers must recurse into.
    """

    def __init__(self, *children):
        self.caches = list(children)


class TestRotatingRewindSafe:
    def test_empty_entries_is_safe(self):
        assert _rotating_rewind_safe([], target_len=0) is True

    def test_plain_kv_cache_is_skipped(self):
        # Standard KVCache is not a ring buffer, so the guard ignores it.
        assert _rotating_rewind_safe([KVCache()], target_len=10) is True

    def test_unwrapped_buffer_is_safe_at_any_target(self):
        # offset (50) <= max_size (100) → buffer hasn't wrapped, every
        # position is still on disk. Any target_len is safe.
        rkv = _rotating(offset=50, max_size=100)
        assert _rotating_rewind_safe([rkv], target_len=0) is True
        assert _rotating_rewind_safe([rkv], target_len=49) is True

    def test_wrapped_buffer_safe_when_target_in_window(self):
        # offset=150, max_size=100 → tokens 0..49 are overwritten, 50..149
        # are still alive. Targets >= 50 are safe.
        rkv = _rotating(offset=150, max_size=100)
        assert _rotating_rewind_safe([rkv], target_len=50) is True
        assert _rotating_rewind_safe([rkv], target_len=100) is True

    def test_wrapped_buffer_unsafe_when_target_overwritten(self):
        # Same wrapped buffer; target 49 falls in the discarded region.
        rkv = _rotating(offset=150, max_size=100)
        assert _rotating_rewind_safe([rkv], target_len=49) is False
        assert _rotating_rewind_safe([rkv], target_len=0) is False

    def test_offset_as_mx_array_with_item(self):
        # The guard handles mx.array offsets by calling .item() — some
        # cache implementations store offset as a 0-dim mx.array rather
        # than a Python int.
        rkv = RotatingKVCache(max_size=100)
        rkv.offset = mx.array(150)
        assert _rotating_rewind_safe([rkv], target_len=49) is False
        assert _rotating_rewind_safe([rkv], target_len=80) is True

    def test_recurses_into_caches_attribute(self):
        # Composite cache wrapper exposes nested layers via .caches.
        unsafe_inner = _rotating(offset=200, max_size=100)
        wrapper = _Container(unsafe_inner)
        assert _rotating_rewind_safe([wrapper], target_len=50) is False
        assert _rotating_rewind_safe([wrapper], target_len=150) is True

    def test_recurses_into_lists_and_tuples(self):
        unsafe = _rotating(offset=200, max_size=100)
        nested_list = [[KVCache(), unsafe]]
        nested_tuple = ((KVCache(), unsafe),)
        assert _rotating_rewind_safe(nested_list, target_len=50) is False
        assert _rotating_rewind_safe(nested_tuple, target_len=50) is False

    def test_mixed_safe_and_unsafe_returns_unsafe(self):
        # Single unsafe layer poisons the whole batch — full re-prefill.
        safe = _rotating(offset=50, max_size=100)
        unsafe = _rotating(offset=200, max_size=100)
        assert _rotating_rewind_safe([safe, unsafe], target_len=49) is False


class TestHasNonTrimmable:
    def test_pure_attention_returns_false(self):
        assert _has_non_trimmable([KVCache(), KVCache()]) is False

    def test_arrays_cache_returns_true(self):
        assert _has_non_trimmable([_arrays_cache()]) is True

    def test_mixed_returns_true(self):
        assert _has_non_trimmable([KVCache(), _arrays_cache(), KVCache()]) is True

    def test_recurses_into_caches_attribute(self):
        wrapper = _Container(KVCache(), _arrays_cache())
        assert _has_non_trimmable([wrapper]) is True

    def test_recurses_into_lists_and_tuples(self):
        nested = [[KVCache(), _arrays_cache()]]
        assert _has_non_trimmable(nested) is True

    def test_empty_returns_false(self):
        assert _has_non_trimmable([]) is False


class TestFirstKvOffset:
    """Pins the helper used by chunked prefill to read "tokens already in
    cache" before the loop starts. Pre-fix code blindly read
    ``prompt_cache[0].offset`` which crashed on hybrid topologies
    (Qwen 3.5/3.6 GatedDeltaNet, Mamba) where layer 0 is an
    ``ArraysCache`` with no ``offset`` attribute.
    """

    def test_empty_returns_zero(self):
        assert _first_kv_offset([]) == 0

    def test_kv_at_index_zero(self):
        # Fresh KVCache reports offset 0 — same as "no prior cache".
        assert _first_kv_offset([KVCache()]) == 0

    def test_arrays_cache_at_index_zero_skipped(self):
        # Production crash repro: hybrid model with ArraysCache at
        # layer 0; helper must walk past it and read the KV layer.
        entries = [_arrays_cache(), _rotating(offset=128, max_size=1024)]
        assert _first_kv_offset(entries) == 128

    def test_all_recurrent_returns_zero(self):
        assert _first_kv_offset([_arrays_cache(), _arrays_cache()]) == 0

    def test_recurses_into_caches_attribute(self):
        wrapper = _Container(_arrays_cache(), _rotating(offset=64, max_size=512))
        assert _first_kv_offset([wrapper]) == 64

    def test_recurses_into_lists_and_tuples(self):
        nested = [[_arrays_cache(), _rotating(offset=32, max_size=256)]]
        assert _first_kv_offset(nested) == 32

    def test_first_kv_layer_wins(self):
        # KV layers advance in lockstep so this is a safety check
        # only — the first hit should be returned without scanning
        # further.
        first = _rotating(offset=100, max_size=1024)
        second = _rotating(offset=200, max_size=1024)
        assert _first_kv_offset([first, second]) == 100


class TestRestoreDeltanetState:
    def test_restores_only_arrays_cache_layers(self):
        ac = _arrays_cache(fill=0.0)
        kv = KVCache()
        # Snapshot states aligned to entries: [None for KV, list for ArraysCache]
        new_state = [mx.full((1, 4), 7.0), mx.full((1, 4), 7.0)]
        snapshot_states = [None, new_state]

        _restore_deltanet_state([kv, ac], snapshot_states)

        # ArraysCache.state replaced.
        for arr in ac.state:
            assert mx.array_equal(arr, mx.full((1, 4), 7.0)).item()

    def test_skips_none_snapshot_entries(self):
        # If snapshot says None for a layer (KV layer), the corresponding
        # cache entry is left untouched even if it happens to be ArraysCache.
        ac = _arrays_cache(fill=1.0)
        original = list(ac.state)
        _restore_deltanet_state([ac], [None])
        for before, after in zip(original, ac.state):
            assert before is after

    def test_skips_non_arrays_cache_when_state_present(self):
        # Defensive: if snapshot has a state list at a position that turns
        # out to map to a non-ArraysCache (cache topology changed across
        # turns), the helper must NOT crash and must NOT clobber.
        kv = KVCache()
        snapshot_states = [[mx.zeros((1, 4))]]
        # Should not raise.
        _restore_deltanet_state([kv], snapshot_states)


class TestPromptCacheStateUpdate:
    def test_first_update_captures_snapshot(self):
        ring = DeltaNetSnapshotRing(max_size=3)
        state = PromptCacheState(snapshot_ring=ring)
        ac = _arrays_cache()

        state.update(token_ids=[1, 2, 3, 4, 5], kv_cache=[ac])

        assert state.token_ids == [1, 2, 3, 4, 5]
        assert state.cache == [ac]
        assert len(ring) == 1

    def test_pure_extension_keeps_snapshots(self):
        # New token sequence is a strict extension — divergence equals
        # the old length, so drop_after is a no-op (snapshot offset
        # equals divergence and drop_after keeps equal-offset entries).
        ring = DeltaNetSnapshotRing(max_size=4)
        state = PromptCacheState(snapshot_ring=ring)
        ac = _arrays_cache()

        state.update(token_ids=[1, 2, 3], kv_cache=[ac])
        state.update(token_ids=[1, 2, 3, 4, 5], kv_cache=[ac])

        # Both snapshots survived, latest captured.
        assert {s.offset for s in ring._snapshots} == {3, 5}

    def test_mid_prefix_divergence_drops_stale_snapshots(self):
        # Concrete reproduction of the silent-correctness bug fixed by
        # drop_after: turn N captures snap@offset=20 from a long cached
        # sequence; turn N+1 arrives with a shorter sequence that
        # diverges at offset=10. The snap@20 references DeltaNet state
        # conditioned on tokens that no longer exist in the live cache.
        # If left in the ring, a later rewind to ~20 would restore stale
        # state into a fresh cache → silently wrong outputs.
        ring = DeltaNetSnapshotRing(max_size=4)
        state = PromptCacheState(snapshot_ring=ring)
        ac = _arrays_cache()

        state.update(token_ids=list(range(20)), kv_cache=[ac])
        assert {s.offset for s in ring._snapshots} == {20}

        # New sequence diverges at index 10.
        new_ids = list(range(10)) + [99, 99, 99]
        state.update(token_ids=new_ids, kv_cache=[ac])

        offsets = {s.offset for s in ring._snapshots}
        # Stale snap@20 dropped; new snap@13 captured.
        assert 20 not in offsets
        assert 13 in offsets

    def test_full_replacement_drops_all_old_snapshots(self):
        # Divergence at offset 0 → every prior snapshot is stale.
        ring = DeltaNetSnapshotRing(max_size=4)
        state = PromptCacheState(snapshot_ring=ring)
        ac = _arrays_cache()

        state.update(token_ids=[1, 2, 3, 4, 5], kv_cache=[ac])
        state.update(token_ids=[9, 9, 9], kv_cache=[ac])

        # Only the new snap@3 remains; the old snap@5 was conditioned
        # on tokens [1..5] which no longer exist in the live cache.
        offsets = {s.offset for s in ring._snapshots}
        assert offsets == {3}

    def test_disabled_ring_skips_capture_and_drop(self):
        ring = DeltaNetSnapshotRing(max_size=0)
        state = PromptCacheState(snapshot_ring=ring)
        ac = _arrays_cache()

        # Should not raise even though ring is disabled.
        state.update(token_ids=[1, 2, 3], kv_cache=[ac])
        state.update(token_ids=[1, 2, 9], kv_cache=[ac])

        assert len(ring) == 0
        assert state.token_ids == [1, 2, 9]

    def test_pure_attention_cache_skips_capture(self):
        # Pure-attention model: no ArraysCache → capture is a no-op even
        # though the ring is enabled.
        ring = DeltaNetSnapshotRing(max_size=3)
        state = PromptCacheState(snapshot_ring=ring)

        state.update(token_ids=[1, 2, 3], kv_cache=[KVCache()])
        assert len(ring) == 0


class TestPromptCacheStateFindPrefixLength:
    def test_returns_zero_when_no_prior_state(self):
        state = PromptCacheState()
        assert state.find_prefix_length([1, 2, 3]) == 0

    def test_full_match_returns_full_length(self):
        state = PromptCacheState()
        state.token_ids = [1, 2, 3]
        assert state.find_prefix_length([1, 2, 3]) == 3

    def test_partial_match(self):
        state = PromptCacheState()
        state.token_ids = [1, 2, 3, 4, 5]
        assert state.find_prefix_length([1, 2, 3, 99, 99]) == 3

    def test_handles_new_shorter_than_cached(self):
        state = PromptCacheState()
        state.token_ids = [1, 2, 3, 4, 5]
        assert state.find_prefix_length([1, 2]) == 2

    def test_complete_divergence(self):
        state = PromptCacheState()
        state.token_ids = [1, 2, 3]
        assert state.find_prefix_length([99, 99, 99]) == 0


class TestPromptCacheStateRewindEnabled:
    def test_default_rewind_enabled(self):
        assert PromptCacheState().rewind_enabled is True

    def test_explicit_disable(self):
        state = PromptCacheState(rewind_enabled=False)
        assert state.rewind_enabled is False

    def test_attaches_default_ring_when_omitted(self):
        # PromptCacheState() with no args still gets a usable ring —
        # chat.py / chat_ui.py / tests rely on this.
        state = PromptCacheState()
        assert state.snapshot_ring is not None
        assert state.snapshot_ring.enabled


class TestTrimCache:
    """Module-level _trim_cache. Used in two places: at start-of-generation
    (trim cached state to common-prefix length before resuming forward),
    and at end-of-generation (trim the just-generated assistant tokens
    back off so the persisted cache state always ends at end-of-user-turn,
    sidestepping the asymmetric-rendering problem on thinking models)."""

    def _kv(self, offset: int, seq_len: int) -> KVCache:
        # Build a standard KVCache with offset and a [B, H, L, D] tensor
        # of length seq_len. _trim_cache should both slice the tensor
        # and update the offset.
        c = KVCache()
        c.offset = offset
        c.keys = mx.zeros((1, 2, seq_len, 4))
        c.values = mx.zeros((1, 2, seq_len, 4))
        return c

    def test_trim_standard_kv_cache(self):
        c = self._kv(offset=20, seq_len=20)
        _trim_cache(c, target_len=12)
        assert int(c.offset) == 12
        assert c.keys.shape[2] == 12
        assert c.values.shape[2] == 12

    def test_no_op_when_already_at_or_below_target(self):
        c = self._kv(offset=8, seq_len=8)
        _trim_cache(c, target_len=12)  # target > current — no shrink
        assert int(c.offset) == 8
        assert c.keys.shape[2] == 8

    def test_recurses_into_lists(self):
        a = self._kv(offset=20, seq_len=20)
        b = self._kv(offset=20, seq_len=20)
        _trim_cache([a, b], target_len=5)
        assert int(a.offset) == 5 and int(b.offset) == 5

    def test_recurses_into_caches_attribute(self):
        # Hybrid wrappers expose a `.caches` list of layer caches.
        inner = self._kv(offset=30, seq_len=30)
        wrapper = _Container(inner)
        _trim_cache(wrapper, target_len=10)
        assert int(inner.offset) == 10
        assert inner.keys.shape[2] == 10

    def test_rotating_cache_uses_native_trim_no_physical_slice(self):
        # _trim_cache must NOT physically slice keys/values on rotating
        # caches — they own their internal ring buffer state. Just call
        # .trim() and stop.
        r = _rotating(offset=20, max_size=64)
        # Plant marker tensors; _trim_cache must leave them untouched.
        r.keys = mx.full((1, 2, 20, 4), 7.0)
        r.values = mx.full((1, 2, 20, 4), 9.0)
        _trim_cache(r, target_len=5)
        # Offset moved, but tensor shape unchanged (rotating manages it).
        assert int(r.offset) == 5
        assert r.keys.shape[2] == 20
        assert r.values.shape[2] == 20

    def test_trim_to_zero(self):
        c = self._kv(offset=10, seq_len=10)
        _trim_cache(c, target_len=0)
        assert int(c.offset) == 0
        assert c.keys.shape[2] == 0

    def test_handles_object_without_offset(self):
        # Should be a no-op (some hybrid layers carry no per-layer offset).
        class NoOffset:
            pass

        _trim_cache(NoOffset(), target_len=5)  # must not raise


class TestPostGenerationCacheBoundary:
    """Reproduces the Gemma 4 26B 8-bit failure mode from the 2026-05-02
    integration log:

        Turn 1: prefill 4613 tokens → generate 121 tokens.
        Pre-fix cache state: 4734 tokens (prefill + asst incl. thinking).
        Turn 2: prompt 4645 tokens (= 4613 prior context + 32 for the
        client's stripped asst echo + new user). Token-divergence at the
        start of the assistant boundary fired backwards trim → SWA
        rewind guard → full re-prefill (Skipped Context: 0, Prompt
        Delta: 4645) on every turn.

    Post-fix invariant: stream_generate's post-generation block calls
    `_trim_cache(c, prefill_len)` on every cache layer before
    `prompt_cache_state.update(prefill_ids_only, trimmed_cache)`, so the
    persisted state always ends at end-of-user-turn. The next turn's
    `find_prefix_length` returns prefill_len cleanly — no backwards
    trim, no rewind guard.

    These tests exercise the exact mechanic without spinning up a real
    model: simulate the cache state stream_generate would produce, run
    the post-generation trim, and check `find_prefix_length` against a
    realistic next-turn prompt.
    """

    # Numbers from the Gemma 4 log.
    PREFILL_LEN = 4613
    GEN_LEN = 121
    NEW_USER_DELTA = 32

    def _build_post_gen_cache(self, prefill_len, gen_len):
        """Cache state as it would be after generation completes:
        offset = prefill + gen, K/V tensors sized accordingly."""
        c = KVCache()
        c.offset = prefill_len + gen_len
        c.keys = mx.zeros((1, 2, prefill_len + gen_len, 4))
        c.values = mx.zeros((1, 2, prefill_len + gen_len, 4))
        return c

    def test_post_generation_trim_anchors_cache_at_end_of_user(self):
        # Mimics stream_generate's post-generation block:
        #   _trim_cache(c, len(full_input_ids_list))
        #   prompt_cache_state.update(full_input_ids_list, tracked_cache)
        prefill_ids = list(range(self.PREFILL_LEN))
        cache = self._build_post_gen_cache(self.PREFILL_LEN, self.GEN_LEN)
        state = PromptCacheState()

        _trim_cache([cache], target_len=len(prefill_ids))
        state.update(prefill_ids, [cache])

        # Persisted token_ids = prefill only, NOT prefill + generated.
        assert state.token_ids == prefill_ids
        assert len(state.token_ids) == self.PREFILL_LEN
        # Cache layer offset and tensor length both reflect the trim.
        assert int(cache.offset) == self.PREFILL_LEN
        assert cache.keys.shape[2] == self.PREFILL_LEN
        assert cache.values.shape[2] == self.PREFILL_LEN

    def test_next_turn_prefix_match_returns_prefill_length(self):
        # Set up post-trim state from turn 1.
        prefill_ids = list(range(self.PREFILL_LEN))
        cache = self._build_post_gen_cache(self.PREFILL_LEN, self.GEN_LEN)
        state = PromptCacheState()
        _trim_cache([cache], target_len=len(prefill_ids))
        state.update(prefill_ids, [cache])

        # Turn 2 prompt: prefill_ids unchanged, then any-shape echoed
        # assistant + new user. The asst tokens here represent the
        # thinking-stripped form the client sends back — could be ANY
        # tokens different from what we generated; the test doesn't
        # care because we never persisted them.
        asst_stripped = list(range(60_000, 60_000 + 10))
        new_user = list(range(70_000, 70_000 + 22))  # 10 + 22 = 32 = log delta
        turn2_prompt = prefill_ids + asst_stripped + new_user
        assert len(turn2_prompt) == self.PREFILL_LEN + self.NEW_USER_DELTA

        # Forward prefix-match: cleanly returns the cached prefill length.
        # No matter what the client did with the asst (thinking,
        # stripping, edits), divergence shows up AFTER the cached
        # boundary, so the cache reuse is safe.
        assert state.find_prefix_length(turn2_prompt) == self.PREFILL_LEN

    def test_without_trim_would_have_diverged_at_asst_boundary(self):
        # Negative control: if we DON'T trim (the pre-fix behavior),
        # token_ids gets the full prefill + generated sequence. The
        # next turn's prompt diverges from it at the asst boundary,
        # which would force a backwards trim — exactly the failure
        # mode the Gemma 4 log demonstrated.
        prefill_ids = list(range(self.PREFILL_LEN))
        # What we generated (e.g., includes thinking content):
        generated_ids = list(range(80_000, 80_000 + self.GEN_LEN))
        full_with_gen = prefill_ids + generated_ids
        cache = self._build_post_gen_cache(self.PREFILL_LEN, self.GEN_LEN)
        state = PromptCacheState()
        # NO _trim_cache call here — pre-fix behavior.
        state.update(full_with_gen, [cache])

        # Turn 2 prompt with thinking-stripped asst.
        asst_stripped = list(range(60_000, 60_000 + 10))
        new_user = list(range(70_000, 70_000 + 22))
        turn2_prompt = prefill_ids + asst_stripped + new_user

        # find_prefix_length walks token-by-token; it would diverge at
        # position PREFILL_LEN (start of asst) — that's the divergence
        # the post-fix avoids by never storing the asst in the first place.
        diverged_at = state.find_prefix_length(turn2_prompt)
        assert diverged_at == self.PREFILL_LEN
        # And the cache's stored token_ids extends PAST that divergence
        # point — which is what triggers the backwards trim that SWA
        # caches can't safely do without a snapshot, leading to the
        # "Hybrid-Cache Rewind Guard: no snapshot available" warning
        # and full re-prefill in the integration log.
        assert len(state.token_ids) == self.PREFILL_LEN + self.GEN_LEN
        assert len(state.token_ids) > diverged_at

    @pytest.mark.parametrize(
        "case_name,asst_echo_len,new_user_len",
        [
            # Client (e.g. OpenAI SDK / some thinking-preserving UI)
            # echoes the full assistant including thinking content. The
            # asst portion is the same length as what we generated.
            ("thinking_preserved", 121, 32),
            # Client (e.g. OpenWebUI) strips <think>...</think> before
            # echoing — asst is much shorter than what we generated.
            # This is the failure case in the 2026-05-02 Gemma 4 log.
            ("thinking_stripped", 10, 22),
            # Regenerate flow: client drops the asst entirely and
            # re-asks for a fresh response. Asst portion is empty.
            ("regenerate_no_asst", 0, 32),
            # Edit flow: client edits the asst before echoing. Different
            # tokens, possibly different length. We don't care what
            # shape — the cache wasn't anchored on it.
            ("edited_asst_longer", 200, 32),
            ("edited_asst_shorter", 5, 32),
        ],
    )
    def test_prefix_match_unaffected_by_asst_echo_shape(
        self, case_name, asst_echo_len, new_user_len
    ):
        """The headline invariant: cache reuse works regardless of how
        the client treats the assistant message. Whether the client
        echoes it back verbatim (thinking included), strips thinking,
        edits the message, or omits it entirely on regenerate — the
        cache is anchored at end-of-user-turn, so prefix-match always
        returns PREFILL_LEN cleanly. The post-cached portion of the
        prompt (asst + new user, in whatever shape) gets prefilled
        forward; no backwards trim is ever requested."""
        prefill_ids = list(range(self.PREFILL_LEN))
        cache = self._build_post_gen_cache(self.PREFILL_LEN, self.GEN_LEN)
        state = PromptCacheState()
        _trim_cache([cache], target_len=len(prefill_ids))
        state.update(prefill_ids, [cache])

        # Build the turn-2 prompt under this case's asst-echo shape.
        # Use a high token-id range so any accidental overlap with the
        # prefill_ids range can't fake a longer prefix match.
        asst_echo = list(range(60_000, 60_000 + asst_echo_len))
        new_user = list(range(70_000, 70_000 + new_user_len))
        turn2_prompt = prefill_ids + asst_echo + new_user

        # Invariant under every case: prefix match returns PREFILL_LEN.
        # Cache extends through end-of-user; whatever follows is brand
        # new from the cache's POV.
        assert state.find_prefix_length(turn2_prompt) == self.PREFILL_LEN, (
            f"case={case_name}: expected prefix_length={self.PREFILL_LEN} "
            f"(end-of-user) but got something else"
        )
        # Cache state itself wasn't mutated by the prefix check.
        assert int(cache.offset) == self.PREFILL_LEN
        assert len(state.token_ids) == self.PREFILL_LEN

    def test_rotating_post_gen_trim_safe_rejects_wrapped_ring_at_user_log_numbers(
        self,
    ):
        """Direct regression for the 2026-05-02 Gemma 4 26B 8-bit
        repetition-loop crash. The numbers come from the user's log:

            offset=4898 (4613 prefill + 285 generated)
            max_size=4096  (Gemma's typical SWA window)
            trim target=4613 (back to end-of-user-turn)

        The OLD lenient `_rotating_rewind_safe` returns True for this
        case (target_len=4613 is above the lower bound of the wrapped
        window, offset - max_size = 802). Trimming the rotating cache
        in this state corrupts attention: ring slots near target hold
        post-target content (from the just-generated tokens), and the
        K/V from positions [517, 802) — needed for the post-trim SWA
        window [517, 4613) — was overwritten during generation.

        The new `_rotating_post_gen_trim_safe` correctly returns False:
        any wrapped ring fails the strict check. The post-gen path
        then falls back to persisting the full end-of-asst state and
        lets the next request's existing rewind guard force full
        re-prefill — same behavior as before the cache-trim feature,
        no improvement on this turn but no corruption either.
        """
        r = _rotating(offset=4898, max_size=4096)

        # The lenient check (still used at start-of-generation rewind
        # path) treats this as safe — DOCUMENTING the gap that motivated
        # the strict variant.
        assert _rotating_rewind_safe([r], target_len=4613) is True

        # The strict check correctly identifies this as unsafe.
        assert _rotating_post_gen_trim_safe([r], target_len=4613) is False

    def test_rotating_post_gen_trim_safe_unwrapped_ring_is_safe(self):
        # Ring that hasn't wrapped yet → trim is genuinely safe (slots
        # near target hold the right content because nothing has been
        # overwritten). The post-gen path is allowed to engage.
        r = _rotating(offset=2000, max_size=4096)
        assert _rotating_post_gen_trim_safe([r], target_len=1500) is True

    def test_rotating_post_gen_trim_safe_recurses_into_caches(self):
        # Hybrid wrappers (Gemma 4 alternates global-attn + local-SWA
        # layers): the helper must visit every layer and return False
        # if ANY rotating layer has wrapped, since the trim is applied
        # uniformly across layers.
        kv = _rotating(offset=1000, max_size=4096)  # safe alone
        swa = _rotating(offset=4898, max_size=4096)  # wrapped
        wrapper = _Container(kv, swa)
        assert _rotating_post_gen_trim_safe([wrapper], target_len=900) is False

    def test_post_gen_branch_persists_full_state_when_swa_wrapped(self):
        """End-to-end: simulate stream_generate's post-gen branch
        decision against the user's log numbers and verify what
        actually gets persisted.

        Reproduces the conditions that produced the Gemma 4 26B
        repetition loop: prefill 4613 tokens, generate 285 tokens,
        SWA layer with max_size=4096 (wrapped during generation),
        plus a standard quantized layer that wouldn't have wrapped.
        The post-gen branch must skip the trim and persist the full
        end-of-asst state (offset 4898), not the trimmed end-of-user
        state (offset 4613). Otherwise the SWA layer's ring contents
        get misaligned and turn 2's attention reads garbage K/V.

        With the strict helper installed, the assertion path runs:
            _rotating_post_gen_trim_safe(...) → False
            ⇒ persist full token list, no trim
        """
        prefill_ids = list(range(4613))
        generated_ids = list(range(80_000, 80_000 + 285))
        swa_layer = _rotating(offset=4898, max_size=4096)
        # Plant a marker so we can verify the layer wasn't trimmed.
        swa_layer.keys = mx.full((1, 2, 4096, 4), 7.0)

        state = PromptCacheState()

        # Mimic the stream_generate post-gen branch verbatim.
        if _rotating_post_gen_trim_safe([swa_layer], target_len=4613):
            for c in [swa_layer]:
                _trim_cache(c, target_len=4613)
            state.update(prefill_ids, [swa_layer])
        else:
            full_ids = prefill_ids + generated_ids
            state.update(full_ids, [swa_layer])

        # Wrapped ring → branch took the "persist full" path.
        assert state.token_ids == prefill_ids + generated_ids
        assert len(state.token_ids) == 4898
        # SWA layer's offset is unchanged (no trim applied).
        assert int(swa_layer.offset) == 4898

    def test_post_gen_branch_trims_when_swa_not_wrapped(self):
        """Counterpart: ring hasn't wrapped → trim path engages,
        end-of-user state is persisted, cache reuse is enabled for
        the next request."""
        prefill_ids = list(range(2000))
        generated_ids = list(range(80_000, 80_000 + 285))
        swa_layer = _rotating(offset=2285, max_size=4096)  # never wrapped

        state = PromptCacheState()

        if _rotating_post_gen_trim_safe([swa_layer], target_len=2000):
            for c in [swa_layer]:
                _trim_cache(c, target_len=2000)
            state.update(prefill_ids, [swa_layer])
        else:
            full_ids = prefill_ids + generated_ids
            state.update(full_ids, [swa_layer])

        # Unwrapped → trim ran, only prefill ids persisted, offset moved.
        assert state.token_ids == prefill_ids
        assert len(state.token_ids) == 2000
        assert int(swa_layer.offset) == 2000

    def test_rotating_post_gen_trim_safe_ignores_non_rotating(self):
        # Standard KVCache and ArraysCache (DeltaNet) shouldn't be
        # consulted by this helper — they're not rotating buffers.
        # The post-gen trim path's overall correctness for those is
        # the responsibility of `_has_non_trimmable` and the standard
        # trim/no-physical-slice logic.
        kv = KVCache()
        kv.offset = 100
        ar = _arrays_cache(num_arrays=2)
        assert _rotating_post_gen_trim_safe([kv, ar], target_len=50) is True

    def test_trim_does_not_corrupt_quantized_kv_cache_state(self):
        """Regression for the 2026-05-02 quantized_matmul TypeError in
        the production Gemma 4 + kv_bits=4 setup.

        QuantizedKVCache.keys is a 3-element list:
            [quantized_uint32, scales, biases]
        The 3 elements are *components of one layer's state*, not three
        chunks along the sequence axis. The earlier `isinstance(list)`
        branch in `_trim_cache` was written for ChunkedKVCache (where
        the list IS a sequence-chunk list) and would walk these 3
        elements as if they were chunks, slice-and-break after the
        first, and silently drop scales+biases. The next call to
        `mx.quantized_matmul(queries, *q_keys, ...)` then explodes
        with `incompatible function arguments` — only 2 positional
        instead of 3.

        The cache reuse path was rarely exercised in production
        because the SWA rewind guard tended to force full re-prefill;
        the post-generation cache trim made reuse the common case and
        exposed the latent bug.
        """

        # Stand-in QuantizedKVCache with the right Python type-name
        # and the same data shape contract.
        class FakeQuantizedKVCache:
            def __init__(self, num_steps: int):
                self.offset = num_steps
                self.group_size = 64
                self.bits = 4
                # Mirror mlx_lm: keys/values are 3-elem lists shaped
                # like [(B, H, num_steps, D//el), (B, H, num_steps, D//gs),
                # (B, H, num_steps, D//gs)] for the affine mode.
                self.keys = [
                    mx.zeros((1, 2, num_steps, 8), dtype=mx.uint32),
                    mx.zeros((1, 2, num_steps, 1)),
                    mx.zeros((1, 2, num_steps, 1)),
                ]
                self.values = [
                    mx.zeros((1, 2, num_steps, 8), dtype=mx.uint32),
                    mx.zeros((1, 2, num_steps, 1)),
                    mx.zeros((1, 2, num_steps, 1)),
                ]

            def trim(self, n):
                n = min(self.offset, n)
                self.offset -= n

        # Match the type-name guard's expectation.
        FakeQuantizedKVCache.__name__ = "QuantizedKVCache"

        c = FakeQuantizedKVCache(num_steps=4734)
        keys_id = id(c.keys)
        values_id = id(c.values)
        keys_lengths_before = [k.shape[2] for k in c.keys]
        values_lengths_before = [v.shape[2] for v in c.values]

        _trim_cache(c, target_len=4613)

        # Offset moved correctly via .trim().
        assert int(c.offset) == 4613
        # The 3-element [quantized, scales, biases] structure is intact:
        # length 3 (NOT 1, which is what the bug produced), same array
        # objects (no slicing, no replacement), same allocated shapes.
        assert isinstance(c.keys, list)
        assert isinstance(c.values, list)
        assert len(c.keys) == 3
        assert len(c.values) == 3
        assert id(c.keys) == keys_id
        assert id(c.values) == values_id
        assert [k.shape[2] for k in c.keys] == keys_lengths_before
        assert [v.shape[2] for v in c.values] == values_lengths_before

    def test_trim_handles_rotating_swa_cache_like_gemma(self):
        # Gemma 4's failure was on its SWA RotatingKVCache. The post-
        # generation trim must work on it too — that's the whole point.
        # _trim_cache delegates to .trim() and does NOT physically slice
        # the keys/values (the rotating cache manages its own ring).
        prefill_len = self.PREFILL_LEN
        r = _rotating(offset=prefill_len + self.GEN_LEN, max_size=8192)
        # Plant marker tensors; _trim_cache must not touch them.
        r.keys = mx.full((1, 2, prefill_len + self.GEN_LEN, 4), 1.0)
        r.values = mx.full((1, 2, prefill_len + self.GEN_LEN, 4), 1.0)

        _trim_cache([r], target_len=prefill_len)

        assert int(r.offset) == prefill_len
        # Tensors unchanged (rotating manages its ring internally).
        assert r.keys.shape[2] == prefill_len + self.GEN_LEN
        assert r.values.shape[2] == prefill_len + self.GEN_LEN


class TestSnapshotPathIntegration:
    """Integration tests that drive `stream_generate`'s post-gen
    branching logic verbatim, asserting both halves of the
    asymmetric-vs-symmetric split (memory.md #30) produce the right
    persisted state. We don't actually run the model — we simulate
    what generation would do to the cache, then exercise the post-gen
    block's decision tree.
    """

    def _build_mixed_cache(self, prefill_len: int, gen_len: int):
        """Mixed cache simulating Gemma 4: alternating standard KV +
        rotating SWA layers. Standard KV at offset = prefill+gen,
        rotating at offset = prefill+gen with ring wrapped past
        max_size."""
        kv = KVCache()
        kv.offset = prefill_len + gen_len
        kv.keys = mx.zeros((1, 2, prefill_len + gen_len, 4))
        kv.values = mx.zeros((1, 2, prefill_len + gen_len, 4))

        swa = _rotating(offset=prefill_len + gen_len, max_size=4096)
        # Plant a marker on the rotating layer so we can verify
        # restore vs. retention.
        swa.keys = mx.full((1, 2, 4096, 32), 99.0)
        swa.values = mx.full((1, 2, 4096, 32), 99.0)
        return [kv, swa]

    def test_asymmetric_path_anchors_cache_at_end_of_user(self):
        # Snapshot was captured at end-of-prefill (before any gen
        # writes). Post-gen restore + trim → cache state reflects
        # end-of-user-turn for every layer type.
        prefill_len = 4613
        gen_len = 271

        # Build cache, snapshot the rotating layer at end-of-prefill
        # state (= what would have been captured at n==0 in the loop).
        prefill_kv = KVCache()
        prefill_kv.offset = prefill_len
        prefill_kv.keys = mx.zeros((1, 2, prefill_len, 4))
        prefill_kv.values = mx.zeros((1, 2, prefill_len, 4))
        prefill_swa = _rotating(offset=prefill_len, max_size=4096)
        prefill_swa.keys = mx.full((1, 2, 4096, 32), 1.0)
        prefill_swa.values = mx.full((1, 2, 4096, 32), 1.0)
        snapshots = _capture_rotating_layers_for_snapshot(
            [prefill_kv, prefill_swa], capture_rotating
        )
        assert len(snapshots) == 1
        assert snapshots[0].layer_index == 1

        # Simulate what generation does to the cache:
        #   - standard KV layer's offset advances + tensors grow.
        #   - rotating layer's offset advances + ring contents
        #     overwritten.
        post_gen_kv = KVCache()
        post_gen_kv.offset = prefill_len + gen_len
        post_gen_kv.keys = mx.zeros((1, 2, prefill_len + gen_len, 4))
        post_gen_kv.values = mx.zeros((1, 2, prefill_len + gen_len, 4))
        post_gen_swa = prefill_swa  # same object, mutated below.
        post_gen_swa.offset = prefill_len + gen_len
        post_gen_swa._idx = (prefill_len + gen_len) % 4096
        post_gen_swa.keys = mx.full((1, 2, 4096, 32), 99.0)
        post_gen_swa.values = mx.full((1, 2, 4096, 32), 99.0)

        tracked = [post_gen_kv, post_gen_swa]
        prefill_ids = list(range(prefill_len))

        # Run the asymmetric path's post-gen logic directly.
        _restore_rotating_layers_from_snapshots(tracked, snapshots)
        for c in tracked:
            if not _is_rotating_kv_layer(c):
                _trim_cache(c, prefill_len)

        # KV layer trimmed back to prefill_len.
        assert int(post_gen_kv.offset) == prefill_len
        assert post_gen_kv.keys.shape[2] == prefill_len
        # SWA layer restored from snapshot — offset and tensor markers
        # back to end-of-prefill values.
        assert int(post_gen_swa.offset) == prefill_len
        assert post_gen_swa.keys[0, 0, 0, 0].item() == 1.0  # restored 1.0, not 99.0

        state = PromptCacheState()
        state.update(prefill_ids, tracked)
        # Persisted state has prefill_len tokens — no gen tokens.
        assert state.token_ids == prefill_ids

    def test_symmetric_path_persists_full_state_including_gen(self):
        # Symmetric: skip snapshot, skip trim, persist full ids
        # (prefill + generated). Next request forward-extends through
        # the assistant turn — same as the pre-fix behavior on
        # symmetric models, no overhead.
        prefill_len = 2000
        gen_len = 200
        tracked = self._build_mixed_cache(prefill_len, gen_len)
        prefill_ids = list(range(prefill_len))
        gen_ids = list(range(80_000, 80_000 + gen_len))

        # Symmetric path doesn't snapshot/restore/trim. Just persist.
        state = PromptCacheState()
        state.update(prefill_ids + gen_ids, tracked)

        assert state.token_ids == prefill_ids + gen_ids
        # Cache offsets are still at end-of-asst.
        assert int(tracked[0].offset) == prefill_len + gen_len
        assert int(tracked[1].offset) == prefill_len + gen_len

    def test_asymmetric_path_with_no_rotating_layers(self):
        # Asymmetric path on a non-SWA model: trim the standard layers,
        # nothing to snapshot. Persist end-of-user.
        prefill_len = 100
        gen_len = 50
        kv = KVCache()
        kv.offset = prefill_len + gen_len
        kv.keys = mx.zeros((1, 2, prefill_len + gen_len, 4))
        kv.values = mx.zeros((1, 2, prefill_len + gen_len, 4))
        tracked = [kv]
        prefill_ids = list(range(prefill_len))

        snapshots = _capture_rotating_layers_for_snapshot(
            tracked, capture_rotating
        )
        assert snapshots == []  # no rotating layers, nothing snapshotted

        # Asymmetric branch with empty snapshot list still trims.
        _restore_rotating_layers_from_snapshots(tracked, snapshots)  # no-op
        for c in tracked:
            if not _is_rotating_kv_layer(c):
                _trim_cache(c, prefill_len)

        assert int(kv.offset) == prefill_len
        assert kv.keys.shape[2] == prefill_len


class _CharTokenizer:
    """Tokenizer stub that encodes a string by char-codepoint per token.
    `len(encode(s))` == `len(s)`. Used to make token offsets equal
    character offsets for deterministic tests of the anchor-offset
    helper.
    """

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class TestComputeAnchorBeforeLatestUserOffset:
    """`_compute_anchor_before_latest_user_offset` finds the byte
    position of the LAST user-turn-open marker in the rendered prompt
    and returns its token offset. Pins:
      - returns the rightmost marker's offset across multiple turns
      - returns None when no marker is recognized
      - works across template families (Gemma 4, Qwen, Llama)
      - the returned offset == prefix length (NOT including the marker)
    """

    def test_returns_offset_at_latest_gemma4_user_turn(self):
        prompt = (
            "[system]\n"
            "<|turn>user\nq1<turn|>\n"
            "<|turn>model\na1<turn|>\n"
            "<|turn>user\nq2<turn|>\n"
            "<|turn>model\n"
        )
        offset = _compute_anchor_before_latest_user_offset(
            prompt, _CharTokenizer()
        )
        # Char tokenizer => offset == char position.
        assert offset == prompt.rfind("<|turn>user\n")
        # Cache anchor would persist tokens [0:offset). Verify the
        # cached portion ENDS just before the second user turn opens.
        assert prompt[:offset].endswith("<turn|>\n")
        assert prompt[offset:].startswith("<|turn>user\nq2")

    def test_returns_offset_for_qwen_chatml_template(self):
        prompt = (
            "<|im_start|>system\n[sys]<|im_end|>\n"
            "<|im_start|>user\nfirst<|im_end|>\n"
            "<|im_start|>assistant\nresp<|im_end|>\n"
            "<|im_start|>user\nsecond<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        offset = _compute_anchor_before_latest_user_offset(
            prompt, _CharTokenizer()
        )
        assert offset == prompt.rfind("<|im_start|>user\n")
        assert prompt[offset:].startswith("<|im_start|>user\nsecond")

    def test_returns_offset_for_llama3_template(self):
        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\nsys<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\nq<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        offset = _compute_anchor_before_latest_user_offset(
            prompt, _CharTokenizer()
        )
        assert offset == prompt.rfind("<|start_header_id|>user<|end_header_id|>\n")

    def test_returns_none_when_no_marker_present(self):
        # Plain text — none of the registered template markers appear.
        # Caller falls back to end-of-user anchoring.
        prompt = "completely plain prompt with no chat template at all"
        assert _compute_anchor_before_latest_user_offset(
            prompt, _CharTokenizer()
        ) is None

    def test_picks_rightmost_marker_across_families(self):
        # Edge: prompt has markers from multiple families. The
        # rightmost wins regardless of which family it belongs to.
        prompt = (
            "<|im_start|>user\nold<|im_end|>\n"
            "<|im_start|>assistant\nresp<|im_end|>\n"
            "<|turn>user\nlatest<turn|>\n"  # newer turn in different family
            "<|turn>model\n"
        )
        offset = _compute_anchor_before_latest_user_offset(
            prompt, _CharTokenizer()
        )
        assert offset == prompt.rfind("<|turn>user\n")
        assert prompt[offset:].startswith("<|turn>user\nlatest")

    def test_offset_is_prefix_length_not_marker_position_when_tokenizer_merges(self):
        # When the tokenizer merges multi-char sequences into single
        # tokens (real BPE/SentencePiece behavior), the returned offset
        # is the TOKEN count of the prefix, not the character position.
        # Stub here: encode always returns 1 token per group of 3 chars
        # (rounded up). Verifies the helper uses tokenizer.encode().
        class _ChunkedTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [0] * ((len(text) + 2) // 3)

        prompt = (
            "abcdef"  # 6 chars
            "<|turn>user\n"  # 12 chars
            "g"
        )
        offset = _compute_anchor_before_latest_user_offset(
            prompt, _ChunkedTokenizer()
        )
        # rfind hit at char 6. Tokenizer chunks 6 chars → 2 tokens.
        # So offset should be 2.
        assert offset == 2

    def test_falls_back_to_plain_encode_when_tokenizer_rejects_kwarg(self):
        # Some SentencePiece-only tokenizers don't accept
        # add_special_tokens. Helper must catch the TypeError and
        # retry without the kwarg.
        class _NoKwargTokenizer:
            def encode(self, text):
                return list(range(len(text)))

        prompt = "x<|turn>user\ny"
        offset = _compute_anchor_before_latest_user_offset(
            prompt, _NoKwargTokenizer()
        )
        assert offset == prompt.rfind("<|turn>user\n")


class TestOpenWebUIRagWrappingRegression:
    """End-to-end regression test for the OpenWebUI tool-flow bug:
    the same user message renders DIFFERENTLY across turns once a
    search-style tool returns citation content. Pre-fix: cache
    anchored at end-of-user (== full prompt length), so turn N+1's
    re-rendered (wrapped) user message diverges from the cached
    plain version → backward trim → wrapped-SWA corruption guard
    fires → full re-prefill (~13s wasted on a 6k-token prompt).

    Post-fix: cache anchored BEFORE the latest user message via the
    mid-prefill snapshot at ``_compute_anchor_before_latest_user_offset``.
    Turn N+1's wrapped user message is just NEW tokens to forward-
    extend through; no backward trim needed.

    The test simulates two consecutive turns through the cache layer:
      Turn 2 prefill: system+tools + user1 + asst1_echo + user2_PLAIN
                     + asst_marker (4646 tokens, char-tokenizer).
      Turn 3 prefill: system+tools + user1 + asst1_echo + user2_WRAPPED
                     + asst2 + tool_resp + asst_marker (~6500 tokens).

    Asserts that turn 3's prompt prefix-matches the entire turn-2 cache
    state — NO divergence within the cached range — so no rewind guard
    fires.
    """

    @pytest.fixture
    def gemma_prompts(self):
        # Faithful reproduction of the Gemma 4 OWUI shape from the
        # production logs (memory.md #29e). Char-tokenizer means
        # token-count == char-count, so positions are easy to read.
        system = "[system block + tool declarations]"  # static prefix
        user1 = "<|turn>user\nwho are you?<turn|>\n"
        asst1_echo = "<|turn>model\nI am a large language model.<turn|>\n"
        user2_plain_marker_and_body = (
            "<|turn>user\n"
            "what's the weather going to be tomorrow in Seattle?<turn|>\n"
        )
        user2_wrapped_marker_and_body = (
            "<|turn>user\n"
            "### Task:\n"
            "Respond to the user query using the provided context, "
            "incorporating inline citations in the format [id]...\n"
            "<context>\n<source id=\"1\"></source>\n</context>\n"
            "what's the weather going to be tomorrow in Seattle?<turn|>\n"
        )
        asst2_and_tools = (
            "<|turn>model\n[tool calls + tool responses content]<turn|>\n"
        )
        asst_marker = "<|turn>model\n"
        turn2_prompt = (
            system + user1 + asst1_echo
            + user2_plain_marker_and_body + asst_marker
        )
        turn3_prompt = (
            system + user1 + asst1_echo
            + user2_wrapped_marker_and_body
            + asst2_and_tools + asst_marker
        )
        return turn2_prompt, turn3_prompt

    def test_turn2_anchor_lies_before_user2(self, gemma_prompts):
        # The fix's headline invariant: turn 2's persisted cache anchor
        # is computed at the position of the LAST <|turn>user marker.
        # That position is BEFORE user2's content, so the cache holds
        # tokens [0:anchor) — system + tools + user1 + asst1.
        turn2_prompt, _ = gemma_prompts
        offset = _compute_anchor_before_latest_user_offset(
            turn2_prompt, _CharTokenizer()
        )
        # Anchor lands at the start of the user2 marker.
        assert offset == turn2_prompt.rfind("<|turn>user\n")
        # And the bytes BEFORE the anchor end at the end of asst1's
        # echoed content.
        assert turn2_prompt[:offset].endswith("<turn|>\n")
        # Crucially: the bytes AT and after the anchor START with the
        # marker (so the user2 body is NOT in the cache).
        assert turn2_prompt[offset:].startswith("<|turn>user\n")
        assert "what's the weather" not in turn2_prompt[:offset]

    def test_turn3_prefix_matches_turn2_anchor_with_no_divergence(self, gemma_prompts):
        # The headline regression: simulate turn 2 persisting its cache
        # at the new anchor offset, then check turn 3's full prompt
        # against that cached state. find_prefix_length must return
        # exactly the anchor length — meaning every token in the cache
        # is also present at the same position in turn 3's prompt
        # (no divergence within the cached range).
        turn2_prompt, turn3_prompt = gemma_prompts
        tokenizer = _CharTokenizer()

        anchor_offset = _compute_anchor_before_latest_user_offset(
            turn2_prompt, tokenizer
        )
        assert anchor_offset is not None

        turn2_ids = tokenizer.encode(turn2_prompt[:anchor_offset])
        turn3_ids = tokenizer.encode(turn3_prompt)

        # Simulate the persisted cache state from turn 2.
        cached_state = PromptCacheState()
        # Skip the cache list — find_prefix_length only reads token_ids.
        cached_state.token_ids = turn2_ids

        prefix_len = cached_state.find_prefix_length(turn3_ids)

        # FULL match — the cache's prefix is preserved verbatim in
        # turn 3's prompt. No backward trim, no rewind guard fires.
        assert prefix_len == len(turn2_ids), (
            f"divergence within cached range: prefix_len={prefix_len}, "
            f"cached={len(turn2_ids)}"
        )
        # And turn 3's prompt has MORE tokens beyond the cache (the
        # wrapped user2 + asst2 + tool_resp + asst_marker), which the
        # next prefill will forward-extend through.
        assert len(turn3_ids) > prefix_len

    def test_pre_fix_behavior_would_have_diverged(self, gemma_prompts):
        # Negative control: the OLD anchoring policy (end-of-user-turn
        # = full prompt length) would have cached turn 2's PLAIN user2
        # bytes. Turn 3's wrapped user2 diverges within that cached
        # range. This pins WHY the fix is needed.
        turn2_prompt, turn3_prompt = gemma_prompts
        tokenizer = _CharTokenizer()

        # Pre-fix anchor: full turn-2 prompt.
        turn2_ids_pre_fix = tokenizer.encode(turn2_prompt)
        turn3_ids = tokenizer.encode(turn3_prompt)

        cached_state = PromptCacheState()
        cached_state.token_ids = turn2_ids_pre_fix

        prefix_len = cached_state.find_prefix_length(turn3_ids)

        # Divergence at the FIRST position where the wrapped user2
        # differs from plain user2 — well before the end of the cache.
        assert prefix_len < len(turn2_ids_pre_fix)
        # And critically, the divergence is WITHIN the cached range
        # (cache_len == 4646-ish, divergence at the start of user2
        # body), which is the condition that triggered the rewind
        # guard fallback in production.
        cache_len = len(turn2_ids_pre_fix)
        assert prefix_len < cache_len
        # The divergence sits exactly at the start of user2's body
        # (right after `<|turn>user\n`).
        marker_pos = turn2_prompt.rfind("<|turn>user\n") + len(
            "<|turn>user\n"
        )
        assert prefix_len == marker_pos

    def test_anchor_is_stable_across_turns_when_user_message_unchanged(
        self, gemma_prompts
    ):
        # Turns 2, 3, 4, ... all answer the same user2 query (just
        # adding tool calls + tool responses to the conversation).
        # The anchor offset should be IDENTICAL across all turns
        # because the latest user message doesn't change. This means
        # the cache persisted on turn 2 covers ALL of turn 3's prefix
        # too — no re-prefill of the cached portion ever needed.
        turn2_prompt, turn3_prompt = gemma_prompts
        tokenizer = _CharTokenizer()

        offset_t2 = _compute_anchor_before_latest_user_offset(
            turn2_prompt, tokenizer
        )
        offset_t3 = _compute_anchor_before_latest_user_offset(
            turn3_prompt, tokenizer
        )
        # Same position both turns (the user2 marker hasn't moved).
        assert offset_t2 == offset_t3
        # And the cached prefix bytes are byte-identical too.
        assert turn2_prompt[:offset_t2] == turn3_prompt[:offset_t3]


class TestAdjustChunkForSnapshotLanding:
    """Pure tests for the chunk-shrinking helper. The chunked-prefill
    loop calls this BEFORE processing each chunk to ensure the chunk
    ends EXACTLY on ``snapshot_at_offset`` when crossing it.
    """

    def test_no_target_returns_original_n(self):
        # Nothing to align to → no shrink.
        assert _adjust_chunk_for_snapshot_landing(
            cumulative_offset=100,
            n_to_process=512,
            snapshot_at_offset=None,
            snapshot_done=False,
        ) == 512

    def test_already_done_returns_original_n(self):
        # Snapshot already captured → don't shrink even if a chunk
        # would still cross the (now stale) target.
        assert _adjust_chunk_for_snapshot_landing(
            cumulative_offset=100,
            n_to_process=512,
            snapshot_at_offset=200,
            snapshot_done=True,
        ) == 512

    def test_chunk_crosses_target_shrinks_to_land_exactly(self):
        # cumulative=4626, n=512 → would end at 5138. target=4700 lies
        # between → shrink to land at 4700.
        assert _adjust_chunk_for_snapshot_landing(
            cumulative_offset=4626,
            n_to_process=512,
            snapshot_at_offset=4700,
            snapshot_done=False,
        ) == 4700 - 4626  # 74

    def test_chunk_lands_exactly_on_target_no_shrink(self):
        # cumulative + n == target → no shrink needed; the post-process
        # capture decision will fire as "capture_and_finalize".
        assert _adjust_chunk_for_snapshot_landing(
            cumulative_offset=4626,
            n_to_process=512,
            snapshot_at_offset=4626 + 512,
            snapshot_done=False,
        ) == 512

    def test_chunk_strictly_before_target_no_shrink(self):
        # cumulative + n < target → chunk doesn't reach target on this
        # iteration, no shrink. This is a "fallback" candidate at the
        # boundary the chunk lands at.
        assert _adjust_chunk_for_snapshot_landing(
            cumulative_offset=4626,
            n_to_process=512,
            snapshot_at_offset=6000,
            snapshot_done=False,
        ) == 512

    def test_already_past_target_no_shrink(self):
        # Defensive: if we somehow overshot before this iteration
        # (snapshot_done==False but cumulative>=target), just return
        # n_to_process. The chunked-prefill loop's post-process check
        # will reach "skip" via _classify_snapshot_action.
        assert _adjust_chunk_for_snapshot_landing(
            cumulative_offset=5900,
            n_to_process=512,
            snapshot_at_offset=5800,
            snapshot_done=False,
        ) == 512

    def test_target_at_last_token_of_loop_reach(self):
        # Boundary case from the production turn-4 fault-window:
        # if snapshot_at_offset == cumulative + n_to_process - 1,
        # the chunk shrinks by one token. After processing, cumulative
        # equals the target exactly.
        n = _adjust_chunk_for_snapshot_landing(
            cumulative_offset=5650,
            n_to_process=221,
            snapshot_at_offset=5871 - 1,
            snapshot_done=False,
        )
        assert n == 5871 - 1 - 5650  # 220


class TestClassifySnapshotAction:
    """Pure tests for the post-process boundary classification helper.

    Three actions: ``skip``, ``capture_and_finalize``, ``capture_as_fallback``.
    The fallback path is the load-bearing addition for the production
    turn-4 issue: when snapshot_at_offset is unreachable (e.g.,
    sits past the chunked-prefill loop's final cumulative offset due
    to BPE context-sensitivity in the helper's prefix tokenization),
    the loop captures at the closest pre-target chunk boundary
    instead of falling through to end-of-user anchoring.
    """

    def test_no_target_returns_skip(self):
        assert _classify_snapshot_action(
            cumulative_offset=100,
            snapshot_at_offset=None,
            snapshot_done=False,
        ) == "skip"

    def test_already_done_returns_skip(self):
        assert _classify_snapshot_action(
            cumulative_offset=4626,
            snapshot_at_offset=4626,
            snapshot_done=True,
        ) == "skip"

    def test_exact_landing_returns_capture_and_finalize(self):
        # Adjust-chunk shrunk this iteration to land exactly on target.
        assert _classify_snapshot_action(
            cumulative_offset=4626,
            snapshot_at_offset=4626,
            snapshot_done=False,
        ) == "capture_and_finalize"

    def test_pre_target_chunk_returns_fallback_capture(self):
        # Earlier chunk boundary; later iterations may replace this
        # capture with a closer or exact one.
        assert _classify_snapshot_action(
            cumulative_offset=5138,
            snapshot_at_offset=5861,
            snapshot_done=False,
        ) == "capture_as_fallback"

    def test_overshot_target_returns_skip(self):
        # Defensive: if we've overshot the target (shouldn't happen
        # with proper shrink logic but the helper handles it), don't
        # capture — the rotating state now includes user-message
        # tokens we explicitly want OUT of the cache.
        assert _classify_snapshot_action(
            cumulative_offset=5900,
            snapshot_at_offset=5861,
            snapshot_done=False,
        ) == "skip"


class TestChunkedPrefillSnapshotCaptureLoop:
    """Integration test that drives the full chunked-prefill capture
    decision flow over multiple iterations using a mocked cache.
    Pins the production turn-4 regression: when ``snapshot_at_offset``
    falls slightly beyond the loop's reach, the fallback capture
    path provides a usable best-available snapshot instead of
    silently skipping the asymmetric anchor entirely.
    """

    def _drive_loop(
        self,
        initial_cache_offset: int,
        new_tokens_count: int,
        prefill_step_size: int,
        snapshot_at_offset,
    ):
        """Simulate the chunked-prefill loop's snapshot bookkeeping.
        Returns ``(captured_offset_or_None, snapshot_done)``.

        Each chunk:
          1. Compute n_to_process = min(prefill_step_size, remaining-1)
          2. Adjust via _adjust_chunk_for_snapshot_landing.
          3. Process (here: just bump cumulative_offset).
          4. Classify via _classify_snapshot_action; capture or skip.
        Mirrors the real generate_step prefill loop exactly minus the
        actual model forward pass.
        """
        cumulative_offset = initial_cache_offset
        remaining = new_tokens_count
        snapshot_done = False
        captured_offset = None  # latest fallback or exact capture

        while remaining > 1:
            n_to_process = min(prefill_step_size, remaining - 1)
            n_to_process = _adjust_chunk_for_snapshot_landing(
                cumulative_offset,
                n_to_process,
                snapshot_at_offset,
                snapshot_done,
            )
            # "Process": move cumulative + drain remaining.
            cumulative_offset += n_to_process
            remaining -= n_to_process

            action = _classify_snapshot_action(
                cumulative_offset,
                snapshot_at_offset,
                snapshot_done,
            )
            if action == "capture_and_finalize":
                captured_offset = cumulative_offset
                snapshot_done = True
            elif action == "capture_as_fallback":
                captured_offset = cumulative_offset

        return captured_offset, snapshot_done

    def test_target_within_loop_lands_exactly(self):
        # Healthy case: target inside the loop's reach. Shrink fires,
        # capture is finalized at the exact target.
        captured, done = self._drive_loop(
            initial_cache_offset=4626,
            new_tokens_count=1246,
            prefill_step_size=512,
            snapshot_at_offset=5861,
        )
        assert captured == 5861
        assert done is True

    def test_target_at_first_chunk_boundary(self):
        # Target lands EXACTLY on the natural first-chunk boundary
        # (no shrink needed). Capture finalizes there.
        captured, done = self._drive_loop(
            initial_cache_offset=0,
            new_tokens_count=1100,
            prefill_step_size=512,
            snapshot_at_offset=512,
        )
        assert captured == 512
        assert done is True

    def test_target_beyond_loop_reach_uses_fallback(self):
        # The PRODUCTION TURN-4 REGRESSION: snapshot_at_offset is
        # one or more tokens BEYOND what the chunked loop processes.
        # New tokens = 1246, prefill_step_size = 512. Loop processes
        # 512+512+221 = 1245 tokens (final 1 token reserved for
        # post-loop forward pass). Final cumulative_offset =
        # initial(4626) + 1245 = 5871. If snapshot_at_offset=5872
        # (or higher), the loop never reaches the target.
        # Pre-fix: silent skip → fallback to end-of-user anchoring →
        # OWUI RAG-wrap divergence on next turn.
        # Post-fix: capture at the closest pre-target boundary
        # (here: 5871, which is the final cumulative).
        captured, done = self._drive_loop(
            initial_cache_offset=4626,
            new_tokens_count=1246,
            prefill_step_size=512,
            snapshot_at_offset=5872,  # one past loop's reach
        )
        assert captured == 5871, (
            f"fallback capture should land at the closest pre-target "
            f"chunk boundary; got {captured}"
        )
        assert done is False  # fallback didn't finalize

    def test_target_far_beyond_loop_reach_still_captures(self):
        # Even when target is wildly beyond reach (e.g., off-by-100
        # due to some tokenization quirk), the loop still captures
        # at the closest pre-target boundary instead of skipping.
        captured, done = self._drive_loop(
            initial_cache_offset=4626,
            new_tokens_count=1246,
            prefill_step_size=512,
            snapshot_at_offset=10_000,  # way past
        )
        assert captured == 5871
        assert done is False

    def test_no_target_skips_all_captures(self):
        # No snapshot target at all: loop never captures.
        captured, done = self._drive_loop(
            initial_cache_offset=0,
            new_tokens_count=2000,
            prefill_step_size=512,
            snapshot_at_offset=None,
        )
        assert captured is None
        assert done is False

    def test_target_replaces_earlier_fallback_on_exact_landing(self):
        # Chain of captures: earlier iterations' fallback captures get
        # replaced by a later exact landing. Verifies the
        # "capture_as_fallback" → "capture_and_finalize" upgrade path.
        # Target sits between iter-2 boundary (1024) and iter-3
        # boundary; shrink fires on iter 3 to land exactly.
        captured, done = self._drive_loop(
            initial_cache_offset=0,
            new_tokens_count=2000,
            prefill_step_size=512,
            snapshot_at_offset=1100,
        )
        # Iter 1 lands at 512 → fallback. Iter 2 lands at 1024 →
        # fallback (replaces). Iter 3 shrinks to land at 1100 → exact.
        assert captured == 1100
        assert done is True

    def test_fallback_keeps_latest_when_multiple_pre_target_chunks(self):
        # When multiple chunk boundaries fall before the (unreachable)
        # target, the LATEST one wins — not an earlier one.
        captured, done = self._drive_loop(
            initial_cache_offset=0,
            new_tokens_count=2000,
            prefill_step_size=512,
            snapshot_at_offset=5000,  # far beyond
        )
        # Iter 1: 512 (fallback). Iter 2: 1024 (replace). Iter 3:
        # 1536 (replace). Iter 4: cumulative=2000-1=1999 (replace).
        # Final fallback should be 1999.
        assert captured == 1999
        assert done is False


class TestCaptureArraysLayersForSnapshot:
    """Pins capture for the Qwen 3.5/3.6 GatedDeltaNet asymmetric anchor
    path. The pre-fix asymmetric path captured rotating-layer state
    only; on hybrid topologies the rotating list is empty so the path
    fell through to end-of-user fallback every turn, forcing full
    re-prefill on the next request.
    """

    def test_pure_attention_returns_empty(self):
        # Gemma 4: rotating + standard KV, no ArraysCache.
        assert _capture_arrays_layers_for_snapshot([KVCache(), KVCache()]) == []

    def test_arrays_only_returns_per_layer_snapshots(self):
        ac1 = _arrays_cache(fill=1.0)
        ac2 = _arrays_cache(fill=2.0)
        snaps = _capture_arrays_layers_for_snapshot([ac1, ac2])
        assert len(snaps) == 2
        assert snaps[0] is not None and snaps[1] is not None
        assert snaps[0][0].tolist() == ac1.state[0].tolist()
        assert snaps[1][0].tolist() == ac2.state[0].tolist()

    def test_mixed_layers_return_positionally_aligned(self):
        # Hybrid topology: KV layers get None, ArraysCache get state.
        # Position alignment is required so the restore path can find
        # each ArraysCache layer by index.
        ac = _arrays_cache(fill=3.0)
        kv = KVCache()
        snaps = _capture_arrays_layers_for_snapshot([kv, ac, kv])
        assert len(snaps) == 3
        assert snaps[0] is None
        assert snaps[1] is not None and snaps[1][0].tolist() == ac.state[0].tolist()
        assert snaps[2] is None

    def test_capture_preserves_state_through_mutation(self):
        # Snapshot is a list of refs to immutable mx arrays. Replacing
        # the live layer's state list afterward must not alter the
        # snapshot's view.
        ac = _arrays_cache(fill=1.0)
        snaps = _capture_arrays_layers_for_snapshot([ac])
        captured_first_array = snaps[0][0]
        ac.state = [mx.full((1, 4), 99.0)]
        assert captured_first_array.tolist() == [[1.0, 1.0, 1.0, 1.0]]


class TestRestoreArraysLayersFromSnapshots:
    def test_restores_arrays_state(self):
        ac = _arrays_cache(fill=1.0)
        snaps = _capture_arrays_layers_for_snapshot([ac])
        ac.state = [mx.full((1, 4), 99.0)]  # mutate live state
        _restore_arrays_layers_from_snapshots([ac], snaps)
        assert ac.state[0].tolist() == [[1.0, 1.0, 1.0, 1.0]]

    def test_skips_none_entries(self):
        # Position alignment with non-ArraysCache layers — None means
        # "this index is not an ArraysCache, leave it alone".
        ac = _arrays_cache(fill=1.0)
        kv = KVCache()
        snaps = [None, [mx.full((1, 4), 7.0)], None]
        _restore_arrays_layers_from_snapshots([kv, ac, kv], snaps)
        assert ac.state[0].tolist() == [[7.0, 7.0, 7.0, 7.0]]

    def test_empty_snapshots_is_noop(self):
        # Pure-attention model: capture returned []; restore must not
        # error and must not touch any layers.
        ac = _arrays_cache(fill=5.0)
        _restore_arrays_layers_from_snapshots([ac], [])
        assert ac.state[0].tolist() == [[5.0, 5.0, 5.0, 5.0]]

    def test_skips_when_layer_type_changes(self):
        # Defensive: snapshot recorded ArraysCache state at index 0,
        # but the live cache at that index is now a KVCache (cache
        # topology changed mid-session). Don't try to assign — would
        # corrupt the KV layer.
        kv = KVCache()
        snaps = [[mx.full((1, 4), 7.0)]]
        _restore_arrays_layers_from_snapshots([kv], snaps)
        # No assignment to kv; absence of exception is the assertion.


class TestAnchorWithinLoopRange:
    """Pins the loop-entry predicate that fixes the short-prefill
    regression. When prior cache reuse leaves a delta below
    ``prefill_step_size`` (Gemma turn-N after a same-anchor turn-N-1),
    the chunked-prefill loop was being skipped entirely, dropping the
    asymmetric anchor for that turn. The next turn would then re-render
    the user message differently and trigger a full re-prefill.
    """

    def test_no_target_returns_false(self):
        assert _anchor_within_loop_range(0, 1000, None) is False

    def test_target_inside_range_returns_true(self):
        # Anchor at 6083, cache at 6000, prompt of 200 tokens →
        # loop processes [6000, 6199), anchor 6083 is inside.
        assert _anchor_within_loop_range(6000, 200, 6083) is True

    def test_target_at_initial_offset_returns_false(self):
        # Cache already at the anchor; no capture needed.
        assert _anchor_within_loop_range(6083, 200, 6083) is False

    def test_target_below_initial_offset_returns_false(self):
        # Cache already past the anchor (older snapshot from prior
        # turn). We don't time-travel backward.
        assert _anchor_within_loop_range(6100, 200, 6083) is False

    def test_target_at_final_token_returns_false(self):
        # Final token reserved for post-loop _step; can't land there.
        # Loop processes [6000, 6199); 6199 is excluded.
        assert _anchor_within_loop_range(6000, 200, 6199) is False

    def test_target_at_pre_final_returns_true(self):
        # Last LANDABLE position — one before the reserved token.
        assert _anchor_within_loop_range(6000, 200, 6198) is True

    def test_short_prefill_with_anchor_in_range_returns_true(self):
        # The PRODUCTION REGRESSION SCENARIO: cache at 6083 (anchor
        # from prior turn), prompt 37 tokens (small re-render), new
        # anchor at 6105 (somewhere in the middle of the new user
        # message). Pre-fix: 37 < 512 → loop skipped → no capture.
        # Post-fix: anchor in range → loop runs.
        assert _anchor_within_loop_range(6083, 37, 6105) is True

    def test_target_far_beyond_range_returns_false(self):
        # Anchor past the loop's reach (the turn-4 fallback case
        # tested elsewhere). Loop entry not gated on this — fallback
        # capture handles it once inside.
        assert _anchor_within_loop_range(0, 100, 5000) is False


class TestShouldCaptureAnchorPrePrefill:
    """Pins the predicate that gates pre-prefill anchor capture. Two
    cases fire (cache state at start of generate_step IS the anchor):
    equality (typical OWUI tool-continuation) and strict-greater
    (degenerate transient where prior fallback advanced the cache).
    """

    def test_equality_returns_true(self):
        # The PRODUCTION SCENARIO: prior turn anchored at 4626;
        # this turn's latest-user-marker is also 4626 (same user
        # message — OWUI tool-continuation). Pre-capture fires.
        assert _should_capture_anchor_pre_prefill(
            snapshot_at_offset=4626,
            initial_cache_offset=4626,
            prompt_cache_present=True,
        ) is True

    def test_strict_greater_returns_true(self):
        # Degenerate transient: a prior turn's fallback advanced
        # the cache to 6253, but this turn's latest-user-marker
        # is still 4626. Can't go back to 4626 (SWA may have
        # wrapped); capture at initial as best-available.
        assert _should_capture_anchor_pre_prefill(
            snapshot_at_offset=4626,
            initial_cache_offset=6253,
            prompt_cache_present=True,
        ) is True

    def test_strict_less_returns_false(self):
        # Normal new-user-message case: anchor is AHEAD of current
        # cache. The chunked-prefill loop will reach it and capture
        # in-loop; pre-capture would be wrong (cache state at
        # initial_cache_offset is NOT the anchor).
        assert _should_capture_anchor_pre_prefill(
            snapshot_at_offset=5500,
            initial_cache_offset=4626,
            prompt_cache_present=True,
        ) is False

    def test_no_target_returns_false(self):
        # Asymmetric rendering disabled or no user-marker found.
        assert _should_capture_anchor_pre_prefill(
            snapshot_at_offset=None,
            initial_cache_offset=4626,
            prompt_cache_present=True,
        ) is False

    def test_cold_start_returns_false(self):
        # First request of a session: no prior cache state to
        # capture. The in-loop capture handles this normally.
        assert _should_capture_anchor_pre_prefill(
            snapshot_at_offset=4601,
            initial_cache_offset=0,
            prompt_cache_present=False,
        ) is False


class TestCaptureAnchorState:
    """Pins the dedupe helper used at both pre-prefill and in-loop
    capture sites. Each call clears its target lists before populating
    so latest-wins semantics hold across multiple in-loop captures.
    """

    def test_populates_all_three_side_channels(self):
        # Gemma-shape cache: rotating layers + plain KV. Helper
        # populates rotating snapshots, leaves arrays empty (no
        # ArraysCache), and writes the offset marker.
        rotating = [_rotating(offset=100, max_size=512)]
        # Pure-attention model: no ArraysCache, so arrays capture
        # returns []. KVCache for non-rotating layer.
        cache = [rotating[0], KVCache()]
        rot_capture: list = []
        arr_capture: list = []
        offset_marker: list = []
        _capture_anchor_state(
            cache,
            offset=100,
            rotating_capture=rot_capture,
            arrays_capture=arr_capture,
            anchor_offset_list=offset_marker,
        )
        assert len(rot_capture) == 1
        assert rot_capture[0].layer_index == 0
        assert arr_capture == []  # no ArraysCache layers
        assert offset_marker == [100]

    def test_clears_lists_before_populating(self):
        # Latest-wins: a second call replaces the first capture's
        # contents (in-loop usage where capture_as_fallback is
        # replaced by capture_and_finalize on a later chunk).
        rotating = [_rotating(offset=200, max_size=512)]
        cache = [rotating[0]]
        rot_capture: list = ["stale_entry"]
        offset_marker: list = [99]
        _capture_anchor_state(
            cache,
            offset=200,
            rotating_capture=rot_capture,
            arrays_capture=None,
            anchor_offset_list=offset_marker,
        )
        assert len(rot_capture) == 1
        assert rot_capture[0].layer_index == 0  # not "stale_entry"
        assert offset_marker == [200]  # not 99

    def test_none_lists_are_no_op(self):
        # Asymmetric rendering disabled / topology has no rotating
        # layers / etc.: passing None for any side-channel skips it.
        cache = [KVCache()]
        # All None — should not raise.
        _capture_anchor_state(
            cache,
            offset=100,
            rotating_capture=None,
            arrays_capture=None,
            anchor_offset_list=None,
        )

    def test_arrays_only_topology(self):
        # Qwen 3.5/3.6 hybrid: ArraysCache + plain KV, no rotating.
        # Helper captures arrays state and offset; rotating stays
        # empty (no rotating layers in the topology).
        ac = _arrays_cache(fill=2.0)
        cache = [ac, KVCache()]
        rot_capture: list = []
        arr_capture: list = []
        offset_marker: list = []
        _capture_anchor_state(
            cache,
            offset=4626,
            rotating_capture=rot_capture,
            arrays_capture=arr_capture,
            anchor_offset_list=offset_marker,
        )
        assert rot_capture == []  # no rotating layers
        assert len(arr_capture) == 2
        assert arr_capture[0] is not None  # ArraysCache at index 0
        assert arr_capture[1] is None  # KVCache at index 1
        assert offset_marker == [4626]


class TestToolContinuationAnchorPath:
    """Pins the production turn-2 cascade observed on 2026-05-03 OWUI
    Gemma-4 trace.

    OWUI tool-continuation flow makes 2-N chat-completion calls per user
    message: model → tool_call → tool_response → model continues → maybe
    another tool_call → ... → final answer. Across these calls the
    rendered ``<|turn>user\\n`` marker stays at the SAME offset (the
    original user message); OWUI doesn't insert a new user-turn marker
    when threading tool_response content.

    The asymmetric anchor logic computes ``snapshot_at_offset`` from the
    rightmost user-turn marker. On call 2+ of a tool flow,
    ``snapshot_at_offset == initial_cache_offset`` (the prior call
    persisted at exactly that anchor). The chunked-prefill loop's
    ``_classify_snapshot_action`` returns ``"skip"`` on every chunk
    because ``cumulative_offset`` advances past ``snapshot_at_offset``
    on the first iteration. No anchor capture fires →
    ``mid_prefill_anchor_offset`` stays empty → asymmetric path falls
    back to end-of-user → the good anchor (4626 in the trace) is
    overwritten with end-of-prefill (6253), which contains
    tool_response content.

    Cascade: on the next NEW user turn (turn 3 "what's 2+2"), the SWA
    ring (max_size=1024) has wrapped past the divergence point. The
    SWA Ring Buffer Corruption Guard correctly refuses the rewind and
    forces full re-prefill (5901 tokens, ~13s wasted in the trace).

    The fix adds a pre-prefill capture step that fires when
    ``initial_cache_offset >= snapshot_at_offset``: the cache state at
    start of generate_step IS the anchor (prior turn's post-gen
    restored layers there), so we capture rotating + arrays snapshots
    BEFORE the prefill loop runs. The post-gen anchor branch then
    persists at the captured offset, leaving the original anchor
    intact.

    This test fails today because ``_should_capture_anchor_pre_prefill``
    doesn't exist yet. After implementing the predicate + wiring it
    into ``generate_step``, the test passes.
    """

    def _drive_full_anchor_flow(
        self,
        initial_cache_offset: int,
        new_tokens_count: int,
        prefill_step_size: int,
        snapshot_at_offset,
        prompt_cache_present: bool = True,
    ):
        """End-to-end simulation of the anchor capture flow:
        pre-prefill check + chunked-prefill loop. Returns
        ``(captured_offset_or_None, snapshot_done)``.

        Mirrors what ``generate_step`` will do post-fix:
          1. Pre-prefill capture if ``_should_capture_anchor_pre_prefill``
             returns True (the new fix).
          2. Chunked-prefill loop with per-chunk
             ``_classify_snapshot_action`` decisions (existing behavior).

        Pre-fix: the predicate doesn't exist; we mirror today's bug
        (no pre-capture) by guarding the import. The in-loop logic
        runs unchanged. For the equality scenario this returns
        ``(None, False)`` — the production bug.
        """
        captured_offset = None
        snapshot_done = False

        # Pre-prefill capture step (the fix's main contribution).
        # Guarded import so this test file stays collectible while
        # the predicate is being implemented.
        try:
            from mlx_vlm.generate import _should_capture_anchor_pre_prefill
        except ImportError:
            _should_capture_anchor_pre_prefill = None

        if _should_capture_anchor_pre_prefill is not None and (
            _should_capture_anchor_pre_prefill(
                snapshot_at_offset=snapshot_at_offset,
                initial_cache_offset=initial_cache_offset,
                prompt_cache_present=prompt_cache_present,
            )
        ):
            captured_offset = initial_cache_offset

        # In-loop capture (mirrors TestChunkedPrefillSnapshotCaptureLoop._drive_loop).
        cumulative_offset = initial_cache_offset
        remaining = new_tokens_count
        while remaining > 1:
            n_to_process = min(prefill_step_size, remaining - 1)
            n_to_process = _adjust_chunk_for_snapshot_landing(
                cumulative_offset,
                n_to_process,
                snapshot_at_offset,
                snapshot_done,
            )
            cumulative_offset += n_to_process
            remaining -= n_to_process

            action = _classify_snapshot_action(
                cumulative_offset,
                snapshot_at_offset,
                snapshot_done,
            )
            if action == "capture_and_finalize":
                captured_offset = cumulative_offset
                snapshot_done = True
            elif action == "capture_as_fallback":
                captured_offset = cumulative_offset

        return captured_offset, snapshot_done

    def test_tool_continuation_equality_captures_at_initial_offset(self):
        """PRODUCTION TURN-2.2 REGRESSION (2026-05-03 OWUI Gemma-4 trace).

        Scenario from the log:
          Turn 2.1 (pre-tool):   anchor=4626 [exact] / prefill_len 4646
          Turn 2.2 (post-tool):  Total Prompt 6253, Skipped 4626, Delta 1627
                                 → fallback at 6253  ← THE BUG

        Inputs match the trace exactly: ``initial_cache_offset=4626``
        (prior turn's persisted anchor), ``new_tokens_count=1627``
        (the RAG-wrap + tool_response content), ``snapshot_at_offset
        =4626`` (the ORIGINAL user-turn marker — OWUI didn't insert
        a new ``<|turn>user\\n`` for the tool-continuation call), and
        ``prefill_step_size=512``.

        Today: pre-prefill capture doesn't exist; in-loop returns
        ``"skip"`` from the first chunk because ``cumulative_offset``
        (= 4626 + 512 = 5138) > ``snapshot_at_offset`` (4626). No
        capture fires; ``captured == None``.

        Post-fix: pre-prefill capture fires (equality case →
        predicate returns True). ``captured == 4626``.
        """
        captured, _ = self._drive_full_anchor_flow(
            initial_cache_offset=4626,
            new_tokens_count=1627,
            prefill_step_size=512,
            snapshot_at_offset=4626,
            prompt_cache_present=True,
        )
        assert captured == 4626, (
            f"Tool-continuation equality case (snapshot==initial==4626) "
            f"must capture at initial_cache_offset; got {captured}. "
            f"Without this, the asymmetric path falls back to end-of-user "
            f"(6253 in the trace), overwriting the good 4626 anchor with "
            f"tool_response content. The next NEW user turn then hits "
            f"SWA Ring Buffer Corruption Guard → full re-prefill of "
            f"~5900 tokens (~13s in the trace)."
        )
