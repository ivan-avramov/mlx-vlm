"""Fork: M48 / C102(b) — retain the latest user turn (and the canonical assistant
turn) across requests on asymmetric-rendering sessions.

Fake caches only; no model. Pre-registered criterion A1 of
docs/specs/c102b-prompt-end-retention.md (stack repo):
  retire offset == prompt_end + canonical_len; a diverging last user turn still
  rewinds to the before-user anchor; a diverging assistant echo rewinds to
  prompt_end; canonical tokens equal the template's history rendering.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from mlx_vlm.generate import ar as ar_module
from mlx_vlm.generate import common as common_module
from mlx_vlm.generate.common import (
    PromptCacheState,
    _retire_asymmetric_session,
    set_session_retain_prompt_end,
)
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput
from mlx_vlm.models.cache import ArraysCache, KVCache
from mlx_vlm.snapshot import DeltaNetSnapshotRing


def _arrays(marker: float) -> ArraysCache:
    c = ArraysCache(1)
    c.state = [mx.full((1, 2, 2), marker)]
    return c


def _kv(offset: int) -> KVCache:
    c = KVCache()
    c.offset = offset
    c.keys = mx.zeros((1, 2, offset, 4))
    c.values = mx.zeros((1, 2, offset, 4))
    return c


def _state_marker(c: ArraysCache) -> float:
    return float(c.state[0][0, 0, 0].item())


# --------------------------------------------------------------------------- ring


class TestRingCaptureStates:
    def test_capture_states_stores_and_orders(self):
        ring = DeltaNetSnapshotRing(max_size=3)
        assert ring.capture_states(10, [None, [mx.ones((1,))]]) is not None
        assert ring.capture_states(5, [None, [mx.ones((1,))]]) is None  # not newer
        assert ring.capture_states(20, [None, [mx.ones((1,))]]) is not None
        assert [s.offset for s in ring._snapshots] == [10, 20]
        assert ring.find_nearest(15).offset == 10

    def test_capture_states_ignores_pure_attention(self):
        ring = DeltaNetSnapshotRing(max_size=3)
        assert ring.capture_states(10, [None, None]) is None
        assert ring.capture_states(10, []) is None
        assert len(ring) == 0


# --------------------------------------------------------------------------- ar.py hook


@pytest.mark.parametrize("draft", [False, True])
def test_generate_step_captures_state_at_prompt_end(draft):
    """The capture lands after the LAST prompt step and before any decode step —
    the dispatch ``n == 0`` capture is one decode token late (P49)."""
    kv, arrays = _kv(0), _arrays(0.0)
    calls = []

    def fake_lm(inputs=None, inputs_embeds=None, cache=None, **kw):
        n = int(inputs_embeds.shape[1]) if inputs_embeds is not None else 1
        cache[0].offset += n
        cache[1].state = [mx.full((1, 2, 2), float(cache[0].offset))]
        calls.append(n)
        return LanguageModelOutput(logits=mx.zeros((1, n, 4)))

    model = MagicMock()
    model.language_model.side_effect = fake_lm
    model.language_model.supports_logits_to_keep = False
    model.get_input_embeddings.return_value = InputEmbeddingsFeatures(
        inputs_embeds=mx.zeros((1, 6, 4))
    )
    rot, arr, off = [], [], []
    kwargs = dict(
        prompt_cache=[kv, arrays],
        max_tokens=3,
        temperature=0,
        prefill_step_size=4,
        prompt_end_rotating_capture=rot,
        prompt_end_arrays_capture=arr,
        prompt_end_offset=off,
    )
    if draft:
        # No drafter is installed for the fake; the point is only that the
        # capture is taken before the speculative/decode hand-off.
        kwargs["draft_model"] = None
    out = list(
        ar_module.generate_step(
            mx.array([[1, 2, 3, 4, 5, 6]]), model, pixel_values=None, mask=None, **kwargs
        )
    )
    assert len(out) == 3
    assert off == [6]  # prompt_end = 6 prompt tokens, no decode tokens
    assert arr and _state_marker(SimpleNamespace(state=arr[1])) == 6.0
    assert _state_marker(arrays) == 6.0 + 3  # live cache moved on during decode
    assert kv.offset == 9


# --------------------------------------------------------------------------- retire


class TestRetire:
    ANCHOR = 100  # before the latest user turn
    PROMPT_END = 160
    GEN = 40

    def _setup(self, canonical=None, ring_size=3, with_arrays=True):
        kv = _kv(self.PROMPT_END + self.GEN)
        cache = [kv, _arrays(999.0)] if with_arrays else [kv]
        full_ids = list(range(self.PROMPT_END))
        state = PromptCacheState(snapshot_ring=DeltaNetSnapshotRing(max_size=ring_size))
        anchor = dict(
            rotating=[], arrays=[None, [mx.full((1, 2, 2), 100.0)]] if with_arrays else [],
            offset=[self.ANCHOR],
        )
        prompt_end = dict(
            rotating=[], arrays=[None, [mx.full((1, 2, 2), 160.0)]] if with_arrays else [],
            offset=[self.PROMPT_END],
        )
        prefilled = []

        def prefill(ids):
            prefilled.append(list(ids))
            kv.offset += len(ids)
            kv.keys = mx.zeros((1, 2, kv.offset, 4))
            kv.values = mx.zeros((1, 2, kv.offset, 4))
            if with_arrays:
                cache[1].state = [mx.full((1, 2, 2), float(kv.offset))]

        return SimpleNamespace(
            kv=kv, cache=cache, full_ids=full_ids, state=state, anchor=anchor,
            prompt_end=prompt_end, prefill=prefill, prefilled=prefilled,
            canonical=canonical,
        )

    def _retire(self, s, canonical_fn=None):
        return _retire_asymmetric_session(
            s.state, s.cache, s.full_ids,
            anchor_rotating=s.anchor["rotating"], anchor_arrays=s.anchor["arrays"],
            anchor_offset=s.anchor["offset"],
            prompt_end_rotating=s.prompt_end["rotating"],
            prompt_end_arrays=s.prompt_end["arrays"], prompt_end_offset=s.prompt_end["offset"],
            canonical_ids=s.canonical, canonical_prefill=canonical_fn or s.prefill,
        )

    def test_retires_at_prompt_end_plus_canonical(self):
        canonical = [1000, 1001, 1002, 1003, 1004]
        s = self._setup(canonical=canonical)
        offset = self._retire(s)
        assert offset == self.PROMPT_END + len(canonical)
        assert s.prefilled == [canonical]  # prefilled once, on the retired cache
        assert s.kv.offset == offset and s.kv.keys.shape[2] == offset
        assert s.state.token_ids == s.full_ids + canonical
        # ring: anchor, prompt_end, canonical end — in that order
        assert [snap.offset for snap in s.state.snapshot_ring._snapshots] == [
            self.ANCHOR, self.PROMPT_END, offset,
        ]
        assert _state_marker(s.cache[1]) == float(offset)

    def test_prompt_end_state_restored_before_canonical_prefill(self):
        seen = []
        s = self._setup(canonical=[7, 8])

        def prefill(ids):
            seen.append((int(s.kv.offset), _state_marker(s.cache[1])))
            s.prefill(ids)

        self._retire(s, canonical_fn=prefill)
        assert seen == [(self.PROMPT_END, 160.0)]  # KV trimmed + DeltaNet restored first

    def test_no_canonical_retires_at_prompt_end(self):
        s = self._setup(canonical=None)
        offset = self._retire(s)
        assert offset == self.PROMPT_END
        assert s.prefilled == []
        assert s.state.token_ids == s.full_ids
        assert s.kv.offset == self.PROMPT_END
        assert _state_marker(s.cache[1]) == 160.0
        assert [snap.offset for snap in s.state.snapshot_ring._snapshots] == [
            self.ANCHOR, self.PROMPT_END,
        ]

    def test_canonical_prefill_failure_falls_back_to_prompt_end(self):
        s = self._setup(canonical=[7, 8, 9])

        def boom(ids):
            s.kv.offset += 1  # partial damage before the failure ...
            s.cache[1][0] = mx.full((1, 2, 2), -1.0)  # ... including an in-place DeltaNet write
            raise RuntimeError("no")

        offset = self._retire(s, canonical_fn=boom)
        assert offset == self.PROMPT_END
        assert s.state.token_ids == s.full_ids
        assert s.kv.offset == self.PROMPT_END
        assert _state_marker(s.cache[1]) == 160.0
        assert [snap.offset for snap in s.state.snapshot_ring._snapshots] == [
            self.ANCHOR, self.PROMPT_END,
        ]

    def test_edited_last_user_turn_rewinds_to_anchor(self):
        s = self._setup(canonical=[7, 8, 9])
        self._retire(s)
        edited = s.full_ids[: self.ANCHOR + 20] + [4242] + s.full_ids[self.ANCHOR + 21 :]
        prefix = s.state.find_prefix_length(edited)
        assert prefix == self.ANCHOR + 20
        assert s.state.snapshot_ring.find_nearest(prefix).offset == self.ANCHOR

    def test_diverging_assistant_echo_rewinds_to_prompt_end(self):
        s = self._setup(canonical=[7, 8, 9])
        self._retire(s)
        echoed = s.full_ids + [7, 8, 4242]
        prefix = s.state.find_prefix_length(echoed)
        assert prefix == self.PROMPT_END + 2
        assert s.state.snapshot_ring.find_nearest(prefix).offset == self.PROMPT_END

    def test_matching_echo_extends_from_canonical_end(self):
        s = self._setup(canonical=[7, 8, 9])
        offset = self._retire(s)
        nxt = s.full_ids + [7, 8, 9] + [11, 12, 13]
        assert s.state.find_prefix_length(nxt) == offset

    def test_stale_ring_entries_past_divergence_are_dropped_first(self):
        # A previous retire left a snapshot at 200; this request diverged at 120,
        # so the anchor capture at 100 must still be accepted (monotonic rule).
        s = self._setup(canonical=[7])
        s.state.token_ids = list(range(120)) + list(range(5000, 5080))
        s.state.snapshot_ring.capture_states(200, [None, [mx.ones((1,))]])
        self._retire(s)
        assert [snap.offset for snap in s.state.snapshot_ring._snapshots] == [
            self.ANCHOR, self.PROMPT_END, self.PROMPT_END + 1,
        ]

    def test_pure_attention_trims_kv_and_keeps_ring_empty(self):
        s = self._setup(canonical=[7, 8], with_arrays=False)
        offset = self._retire(s)
        assert offset == self.PROMPT_END + 2
        assert s.kv.offset == offset
        assert len(s.state.snapshot_ring) == 0

    def test_no_anchor_capture_still_retires_at_prompt_end(self):
        s = self._setup(canonical=None)
        s.anchor = dict(rotating=[], arrays=[], offset=[])
        offset = self._retire(s)
        assert offset == self.PROMPT_END
        assert [snap.offset for snap in s.state.snapshot_ring._snapshots] == [self.PROMPT_END]


def test_toggle_default_on_and_setter():
    assert common_module.session_retain_prompt_end() is True
    set_session_retain_prompt_end(False)
    try:
        assert common_module.session_retain_prompt_end() is False
    finally:
        set_session_retain_prompt_end(True)


# --------------------------------------------------------------------------- server: canonical suffix


class _CharTokenizer:
    """One token per character: offsets equal string positions."""

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _qwen_like_render(messages, add_generation_prompt=True, **kwargs):
    out = ""
    for m in messages:
        content = m.get("content") or ""
        if m["role"] == "assistant" and kwargs.get("preserve_thinking", True):
            reasoning = (m.get("reasoning_content") or "").strip()
            content = f"<think>\n{reasoning}\n</think>\n\n{content}"
        out += f"<|im_start|>{m['role']}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n"
    return out


class TestCanonicalSuffix:
    MESSAGES = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "big pasted file ..."},
    ]

    def _call(self, answer, **over):
        from mlx_vlm.server.openai import _canonical_assistant_suffix

        tk = _CharTokenizer()
        prompt = _qwen_like_render(self.MESSAGES, add_generation_prompt=True, preserve_thinking=True)
        prompt_ids = tk.encode(prompt)
        kw = dict(
            answer_text=answer, prompt_ids=prompt_ids, boundary=len(prompt_ids),
            messages=self.MESSAGES, tokenizer=tk, render=_qwen_like_render,
            template_kwargs={"preserve_thinking": True},
            tool_calls_present=False,
        )
        kw.update(over)
        if "prompt_ids" in over and "boundary" not in over:
            kw["boundary"] = len(over["prompt_ids"])
        return _canonical_assistant_suffix(**kw), tk

    def test_matches_history_rendering_without_thinking(self):
        suffix, tk = self._call("<think>\nplan\n</think>\n\nThe answer is 42.")
        # exactly the template's HISTORY form of the turn, thinking stripped, up to
        # (not including) the next user-turn marker
        assert tk.decode(suffix) == "<think>\n\n</think>\n\nThe answer is 42.<|im_end|>\n"

    def test_plain_answer(self):
        suffix, tk = self._call("Hello.")
        assert tk.decode(suffix) == "<think>\n\n</think>\n\nHello.<|im_end|>\n"

    def test_tool_calls_disable_canonical(self):
        suffix, _ = self._call("<tool_call>{}</tool_call>", tool_calls_present=True)
        assert suffix is None

    def test_empty_content_disables_canonical(self):
        suffix, _ = self._call("<think>\nonly thinking\n</think>\n\n")
        assert suffix is None

    def test_echo_diverging_before_the_boundary_disables_canonical(self):
        suffix, _ = self._call("Hello.", prompt_ids=[1, 2, 3])
        assert suffix is None


def test_session_manager_configure_publishes_toggle():
    from mlx_vlm.server import session_manager

    try:
        session_manager.configure(session_retain_prompt_end=False)
        assert common_module.session_retain_prompt_end() is False
        session_manager.configure(session_retain_prompt_end=True)
        assert common_module.session_retain_prompt_end() is True
    finally:
        set_session_retain_prompt_end(True)


def test_cli_flag_is_declared_with_env_fallback(monkeypatch):
    from mlx_vlm.server import cli

    monkeypatch.setenv("MLX_VLM_SESSION_RETAIN_PROMPT_END", "off")
    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    if parser is None:
        pytest.skip("cli has no build_parser()")
    assert parser.parse_args([]).cache_session_retain_prompt_end == "off"


def test_restore_arrays_does_not_alias_the_snapshot():
    from mlx_vlm.generate.common import _restore_arrays_layers_from_snapshots

    c = _arrays(1.0)
    snap = [[mx.full((1, 2, 2), 7.0)]]
    _restore_arrays_layers_from_snapshots([c], snap)
    c[0] = mx.full((1, 2, 2), 8.0)  # what a forward pass does
    assert float(snap[0][0][0, 0, 0].item()) == 7.0  # snapshot untouched


def test_rewind_restore_does_not_alias_the_ring_entry():
    """Pre-existing hazard found in the M48 review: after a snapshot-ring rewind the
    next forward pass wrote into the ring's own state list."""
    from mlx_vlm.generate.common import _restore_deltanet_state

    ring = DeltaNetSnapshotRing(max_size=3)
    c = _arrays(3.0)
    ring.capture(offset=10, cache=[c])
    c[0] = mx.full((1, 2, 2), 4.0)  # generation moves on
    snap = ring.find_nearest(10)
    _restore_deltanet_state([c], snap.states)
    assert _state_marker(c) == 3.0
    c[0] = mx.full((1, 2, 2), 5.0)  # forward pass after the rewind
    assert float(snap.states[0][0][0, 0, 0].item()) == 3.0  # ring entry untouched


class _BorrowOnceTokenizer(_CharTokenizer):
    """First encode raises pyo3's transient borrow error (seen live 2026-09-28)."""

    def __init__(self):
        self.calls = 0

    def encode(self, text, add_special_tokens=False):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Already borrowed")
        return super().encode(text, add_special_tokens)


def test_anchor_offset_retries_a_transient_borrow_error():
    from mlx_vlm.generate.common import _compute_anchor_before_latest_user_offset

    tk = _BorrowOnceTokenizer()
    prompt = "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert _compute_anchor_before_latest_user_offset(prompt, tk) == prompt.index("<|im_start|>user\n")
    assert tk.calls == 2


def test_canonical_suffix_retries_a_transient_borrow_error():
    from mlx_vlm.server.openai import _canonical_assistant_suffix

    tk = _BorrowOnceTokenizer()
    msgs = TestCanonicalSuffix.MESSAGES
    prompt_ids = _CharTokenizer().encode(_qwen_like_render(msgs, add_generation_prompt=True))
    suffix = _canonical_assistant_suffix(
        answer_text="Hello.", prompt_ids=prompt_ids, boundary=len(prompt_ids), messages=msgs,
        tokenizer=tk, render=_qwen_like_render, template_kwargs={"preserve_thinking": True},
        tool_calls_present=False,
    )
    assert _CharTokenizer().decode(suffix) == "<think>\n\n</think>\n\nHello.<|im_end|>\n"


# --------------------------------------------------------------------------- review round (Codex, 2026-09-28)


def test_buffered_window_rewind_guard_matches_the_real_cache_behaviour():
    """P3 repro with the real classes: window 4, buffer 32, tokens 0..36, rewind to 35."""
    from mlx_vlm.generate.common import _rotating_rewind_safe
    from mlx_vlm.models.cache import BufferedRotatingKVCache, RotatingKVCache

    def tok(i):
        return mx.full((1, 1, 1, 2), float(i))

    base = RotatingKVCache(max_size=4)
    for i in range(37):
        base.update_and_fetch(tok(i), tok(i))
    c = BufferedRotatingKVCache.from_cache(base, buffer_size=32)
    assert c.start_position == 33
    assert _rotating_rewind_safe([c], 35) is False  # window [31, 35) not retained
    assert _rotating_rewind_safe([c], 37) is True


def test_rotating_restore_refuses_a_changed_layout():
    """P2: MTP swaps RotatingKVCache for BufferedRotatingKVCache after prefill."""
    from mlx_vlm.models.cache import BufferedRotatingKVCache, RotatingKVCache
    from mlx_vlm.snapshot import capture_rotating, restore_rotating

    plain = RotatingKVCache(max_size=8)
    plain.update_and_fetch(mx.ones((1, 1, 3, 2)), mx.ones((1, 1, 3, 2)))
    snap = capture_rotating(plain, 0)
    assert snap.layer_type == "RotatingKVCache"
    buffered = BufferedRotatingKVCache.from_cache(plain, buffer_size=16)
    with pytest.raises(TypeError):
        restore_rotating(buffered, snap)
    restore_rotating(plain, snap)  # same layout still restores


class TestRetireContainment(TestRetire):
    def test_prompt_end_offset_mismatch_is_refused(self):
        s = self._setup(canonical=[7])
        s.prompt_end["offset"] = [self.PROMPT_END + 10]  # e.g. media token expansion
        with pytest.raises(ValueError):
            self._retire(s)

    def test_rotating_layout_mismatch_drops_the_session(self):
        from mlx_vlm.models.cache import BufferedRotatingKVCache, RotatingKVCache
        from mlx_vlm.snapshot import capture_rotating

        s = self._setup(canonical=[7])
        plain = RotatingKVCache(max_size=8)
        plain.update_and_fetch(mx.ones((1, 1, 3, 2)), mx.ones((1, 1, 3, 2)))
        s.prompt_end["rotating"] = [capture_rotating(plain, 2)]
        s.cache.append(BufferedRotatingKVCache.from_cache(plain, buffer_size=16))
        s.state.token_ids = [1, 2, 3]; s.state.cache = s.cache
        assert self._retire(s) is None
        assert s.state.token_ids is None and s.state.cache is None  # dropped, not published
        assert s.prefilled == []

    def test_anchor_survives_repeated_continuations(self):
        """P6: continuation turns without a new user marker must not evict the anchor."""
        s = self._setup(canonical=[7])
        self._retire(s)
        ring = s.state.snapshot_ring
        for k in range(1, 4):  # three more retires, same anchor, growing prompt ends
            end = self.PROMPT_END + 40 * k
            ring.capture_states(end, [None, [mx.ones((1,))]])
        offsets = [snap.offset for snap in ring._snapshots]
        assert self.ANCHOR in offsets and len(offsets) == 3
        assert [snap.pinned for snap in ring._snapshots if snap.offset == self.ANCHOR] == [True]


def test_ring_pin_moves_to_the_latest_anchor():
    ring = DeltaNetSnapshotRing(max_size=2)
    ring.capture_states(10, [None, [mx.ones((1,))]], pinned=True)
    ring.capture_states(20, [None, [mx.ones((1,))]])
    ring.capture_states(30, [None, [mx.ones((1,))]], pinned=True)  # new anchor: old pin released
    assert [(s.offset, s.pinned) for s in ring._snapshots] == [(10, False), (30, True)] or \
           [(s.offset, s.pinned) for s in ring._snapshots] == [(20, False), (30, True)]
    ring.capture_states(40, [None, [mx.ones((1,))]])
    assert 30 in [s.offset for s in ring._snapshots] and len(ring) == 2


def test_canonical_prefill_primes_absolute_positions():
    """P1: the canonical suffix alone would yield suffix-local positions; the prefill
    must present FULL-sequence position metadata so the LM slices at the cache offset."""
    from mlx_vlm.generate.dispatch import _prefill_canonical_suffix

    seen = {}

    class LM:
        _rope_deltas = None
        _position_ids = None

        def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None, mask=None):
            n = int(input_ids.shape[-1])
            return mx.broadcast_to(mx.arange(n)[None, None, :], (3, 1, n)), mx.zeros((1, 1))

        def __call__(self, inputs=None, inputs_embeds=None, cache=None, **kw):
            pid = kw.get("position_ids")
            seen.setdefault("position_ids", []).append(None if pid is None else pid.shape)
            n = int(inputs_embeds.shape[1]) if inputs_embeds is not None else int(inputs.shape[-1])
            cache[0].offset += n
            return LanguageModelOutput(logits=mx.zeros((1, n, 4)))

    model = MagicMock()
    model.language_model = LM()
    model.get_input_embeddings.side_effect = lambda ids, pv, **kw: InputEmbeddingsFeatures(
        inputs_embeds=mx.zeros((1, int(ids.shape[-1]), 4)),
        position_ids=mx.zeros((3, 1, int(ids.shape[-1]))),  # suffix-local (what P1 warned about)
    )
    kv = _kv(160)
    _prefill_canonical_suffix(model, [kv], list(range(160)), [7, 8, 9, 10], {"prefill_step_size": 2})
    assert kv.offset == 164
    assert seen["position_ids"] and all(shape[-1] == 164 for shape in seen["position_ids"])


def test_prompt_end_capture_precedes_the_speculative_handoff(monkeypatch):
    """P8: with a drafter the capture must already exist when run_speculative_rounds is entered."""
    kv, arrays = _kv(0), _arrays(0.0)

    def fake_lm(inputs=None, inputs_embeds=None, cache=None, **kw):
        n = int(inputs_embeds.shape[1]) if inputs_embeds is not None else 1
        cache[0].offset += n
        cache[1].state = [mx.full((1, 2, 2), float(cache[0].offset))]
        return LanguageModelOutput(logits=mx.zeros((1, n, 4)))

    model = MagicMock(); model.language_model.side_effect = fake_lm
    model.get_input_embeddings.return_value = InputEmbeddingsFeatures(inputs_embeds=mx.zeros((1, 5, 4)))
    off, arr = [], []
    entered = {}

    def fake_rounds(model_, draft_model, prompt_cache, *a, **kw):
        entered["offset_at_entry"] = list(off); entered["kv_offset"] = int(prompt_cache[0].offset)
        yield 1, mx.zeros((4,))

    import mlx_vlm.speculative.drafters as drafters_module

    monkeypatch.setattr(ar_module, "run_speculative_rounds", fake_rounds)
    monkeypatch.setattr(drafters_module, "validate_drafter_compatibility", lambda *a, **k: None)
    monkeypatch.setattr(ar_module, "speculative_prefill_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(ar_module, "SpeculativePrefill", lambda *a, **k: SimpleNamespace(
        kwargs={}, append=lambda out: None, finish=lambda out: out))
    drafter = MagicMock()
    out = list(ar_module.generate_step(
        mx.array([[1, 2, 3, 4, 5]]), model, pixel_values=None, mask=None,
        prompt_cache=[kv, arrays], max_tokens=1, temperature=0, prefill_step_size=4,
        draft_model=drafter, draft_kind="mtp",
        prompt_end_rotating_capture=[], prompt_end_arrays_capture=arr, prompt_end_offset=off,
    ))
    assert out and entered["offset_at_entry"] == [5] and entered["kv_offset"] == 5
    assert _state_marker(SimpleNamespace(state=arr[1])) == 5.0


class TestCanonicalSuffixTemplates:
    """P5/P8: the guarantee holds when the generation header is a prefix of the history form."""

    MESSAGES = TestCanonicalSuffix.MESSAGES

    def _run(self, render_gen, render_hist, answer="Hi."):
        from mlx_vlm.server.openai import _canonical_assistant_suffix

        from mlx_vlm.server.openai import _retention_boundary

        tk = _CharTokenizer()
        prompt = render_gen(self.MESSAGES)
        ids = tk.encode(prompt)
        render = lambda msgs, **kw: render_hist(msgs)  # noqa: E731
        boundary = _retention_boundary(prompt_ids=ids, messages=self.MESSAGES, tokenizer=tk,
                                       render=render, template_kwargs={})
        suffix = _canonical_assistant_suffix(
            answer_text=answer, prompt_ids=ids, boundary=boundary, messages=self.MESSAGES,
            tokenizer=tk, render=render, template_kwargs={}, tool_calls_present=False)
        return (boundary, suffix, ids), tk

    def test_qwen_thinking_disabled_header_is_compatible(self):
        # generation: '...assistant\n<think>\n\n</think>\n\n'; history: same + content
        def gen(msgs):
            return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + \
                   "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        def hist(msgs):
            out = ""
            for m in msgs:
                c = m.get("content") or ""
                if m["role"] == "assistant": c = f"<think>\n\n</think>\n\n{c}"
                out += f"<|im_start|>{m['role']}\n{c}<|im_end|>\n"
            return out + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        (boundary, suffix, ids), tk = self._run(gen, hist)
        assert boundary == len(ids)
        assert tk.decode(suffix) == "Hi.<|im_end|>\n"

    def test_forced_open_thinking_header_retains_before_the_unstable_tail(self):
        # Gemma-style: generation ends with an OPEN thinking marker the history form drops.
        def gen(msgs):
            return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + \
                   "<|im_start|>assistant\n<|think|>"
        def hist(msgs):
            return "".join(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n" for m in msgs) + \
                   "<|im_start|>assistant\n<|think|>"
        # Gemma-style: the open marker is not in the history form, so the boundary
        # sits BEFORE it (the whole user turn is still retained) and the canonical
        # suffix restarts from there.
        (boundary, suffix, ids), tk = self._run(gen, hist)
        assert tk.decode(ids[:boundary]).endswith("<|im_start|>assistant\n")
        assert boundary == len(ids) - len("<|think|>")
        assert tk.decode(suffix) == "Hi.<|im_end|>\n"


def test_cached_request_forwards_the_request_scoped_hook(monkeypatch):
    """P7: the hook travels with the queued request, never on the session object."""
    import sys
    from queue import Queue

    from mlx_vlm.server import generation as G

    seen = {}

    def fake_stream_generate(*a, **kw):
        seen.update(kw)
        return iter(())

    monkeypatch.setattr(sys.modules["mlx_vlm.generate"], "stream_generate", fake_stream_generate)
    import threading

    rg = G.ResponseGenerator.__new__(G.ResponseGenerator)
    rg.model = SimpleNamespace(); rg.processor = SimpleNamespace(); rg.vision_cache = None
    rg.kv_bits = None; rg.kv_group_size = None; rg.kv_quant_scheme = None; rg.quantized_kv_start = None
    rg._cancelled = set(); rg._cancel_lock = threading.Lock()
    hook = lambda text, ids: None  # noqa: E731
    rg._process_cached_request(
        rqueue=Queue(), prompt="hello", images=None, args=G.GenerationArguments(),
        prompt_tokens=1, prompt_cache_state=SimpleNamespace(), session_retention=hook,
    )
    assert seen.get("session_retention") is hook


# --------------------------------------------------------------------------- review round 2


def test_ring_pin_promotes_an_existing_entry_at_the_same_offset():
    """P9: a continuation whose latest user turn starts exactly where the previous
    retire ended must move the pin to that existing entry."""
    ring = DeltaNetSnapshotRing(max_size=3)
    ring.capture_states(100, [None, [mx.ones((1,))]], pinned=True)
    ring.capture_states(160, [None, [mx.ones((1,))]])
    ring.capture_states(161, [None, [mx.ones((1,))]])
    assert ring.capture_states(161, [None, [mx.ones((1,))]], pinned=True) is not None
    assert [(s.offset, s.pinned) for s in ring._snapshots] == [(100, False), (160, False), (161, True)]
    ring.capture_states(200, [None, [mx.ones((1,))]]); ring.capture_states(201, [None, [mx.ones((1,))]])
    assert 161 in [s.offset for s in ring._snapshots] and ring.find_nearest(180).offset == 161


class TestRetainBoundaryLanding:
    def _model(self, kv, arrays, n_prompt):
        def fake_lm(inputs=None, inputs_embeds=None, cache=None, **kw):
            n = int(inputs_embeds.shape[1]) if inputs_embeds is not None else 1
            cache[0].offset += n
            cache[1].state = [mx.full((1, 2, 2), float(cache[0].offset))]
            return LanguageModelOutput(logits=mx.zeros((1, n, 4)))
        model = MagicMock(); model.language_model.side_effect = fake_lm
        model.get_input_embeddings.return_value = InputEmbeddingsFeatures(inputs_embeds=mx.zeros((1, n_prompt, 4)))
        return model

    def _run(self, retain, step, n_prompt=6, initial=0):
        kv, arrays = _kv(initial), _arrays(float(initial))
        off, arr = [], []
        list(ar_module.generate_step(
            mx.array([list(range(1, n_prompt + 1))]), self._model(kv, arrays, n_prompt),
            pixel_values=None, mask=None, prompt_cache=[kv, arrays], max_tokens=1, temperature=0,
            prefill_step_size=step, prompt_end_rotating_capture=[], prompt_end_arrays_capture=arr,
            prompt_end_offset=off, retain_at_offset=retain))
        return off, (arr[1][0][0, 0, 0].item() if arr else None)

    def test_one_before_prompt_end_lands_on_the_last_chunk(self):
        off, marker = self._run(retain=5, step=4)
        assert off == [5] and marker == 5.0

    def test_mid_prompt_boundary_shrinks_the_chunk(self):
        off, marker = self._run(retain=3, step=4)
        assert off == [3] and marker == 3.0

    def test_prompt_end_is_captured_after_the_final_step(self):
        off, marker = self._run(retain=None, step=4)
        assert off == [6] and marker == 6.0

    def test_default_chunking_lands_the_boundary_too(self):
        # prefill_step_size=None resolves to the default chunk size inside
        # generate_step, so the landing still happens.
        off, marker = self._run(retain=5, step=None)
        assert off == [5] and marker == 5.0

    def test_boundary_at_the_live_cache_offset_is_captured_from_the_start_state(self):
        off, marker = self._run(retain=10, step=4, initial=10)
        assert off == [10] and marker == 10.0


def test_retire_at_a_boundary_before_prompt_end():
    t = TestRetire(); s = t._setup(canonical=[7, 8])
    s.prompt_end["offset"] = [t.PROMPT_END - 1]
    s.prompt_end["arrays"] = [None, [mx.full((1, 2, 2), 159.0)]]
    offset = _retire_asymmetric_session(
        s.state, s.cache, s.full_ids,
        anchor_rotating=[], anchor_arrays=s.anchor["arrays"], anchor_offset=s.anchor["offset"],
        prompt_end_rotating=[], prompt_end_arrays=s.prompt_end["arrays"], prompt_end_offset=s.prompt_end["offset"],
        canonical_ids=[7, 8], canonical_prefill=s.prefill, boundary=t.PROMPT_END - 1)
    assert offset == t.PROMPT_END + 1
    assert s.state.token_ids == s.full_ids[: t.PROMPT_END - 1] + [7, 8]
    assert [snap.offset for snap in s.state.snapshot_ring._snapshots] == [t.ANCHOR, t.PROMPT_END - 1, offset]


def _pick_tokenizer():
    import glob, os
    for d in glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--caslca--Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed/snapshots/*/")):
        if os.path.exists(d + "tokenizer_config.json"):
            try:
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(d)
            except Exception:
                return None
    return None


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_shipped_qwen_template_boundary_and_canonical_suffix(enable_thinking):
    """P5 with the SHIPPED tokenizer/template: thinking-on generation ends in
    ``<think>\n`` whose final token re-tokenises with the history's ``\n\n`` — the
    boundary is one token short of prompt end and the canonical suffix restarts there."""
    tk = _pick_tokenizer()
    if tk is None:
        pytest.skip("pick tokenizer not in the local HF cache")
    from mlx_vlm.server.openai import _canonical_assistant_suffix, _retention_boundary

    msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Say hi."}]
    tkw = {"preserve_thinking": True, "enable_thinking": enable_thinking}
    render = lambda m, **kw: tk.apply_chat_template(m, tokenize=False, **kw)  # noqa: E731
    prompt = render(msgs, add_generation_prompt=True, **tkw)
    ids = tk.encode(prompt, add_special_tokens=False)
    boundary = _retention_boundary(prompt_ids=ids, messages=msgs, tokenizer=tk, render=render, template_kwargs=tkw)
    assert boundary is not None and len(ids) - 2 <= boundary <= len(ids)
    if enable_thinking:
        assert boundary == len(ids) - 1  # the merged newline
    suffix = _canonical_assistant_suffix(
        answer_text="<think>\nplan\n</think>\n\nHi.", prompt_ids=ids, boundary=boundary, messages=msgs,
        tokenizer=tk, render=render, template_kwargs=tkw, tool_calls_present=False)
    assert suffix
    nxt = render(msgs + [{"role": "assistant", "content": "Hi."}, {"role": "user", "content": "."}],
                 add_generation_prompt=True, **tkw)
    nxt_ids = tk.encode(nxt, add_special_tokens=False)
    assert nxt_ids[: boundary + len(suffix)] == ids[:boundary] + suffix  # byte-exact through the assistant turn
    assert tk.decode(ids[:boundary] + suffix).endswith("Hi.<|im_end|>\n")


# --------------------------------------------------------------------------- review round 3


def test_fallback_capture_is_not_pinned_and_the_true_anchor_survives():
    """P10: repeated continuations capture the live cache offset as a fallback anchor;
    only an exact landing on the user marker may take the pin."""
    t = TestRetire(); s = t._setup(canonical=None)
    t._retire(s)  # exact anchor 100 (anchor_target None => exact)
    ring = s.state.snapshot_ring
    for end in (200, 240, 280):
        s2 = SimpleNamespace(
            state=s.state, cache=s.cache, full_ids=list(range(end)),
            anchor=dict(rotating=[], arrays=[None, [mx.ones((1,))]], offset=[end - 40]),
            prompt_end=dict(rotating=[], arrays=[None, [mx.full((1, 2, 2), float(end))]], offset=[end]),
            prefill=s.prefill, canonical=None)
        s2.kv = s.kv; s2.kv.offset = end + 5
        _retire_asymmetric_session(
            s2.state, s2.cache, s2.full_ids, anchor_rotating=[], anchor_arrays=s2.anchor["arrays"],
            anchor_offset=s2.anchor["offset"], prompt_end_rotating=[], prompt_end_arrays=s2.prompt_end["arrays"],
            prompt_end_offset=s2.prompt_end["offset"], canonical_ids=None, canonical_prefill=s2.prefill,
            boundary=end, anchor_target=100)  # marker still at 100; capture is a fallback
    offsets = [(snap.offset, snap.pinned) for snap in ring._snapshots]
    assert (100, True) in offsets and len(offsets) == 3
    assert ring.find_nearest(120).offset == 100


def test_hybrid_cache_without_any_capture_is_not_published():
    """P4 missing-capture route: the pure-attention fallback trims KV only; a hybrid cache
    must not take it. (The dispatcher branch keys on _has_non_trimmable.)"""
    from mlx_vlm.generate.common import _has_non_trimmable

    assert _has_non_trimmable([_kv(10), _arrays(1.0)]) is True
    assert _has_non_trimmable([_kv(10)]) is False
