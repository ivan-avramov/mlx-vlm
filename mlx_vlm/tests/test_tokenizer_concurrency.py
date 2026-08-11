"""Concurrency safety of the per-request tokenizer encodes.

HF's fast (Rust-backed) tokenizer is not safe under concurrent calls from
multiple request threads, and continuous batching shares ONE tokenizer across
all in-flight requests. `7e3477b` fixed three call sites by routing them
through `prompt_utils.cached_special_token_encode`; it missed the raw encodes
inside `ThinkingBudgetCriteria.__init__` (which the *fixed* sites construct)
and inside `ThinkingAwareLogitsProcessor.__init__`.

**The exact mechanism, established empirically rather than assumed** (see
`_RefCellTokenizer` for the model it produced):

    transformers' `set_truncation_and_padding()` runs at the top of every
    `.encode()` and calls `self._tokenizer.no_padding()` / `no_truncation()`
    -- but ONLY when the backend currently has padding/truncation set
    (tokenization_utils_tokenizers.py:815-820). Those are `&mut self` pyo3
    methods, so they `borrow_mut()` the Rust RefCell. A *long* concurrent
    `encode`/`encode_batch` releases the GIL while holding a shared `borrow()`,
    and the `borrow_mut()` then fails with pyo3's `PyBorrowMutError`, whose
    message is "Already borrowed".

Two consequences that scoped this fix, both measured on a real
`Qwen2Tokenizer` before any code was changed:

- The **victim** is always a plain `.encode()`; the **holder** is a long
  padded/truncated encode (a processor call during `prepare_inputs`).
- **`.decode()` is not involved at all.** It is neither victim nor holder:
  `_decode` goes straight to `self._tokenizer.decode(...)` and never calls
  `set_truncation_and_padding`, so it never `borrow_mut`s; and racing 8 long
  decodes against plain encodes with padding pinned on produced zero errors.
  That is why `dispatch.py`'s `tokenizer.decode(input_ids...)`, two lines above
  the reported traceback on the same code path, needs no change.
"""

import threading
import time

import pytest

from mlx_vlm.prompt_utils import (
    _TOKEN_ENCODE_CACHE,
    THINKING_FORMATS,
    _encode_retrying_on_borrow_error,
    prewarm_special_token_encodes,
)
from mlx_vlm.structured import ThinkingAwareLogitsProcessor
from mlx_vlm.utils import ThinkingBudgetCriteria


@pytest.fixture(autouse=True)
def _clear_encode_cache():
    """Isolate tests from each other's cache entries.

    `cached_special_token_encode` keys on `id(tokenizer)`, so a fake tokenizer
    that has been garbage collected can have its address reused by the next
    one and hand back the previous fake's ids. Clearing keeps these tests
    deterministic; the keying itself is a separate pre-existing concern.
    """
    _TOKEN_ENCODE_CACHE.clear()
    yield
    _TOKEN_ENCODE_CACHE.clear()


class _RefCellTokenizer:
    """Models the Rust RefCell borrow semantics that produce "Already borrowed".

    Faithful to the mechanism established in this module's docstring: every
    `encode()` first attempts the `borrow_mut()` that
    `set_truncation_and_padding` performs, which raises if any other thread is
    inside the shared borrow; it then holds a shared borrow across a simulated
    GIL-released Rust call.

    This models the case that actually occurs in production -- a plain encode
    racing a concurrent padded processor encode -- deterministically, so it can
    gate in CI where no model is available.
    """

    def __init__(self):
        self._guard = threading.Lock()
        self._shared_borrows = 0
        self.encode_calls = []

    def encode(self, text, add_special_tokens=True):
        self.encode_calls.append(text)
        # set_truncation_and_padding() -> no_padding() -> Rust borrow_mut().
        with self._guard:
            if self._shared_borrows:
                raise RuntimeError("Already borrowed")
            self._shared_borrows += 1
        try:
            # The Rust encode call, with the GIL released.
            time.sleep(0.002)
            return [1000 + (len(text) % 97)]
        finally:
            with self._guard:
                self._shared_borrows -= 1

    def decode(self, ids, **kwargs):  # pragma: no cover - not exercised
        return ""


def _run_concurrently(target, threads=8):
    """Run `target` on N threads, returning every exception raised."""
    errors = []
    barrier = threading.Barrier(threads)

    def worker():
        barrier.wait()
        try:
            target()
        except BaseException as exc:  # noqa: BLE001 - the assertion is the point
            errors.append(f"{type(exc).__name__}: {exc}")

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    return errors


def test_thinking_budget_criteria_is_concurrency_safe():
    """Eight request threads may construct the criteria against one tokenizer.

    Pre-fix this raises `RuntimeError: Already borrowed` from
    `utils.py`'s three raw `tokenizer.encode(...)` calls -- the production
    traceback.
    """
    tokenizer = _RefCellTokenizer()

    def build():
        ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=64,
            thinking_end_token="</think>",
            thinking_start_token="<think>",
            enable_thinking=True,
        )

    assert _run_concurrently(build) == []


def test_thinking_aware_logits_processor_is_concurrency_safe():
    """The same defect in `structured.py`, found by counting the call sites.

    `ResponseGenerator._wrap_processors_until_thinking_done` constructs one of
    these per logits processor per request, on the shared tokenizer.
    """
    tokenizer = _RefCellTokenizer()

    def build():
        ThinkingAwareLogitsProcessor(
            processor=lambda tokens, logits: logits,
            tokenizer=tokenizer,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
            enable_thinking=True,
        )

    assert _run_concurrently(build) == []


def test_thinking_budget_criteria_encodes_each_string_once():
    """The structural half: repeated construction must not re-encode.

    A concurrency test can only observe a race it happens to win; this pins the
    property that removes the race -- one encode per distinct string per
    tokenizer, no matter how many requests arrive.
    """
    tokenizer = _RefCellTokenizer()
    for _ in range(5):
        ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=64,
            thinking_end_token="</think>",
            thinking_start_token="<think>",
            enable_thinking=True,
        )

    assert sorted(tokenizer.encode_calls) == ["\n", "</think>", "<think>"]


def test_thinking_budget_criteria_resolves_the_same_ids_as_a_raw_encode():
    """The cache must not change the values the criteria resolves.

    Guards the restated contract of the call sites: they take `[-1]` of the
    returned list, so the helper has to return the full id list.
    """
    tokenizer = _RefCellTokenizer()
    expected_end = tokenizer.encode("</think>", add_special_tokens=False)[-1]
    expected_start = tokenizer.encode("<think>", add_special_tokens=False)[-1]

    criteria = ThinkingBudgetCriteria(
        tokenizer=tokenizer,
        thinking_budget=64,
        thinking_end_token="</think>",
        thinking_start_token="<think>",
        enable_thinking=True,
    )

    assert criteria.thinking_end_token_id == expected_end
    assert criteria.thinking_start_token_id == expected_start
    assert criteria._forced_sequence[-1] == expected_end


class _FlakyTokenizer:
    """Raises the borrow error `fail_times` times, then succeeds."""

    def __init__(self, fail_times, message="Already borrowed"):
        self.remaining = fail_times
        self.message = message
        self.calls = 0

    def encode(self, text, add_special_tokens=True):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError(self.message)
        return [7]


def test_retry_gets_past_a_transient_borrow_collision():
    """The cold miss is what the retry exists for: a passing window is enough."""
    tokenizer = _FlakyTokenizer(fail_times=3)

    assert _encode_retrying_on_borrow_error(tokenizer, "</think>", backoff=0) == [7]
    assert tokenizer.calls == 4


def test_retry_reraises_once_attempts_are_exhausted():
    """A permanent failure must surface, not become a wrong token id."""
    tokenizer = _FlakyTokenizer(fail_times=99)

    with pytest.raises(RuntimeError, match="Already borrowed"):
        _encode_retrying_on_borrow_error(tokenizer, "</think>", attempts=3, backoff=0)
    assert tokenizer.calls == 3


def test_retry_does_not_swallow_an_unrelated_runtime_error():
    """Only the borrow collision is transient; everything else propagates."""
    tokenizer = _FlakyTokenizer(fail_times=99, message="vocab file is corrupt")

    with pytest.raises(RuntimeError, match="vocab file is corrupt"):
        _encode_retrying_on_borrow_error(tokenizer, "</think>", backoff=0)
    assert tokenizer.calls == 1


def test_prewarm_covers_every_registered_delimiter_and_the_defaults():
    """The pre-warm is what keeps a cold miss off the request thread.

    If a format's literals are missing from it, the first request using that
    family encodes on a request thread and the race returns.
    """
    tokenizer = _RefCellTokenizer()
    prewarm_special_token_encodes(tokenizer)

    warmed = set(tokenizer.encode_calls)
    assert {"<think>", "</think>", "\n"} <= warmed
    for fmt in THINKING_FORMATS:
        assert set(fmt.openers) <= warmed, fmt.name
        assert set(fmt.closers) <= warmed, fmt.name

    # Warming twice must not re-encode: it goes through the same cache.
    before = len(tokenizer.encode_calls)
    prewarm_special_token_encodes(tokenizer)
    assert len(tokenizer.encode_calls) == before


def test_prewarm_survives_a_tokenizer_that_cannot_encode():
    """Warming is advisory -- it must never make model loading fail."""

    class _Hostile:
        def encode(self, text, add_special_tokens=True):
            raise ValueError("no vocab")

    prewarm_special_token_encodes(_Hostile())  # must not raise


# ── The real-tokenizer reproduction ──────────────────────────────────────────
# Kept because the fake above is only as good as the mechanism it models. This
# is the production shape: a padded batch encode (the borrow holder, e.g. a
# processor call in prepare_inputs) racing the criteria construction.

_REPRO_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def _load_real_fast_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:  # pragma: no cover
        return None
    try:
        tok = AutoTokenizer.from_pretrained(_REPRO_MODEL, local_files_only=True)
    except Exception:  # pragma: no cover - not cached on this machine
        return None
    return tok if getattr(tok, "is_fast", False) else None


def test_real_fast_tokenizer_under_concurrent_padded_encodes():
    """N threads building the criteria while a padded batch encode runs.

    Mirrors the server: `ResponseGenerator._initialize_model` pre-warms the
    encode cache while it is still single-threaded, then request threads
    construct the criteria concurrently. Pre-fix the pre-warm is irrelevant --
    `__init__` used the raw tokenizer and ignored the cache -- so this raises
    `RuntimeError: Already borrowed`; post-fix the request threads perform no
    encode at all.

    The holder threads here tokenize back-to-back, saturating the tokenizer far
    beyond what real prefill does. That is deliberate: it is what proved the
    cache alone is insufficient and that the pre-warm is load-bearing.

    Skipped when the model is not in the local HF cache, which is why the
    `_RefCellTokenizer` tests above are the ones that gate.
    """
    tokenizer = _load_real_fast_tokenizer()
    if tokenizer is None:
        pytest.skip(f"{_REPRO_MODEL} not available in the local HF cache")

    prewarm_special_token_encodes(tokenizer)

    long_text = "The quick brown fox jumps over the lazy dog. " * 800
    errors = []
    stop = threading.Event()

    def holder():
        # Sets padding/truncation on the backend and releases the GIL inside
        # the Rust call -- the shared borrow the criteria's encode collides with.
        while not stop.is_set():
            try:
                tokenizer(
                    [long_text, long_text],
                    padding=True,
                    truncation=True,
                    max_length=4096,
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(f"holder {type(exc).__name__}: {exc}")
                return

    holders = [threading.Thread(target=holder) for _ in range(2)]
    for t in holders:
        t.start()
    try:

        def build():
            for _ in range(40):
                ThinkingBudgetCriteria(
                    tokenizer=tokenizer,
                    thinking_budget=64,
                    thinking_end_token="</think>",
                    thinking_start_token="<think>",
                    enable_thinking=True,
                )

        errors.extend(_run_concurrently(build, threads=6))
    finally:
        stop.set()
        for t in holders:
            t.join()

    assert errors == []
