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

import gc
import threading
import time
import weakref

import pytest

from mlx_vlm.prompt_utils import (
    _TOKEN_ENCODE_CACHE,
    THINKING_FORMATS,
    _encode_retrying_on_borrow_error,
    cached_special_token_encode,
    prewarm_special_token_encodes,
)
from mlx_vlm.structured import ThinkingAwareLogitsProcessor
from mlx_vlm.utils import ThinkingBudgetCriteria


@pytest.fixture(autouse=True)
def _clear_encode_cache():
    """Isolate tests from each other's cache entries."""
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


# ── Cache identity: an entry must not outlive the tokenizer it describes ─────
# Keying on `id(tokenizer)` is unsafe on a server that swaps models on demand:
# CPython reuses the address of a collected object almost immediately, so a
# stale entry hands the NEXT model's requests the PREVIOUS model's token ids.
# That is silent -- `ThinkingBudgetCriteria` forces a wrong closing token and
# the budget clip corrupts output instead of raising.


class _VocabTokenizer:
    """A fake whose ids depend on the instance, so a stale hit is visible."""

    def __init__(self, offset):
        self.offset = offset
        self.calls = 0

    def encode(self, text, add_special_tokens=True):
        self.calls += 1
        return [self.offset + len(text)]


def _force_address_reuse(addr, factory, attempts=500):
    """Allocate until one object lands on `addr`, or give up.

    CPython's allocator recycles a freed same-size block on the very next
    allocation ~99% of the time; the loop covers the rest. Objects that miss
    are kept alive so they cannot be recycled into the address themselves.
    """
    misses = []
    for _ in range(attempts):
        candidate = factory()
        if id(candidate) == addr:
            return candidate, misses
        misses.append(candidate)
    return None, misses


def test_cache_does_not_serve_a_recycled_tokenizer_address():
    """A new tokenizer at a recycled address must not inherit cached ids.

    This is the model-swap case: mlx-serve loads models on demand and evicts
    them on an inactivity timeout, so tokenizers really are collected here.
    """
    first = _VocabTokenizer(offset=1000)
    addr = id(first)
    assert cached_special_token_encode(first, "</think>") == [1008]

    del first
    gc.collect()

    second, misses = _force_address_reuse(addr, lambda: _VocabTokenizer(offset=2000))
    if second is None:
        pytest.skip("CPython did not recycle the address in this run")

    # Pre-fix this returns [1008] -- the *previous* model's id for "</think>".
    assert cached_special_token_encode(second, "</think>") == [2008]
    assert second.calls == 1
    del misses


def test_cache_entry_is_dropped_when_its_tokenizer_is_collected():
    """No unbounded growth across model reloads."""
    tokenizer = _VocabTokenizer(offset=1000)
    cached_special_token_encode(tokenizer, "</think>")
    assert len(_TOKEN_ENCODE_CACHE) == 1

    ref = weakref.ref(tokenizer)
    del tokenizer
    gc.collect()

    assert ref() is None
    assert len(_TOKEN_ENCODE_CACHE) == 0


def test_cache_keeps_two_live_tokenizers_distinct():
    """Two models served concurrently must not share entries.

    HF tokenizers use identity equality (verified: neither `TokenizersBackend`
    nor `Qwen2Tokenizer` defines `__eq__`/`__hash__`), so distinct instances
    stay distinct keys even when their vocabularies are identical.
    """
    a = _VocabTokenizer(offset=1000)
    b = _VocabTokenizer(offset=2000)

    assert cached_special_token_encode(a, "</think>") == [1008]
    assert cached_special_token_encode(b, "</think>") == [2008]
    # And both are still served from cache, not re-encoded.
    assert cached_special_token_encode(a, "</think>") == [1008]
    assert cached_special_token_encode(b, "</think>") == [2008]
    assert (a.calls, b.calls) == (1, 1)


def test_cache_tolerates_a_tokenizer_that_cannot_be_weak_referenced():
    """Correctness must not depend on the key being cacheable.

    An object with `__slots__` and no `__weakref__` cannot be a weak key. Such
    a tokenizer simply goes uncached -- it must still get the right ids rather
    than raising, and must never be served another object's entry.
    """

    class _NoWeakref:
        __slots__ = ("offset",)

        def __init__(self, offset):
            self.offset = offset

        def encode(self, text, add_special_tokens=True):
            return [self.offset + len(text)]

    with pytest.raises(TypeError):
        weakref.ref(_NoWeakref(1000))

    tokenizer = _NoWeakref(offset=3000)
    assert cached_special_token_encode(tokenizer, "</think>") == [3008]
    assert cached_special_token_encode(tokenizer, "</think>") == [3008]


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


# ── Instance-level serialization (docs/proposals/tokenizer-concurrency.md) ───
# A fork-only block: tests of `mlx_vlm.tokenizer_lock` against an in-memory
# `PreTrainedTokenizerFast`, nothing from the Hugging Face cache.

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import PreTrainedTokenizerFast, ProcessorMixin  # noqa: E402

try:
    import mlx_vlm.tokenizer_lock as _lock_module  # noqa: E402
    from mlx_vlm.tokenizer_lock import (  # noqa: E402
        is_serialized,
        resolve_fast_tokenizer,
        serialize_fast_tokenizer,
        tokenizer_lock,
    )
except ImportError:  # the baseline: the test must fail by collision, not import
    _lock_module = None

    def serialize_fast_tokenizer(obj, **kwargs):
        return obj

    def is_serialized(obj):
        return False

    def resolve_fast_tokenizer(obj):
        return obj

    _FALLBACK_LOCK = threading.RLock()

    def tokenizer_lock(tok):
        return _FALLBACK_LOCK


_WORDS = [
    "[PAD]",
    "[UNK]",
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
]
_LONG = " ".join(_WORDS[2:] * 3000)


def _make_fast_tokenizer():
    vocab = {w: i for i, w in enumerate(_WORDS)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )


def _stress(tok, iterations=5000, seconds=10):
    """One thread alternates padding, one encodes. Returns (errors, longest call s)."""
    errors, longest = [], [0.0]
    stop = threading.Event()
    barrier = threading.Barrier(2)

    def run(name, call):
        barrier.wait()
        deadline = time.monotonic() + seconds
        for i in range(iterations):
            if stop.is_set() or time.monotonic() > deadline:
                return
            try:
                t0 = time.monotonic()
                call(i)
                longest[0] = max(longest[0], time.monotonic() - t0)
            except BaseException as exc:  # noqa: BLE001
                errors.append(f"{name} {type(exc).__name__}: {exc}")
                stop.set()
                return

    threads = [
        threading.Thread(
            target=run,
            args=("mutator", lambda i: tok([_LONG, _LONG[:50]], padding=bool(i % 2))),
        ),
        threading.Thread(target=run, args=("encoder", lambda i: tok.encode(_LONG))),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors, longest[0]


class _TimingLock:
    """Wraps the registry's RLock; records the longest time spent acquiring it."""

    def __init__(self, real):
        self._real = real
        self.longest = 0.0

    def acquire(self, *args, **kwargs):
        t0 = time.monotonic()
        got = self._real.acquire(*args, **kwargs)
        self.longest = max(self.longest, time.monotonic() - t0)
        return got

    def release(self):
        self._real.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def _is_owned(self):
        return self._real._is_owned()


def test_concurrent_padding_mutation_and_encode_do_not_collide():
    """Criteria 1(a) and 1(i): fails by `Already borrowed` on the baseline.

    1(i) bounds the lock WAIT (the encode itself is allowed to take as long as it
    takes); the whole-call figure is printed for the record only."""
    tok = serialize_fast_tokenizer(_make_fast_tokenizer())
    timing = _TimingLock(tokenizer_lock(tok))
    if _lock_module is not None:
        _lock_module._LOCKS[tok] = timing
    errors, longest = _stress(tok)
    print(f"longest whole call {longest:.3f}s, longest lock wait {timing.longest:.3f}s")
    assert errors == []
    assert timing.longest < 1.0


from mlx_vlm.tokenizer_utils import (  # noqa: E402
    NaiveStreamingDetokenizer,
    TokenizerWrapper,
)

_CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }} {% endfor %}"


class _Processorlike:
    """Holds a tokenizer the way `ProcessorMixin` does, without forwarding."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _BorrowCheckingBackend:
    """Proxies the Rust backend; records every call that touches padding,
    truncation or the encode itself made without the instance lock held."""

    _SETTERS = ("enable_padding", "no_padding", "enable_truncation", "no_truncation")

    def __init__(self, real, tok):
        self._real = real
        self._tok = tok
        self.violations = []
        self.calls = []
        self.hold = None  # (entered, release) events for the next encode_batch
        self.setter_reached = threading.Event()

    def _check(self, name):
        self.calls.append(name)
        if not tokenizer_lock(self._tok)._is_owned():
            self.violations.append(name)

    def encode_batch(self, *args, **kwargs):
        self._check("encode_batch")
        hold, self.hold = self.hold, None
        if hold is not None:
            entered, release = hold
            entered.set()
            assert release.wait(10)
        return self._real.encode_batch(*args, **kwargs)

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if name not in self._SETTERS:
            return attr

        def checked(*args, **kwargs):
            self._check(name)
            self.setter_reached.set()
            return attr(*args, **kwargs)

        return checked


def _proxied():
    tok = serialize_fast_tokenizer(_make_fast_tokenizer())
    tok.chat_template = _CHAT_TEMPLATE
    proxy = _BorrowCheckingBackend(tok._tokenizer, tok)
    tok._tokenizer = proxy
    return tok, proxy


@pytest.mark.parametrize("driver", ["call", "encode", "chat_template"])
def test_every_padding_truncation_and_encode_call_runs_under_the_lock(driver):
    tok, proxy = _proxied()
    batch = ["the quick fox", "lazy dog"]
    messages = [{"role": "user", "content": "the quick fox"}]
    for padding, truncation in [(1, 0), (0, 1), (0, 0), (1, 1)]:
        kwargs = dict(padding=bool(padding), truncation=bool(truncation), max_length=3)
        if driver == "call":
            tok(batch, **kwargs)
        elif driver == "encode":
            tok.encode("the quick fox", **kwargs)
        else:
            tok.apply_chat_template(messages, tokenize=True, **kwargs)
    assert "encode_batch" in proxy.calls
    assert {"enable_padding", "no_padding"} <= set(proxy.calls)
    assert {"enable_truncation", "no_truncation"} <= set(proxy.calls)
    assert proxy.violations == []


def test_a_settings_change_waits_for_an_encode_in_flight():
    tok, proxy = _proxied()
    entered, release = threading.Event(), threading.Event()
    proxy.hold = (entered, release)
    holder = threading.Thread(target=lambda: tok(["the quick fox", "lazy dog"]))
    holder.start()
    assert entered.wait(10)
    proxy.setter_reached.clear()
    changer = threading.Thread(
        target=lambda: tok(["the quick fox", "lazy dog"], padding=True)
    )
    changer.start()
    assert not proxy.setter_reached.wait(0.3), "setter ran during an encode"
    release.set()
    holder.join(10)
    changer.join(10)
    assert proxy.setter_reached.is_set()
    assert proxy.violations == []


def test_one_lock_per_tokenizer_however_it_is_reached():
    tok = _make_fast_tokenizer()
    wrapper = TokenizerWrapper(tok)
    processor = _Processorlike(tok)
    # Resolving through the wrapper first, on a fresh tokenizer.
    first = tokenizer_lock(resolve_fast_tokenizer(wrapper))
    assert first is tokenizer_lock(resolve_fast_tokenizer(processor))
    assert first is tokenizer_lock(resolve_fast_tokenizer(tok))
    assert first is tokenizer_lock(resolve_fast_tokenizer(TokenizerWrapper(tok)))
    assert tokenizer_lock(_make_fast_tokenizer()) is not first


def test_serializing_twice_leaves_the_bound_functions_alone_and_skips_wrappers():
    tok = _make_fast_tokenizer()
    wrapper = TokenizerWrapper(tok)
    assert serialize_fast_tokenizer(wrapper) is wrapper
    assert is_serialized(wrapper) and is_serialized(tok)
    before = (vars(tok)["_encode_plus"], vars(tok)["set_truncation_and_padding"])
    serialize_fast_tokenizer(tok)
    serialize_fast_tokenizer(wrapper)
    after = (vars(tok)["_encode_plus"], vars(tok)["set_truncation_and_padding"])
    assert all(a is b for a, b in zip(before, after))
    assert "_encode_plus" not in vars(wrapper)
    assert "_mlx_vlm_serialized" not in vars(wrapper)


def test_nothing_resolves_means_nothing_changes():
    obj = object()
    assert serialize_fast_tokenizer(obj) is obj
    assert not is_serialized(obj)
    assert resolve_fast_tokenizer(_Processorlike(object())) is None


def test_outputs_are_identical_before_and_after_serialization():
    prompts = [_LONG[:400], "the quick fox", "lazy dog over the"]
    variants = [
        dict(padding=True),
        dict(padding=False),
        dict(padding=True, truncation=True, max_length=5),
        dict(padding="max_length", truncation=True, max_length=8),
    ]
    plain = _make_fast_tokenizer()
    guarded = serialize_fast_tokenizer(_make_fast_tokenizer())
    expected = [plain(prompts, **kw).data for kw in variants]
    assert [guarded(prompts, **kw).data for kw in variants] == expected

    results, errors = [], []

    def worker():
        try:
            results.append([guarded(prompts, **kw).data for kw in variants])
        except BaseException as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert results == [expected] * 6


def _write_tokenizer_dir(path):
    import json

    path.mkdir(parents=True, exist_ok=True)
    _make_fast_tokenizer().backend_tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
            }
        )
    )
    (path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    return path


def test_loaders_return_serialized_tokenizers(tmp_path):
    """`gpt2` has no processor, so `AutoProcessor` falls back to the tokenizer."""
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import load_processor

    path = _write_tokenizer_dir(tmp_path / "model")
    wrapper = load_tokenizer(path)
    assert isinstance(wrapper, TokenizerWrapper)
    assert is_serialized(wrapper._tokenizer)
    processor = load_processor(path, add_detokenizer=False)
    assert is_serialized(processor)


def test_retry_logs_one_warning_per_retried_attempt_without_the_prompt(caplog):
    import logging

    prompt = "SECRET-PROMPT-TEXT"
    with caplog.at_level(logging.WARNING, logger="mlx_vlm.prompt_utils"):
        _encode_retrying_on_borrow_error(_FlakyTokenizer(3), prompt, backoff=0)
    records = [r for r in caplog.records if r.name == "mlx_vlm.prompt_utils"]
    assert len(records) == 3
    assert all(prompt not in r.getMessage() for r in records)
    assert all("_FlakyTokenizer" in r.getMessage() for r in records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mlx_vlm.prompt_utils"):
        _encode_retrying_on_borrow_error(_FlakyTokenizer(0), prompt, backoff=0)
    assert caplog.records == []


def test_installed_fast_tokenizer_exposes_the_names_the_lock_rebinds():
    from mlx_vlm.tokenizer_lock import _REQUIRED

    for name in _REQUIRED:
        assert callable(getattr(PreTrainedTokenizerFast, name, None)), name


def test_a_class_missing_a_required_name_logs_one_error_and_is_unchanged(caplog):
    import logging

    class _Future(PreTrainedTokenizerFast):
        pass

    tok = _Future(tokenizer_object=_make_fast_tokenizer().backend_tokenizer)
    _Future.set_truncation_and_padding = None  # a future version without the name
    with caplog.at_level(logging.ERROR):
        assert serialize_fast_tokenizer(tok) is tok
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "set_truncation_and_padding" in errors[0].getMessage()
    assert "tokenizer-concurrency" in errors[0].getMessage()
    assert not is_serialized(tok)
    assert "_encode_plus" not in vars(tok)


def test_a_serialized_tokenizer_survives_copy_pickle_and_save(tmp_path):
    import copy
    import pickle

    tok = serialize_fast_tokenizer(_make_fast_tokenizer())
    clone = copy.deepcopy(tok)
    assert is_serialized(clone)
    assert vars(clone)["_encode_plus"].tok is clone
    assert tokenizer_lock(clone) is not tokenizer_lock(tok)
    assert clone(["the fox"]).data == tok(["the fox"]).data
    restored = pickle.loads(pickle.dumps(tok))
    # Both routes re-serialize on restore: no manual call is needed.
    assert is_serialized(restored)
    assert vars(restored)["_encode_plus"].tok is restored
    assert restored(["the fox"]).data == tok(["the fox"]).data
    errors, _longest = _stress(restored, iterations=1500, seconds=5)
    assert errors == []
    errors, _longest = _stress(clone, iterations=1500, seconds=5)
    assert errors == []

    holder = serialize_fast_tokenizer(_ProcessorWithoutDetokenizer(tok))
    assert is_serialized(holder)
    holder.save_pretrained(str(tmp_path / "saved"))
    assert (tmp_path / "saved" / "tokenizer.json").is_file()
    assert (tmp_path / "saved" / "processor_config.json").is_file()


class _ProcessorWithoutDetokenizer(ProcessorMixin):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)


def _directory_bytes(path):
    return {
        str(p.relative_to(path)): p.read_bytes()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def test_saving_a_serialized_object_writes_the_same_bytes_as_an_unserialized_one(
    tmp_path,
):
    """Criterion 1(k): serialization leaves no trace in what ``save_pretrained`` writes."""
    from transformers import AutoProcessor

    from mlx_vlm.utils import load_processor

    pairs = {}
    pairs["tokenizer"] = (
        serialize_fast_tokenizer(_make_fast_tokenizer()),
        _make_fast_tokenizer(),
    )
    path = _write_tokenizer_dir(tmp_path / "model")
    pairs["fallback"] = (
        load_processor(path, add_detokenizer=False),
        AutoProcessor.from_pretrained(path),
    )
    for label, (serialized, plain) in pairs.items():
        assert is_serialized(serialized), label
        assert not is_serialized(plain), label
        serialized.save_pretrained(tmp_path / f"{label}-serialized")
        plain.save_pretrained(tmp_path / f"{label}-plain")
        written = _directory_bytes(tmp_path / f"{label}-serialized")
        assert written, label
        assert written == _directory_bytes(tmp_path / f"{label}-plain"), label


def test_a_class_that_stores_attributes_elsewhere_logs_one_error_and_is_unchanged(
    caplog,
):
    import logging

    class _Elsewhere(PreTrainedTokenizerFast):
        def __setattr__(self, key, value):
            if key in (
                "_encode_plus",
                "set_truncation_and_padding",
                "_mlx_vlm_serialized",
            ):
                self.__dict__.setdefault("_elsewhere", {})[key] = value
                return
            super().__setattr__(key, value)

    tok = _Elsewhere(tokenizer_object=_make_fast_tokenizer().backend_tokenizer)
    before = dict(vars(tok))
    with caplog.at_level(logging.ERROR):
        assert serialize_fast_tokenizer(tok) is tok
    assert not is_serialized(tok)
    for name in ("_encode_plus", "set_truncation_and_padding", "_mlx_vlm_serialized"):
        assert name not in vars(tok), name
    assert {k: v for k, v in vars(tok).items() if k != "_elsewhere"} == before
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "_Elsewhere" in errors[0].getMessage()
    assert "tokenizer-concurrency" in errors[0].getMessage()


def test_a_refused_second_assignment_leaves_the_instance_as_it_was(caplog):
    import logging

    class _Refusing(PreTrainedTokenizerFast):
        def __setattr__(self, key, value):
            if key == "set_truncation_and_padding":
                raise AttributeError("SECRET-EXCEPTION-TEXT")
            super().__setattr__(key, value)

    tok = _Refusing(tokenizer_object=_make_fast_tokenizer().backend_tokenizer)
    vars(tok)["_encode_plus"] = prior = object()  # a prior attribute is put back
    before = dict(vars(tok))
    with caplog.at_level(logging.ERROR):
        assert serialize_fast_tokenizer(tok) is tok
    assert vars(tok) == before
    assert vars(tok)["_encode_plus"] is prior
    assert not is_serialized(tok)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    message = errors[0].getMessage()
    assert "_Refusing" in message
    assert "set_truncation_and_padding" in message
    assert "AttributeError" in message
    assert "SECRET-EXCEPTION-TEXT" not in message


def test_loaders_with_the_default_detokenizer_return_serialized_tokenizers(tmp_path):
    from mlx_vlm.utils import load_processor

    path = _write_tokenizer_dir(tmp_path / "model")
    processor = load_processor(path)
    assert is_serialized(processor)
    assert hasattr(processor, "detokenizer")


def test_a_processormixin_around_the_tokenizer_is_serialized(tmp_path):
    import json

    from transformers import AutoProcessor
    from transformers.processing_utils import ProcessorMixin

    from mlx_vlm.utils import load_processor

    path = _write_tokenizer_dir(tmp_path / "model")
    (path / "processor_config.json").write_text(
        json.dumps({"processor_class": "CLIPProcessor"})
    )
    (path / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "CLIPImageProcessor"})
    )
    try:
        processor = AutoProcessor.from_pretrained(path)
        if not isinstance(processor, ProcessorMixin):
            raise TypeError(type(processor).__name__)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no ProcessorMixin constructible offline: {type(exc).__name__}")
    assert is_serialized(load_processor(path))


def test_a_shallow_copy_is_unsupported_and_stays_bound_to_the_original():
    import copy

    tok = serialize_fast_tokenizer(_make_fast_tokenizer())
    clone = copy.copy(tok)
    assert vars(clone)["_encode_plus"].tok is tok
    serialize_fast_tokenizer(clone)
    assert vars(clone)["_encode_plus"].tok is tok
    assert clone._tokenizer is tok._tokenizer  # one shared Rust backend
    assert is_serialized(tok)
