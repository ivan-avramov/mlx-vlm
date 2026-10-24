"""Synthetic fixtures; never consult a model repository or HF cache."""

import importlib
import threading
import time

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

try:
    from mlx_vlm.tokenizer_lock import serialize_fast_tokenizer, tokenizer_lock

    USING_FALLBACK = False
except ImportError:
    USING_FALLBACK = True

    def serialize_fast_tokenizer(obj, **kwargs):
        return obj

    def tokenizer_lock(tok):
        return threading.RLock()


CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
LONG = "alpha beta gamma delta " * 1200
BATCH = [LONG, "alpha beta " * 900]


def make_tokenizer(cls=PreTrainedTokenizerFast):
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3, "gamma": 4, "delta": 5},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return cls(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        chat_template=CHAT_TEMPLATE,
        model_max_length=100000,
    )


@pytest.fixture
def tok():
    return make_tokenizer()


@pytest.fixture
def api():
    # New-interface tests intentionally report the absent module on the baseline.
    return importlib.import_module("mlx_vlm.tokenizer_lock")


@pytest.fixture
def serialization(record_property):
    record_property("tokenizer_lock_fallback", USING_FALLBACK)
    return serialize_fast_tokenizer


@pytest.fixture
def lock_waits(monkeypatch):
    waits = []
    if not USING_FALLBACK:
        module = importlib.import_module("mlx_vlm.tokenizer_lock")
        original = module.tokenizer_lock

        class MeasuredLock:
            def __init__(self, lock):
                self.lock = lock

            def __enter__(self):
                started = time.monotonic()
                self.lock.acquire()
                waits.append(time.monotonic() - started)
                return self

            def __exit__(self, *exc):
                self.lock.release()

            def __getattr__(self, name):
                return getattr(self.lock, name)

        # Observe the real method critical sections without adding an outer lock
        # that could make an unprotected implementation appear safe.
        monkeypatch.setattr(
            module, "tokenizer_lock", lambda tok: MeasuredLock(original(tok))
        )
    return waits


def run_workers(actions, *, seconds, iterations=5000):
    """Collect *every* worker exception, including barrier failures, and stop peers."""
    barrier = threading.Barrier(len(actions))
    stop = threading.Event()
    errors = []
    counts = [0] * len(actions)
    deadline = time.monotonic() + seconds

    def worker(index, action):
        try:
            barrier.wait(timeout=2)
            for i in range(iterations):
                if stop.is_set() or time.monotonic() >= deadline:
                    break
                action(i)
                counts[index] += 1
        except BaseException as exc:
            errors.append((index, type(exc).__name__, str(exc)))
            stop.set()

    threads = [
        threading.Thread(target=worker, args=(i, action), daemon=True)
        for i, action in enumerate(actions)
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join(timeout=max(0, deadline - time.monotonic()) + 2)
    finally:
        stop.set()
    assert not any(thread.is_alive() for thread in threads), "worker did not stop"
    assert not errors, f"fallback={USING_FALLBACK}; thread exceptions={errors}"
    assert all(counts), f"a worker performed no calls: {counts}"
    return counts
