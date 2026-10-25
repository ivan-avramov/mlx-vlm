"""Real Rust-backend reproductions plus event-controlled critical-section probes."""

import threading
import time
from types import SimpleNamespace

import pytest
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from mlx_vlm.utils import prepare_inputs

from .conftest import BATCH, LONG, USING_FALLBACK, run_workers, tokenizer_lock


def plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    return value


def invoke(tok, entry, *, long=False):
    text = LONG if long else "alpha beta gamma"
    if entry == "call":
        return dict(tok(text, add_special_tokens=False))
    if entry == "encode":
        return tok.encode(text, add_special_tokens=False)
    if entry == "tokenize":
        return tok.tokenize(text, add_special_tokens=False)
    if entry == "chat":
        return tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True
        )
    if entry == "prepare_inputs":
        return plain(
            prepare_inputs(
                SimpleNamespace(tokenizer=tok),
                prompts=[text],
                padding=False,
                return_tensors="np",
            )
        )
    if entry == "setter":
        return tok.set_truncation_and_padding(
            PaddingStrategy.DO_NOT_PAD,
            TruncationStrategy.DO_NOT_TRUNCATE,
            None,
            0,
            None,
            None,
        )
    raise AssertionError(entry)


@pytest.mark.parametrize("victim", ["encode", "prepare_inputs", "chat"])
def test_alternating_batch_against_public_encode_paths(
    tok, serialization, victim, record_property, lock_waits
):
    """Reproduces: padded/unpadded Rust batch encodes race each named victim.

    Every result is compared to an uncontended result, including masks and
    truncation. The encode pair runs to 5000 calls/thread or ten seconds;
    the additional pairs have two-second budgets to keep the directory <30s.
    """
    expected_batches = {
        pad: dict(tok(BATCH, padding=pad, add_special_tokens=False))
        for pad in (False, True)
    }
    expected_truncated = dict(tok(BATCH, padding=True, truncation=True, max_length=127))
    expected_victim = invoke(tok, victim, long=True)
    serialization(tok)

    def mutator(i):
        pad = bool(i % 2)
        actual = dict(tok(BATCH, padding=pad, add_special_tokens=False))
        assert actual == expected_batches[pad]
        if i % 11 == 0:
            assert (
                dict(tok(BATCH, padding=True, truncation=True, max_length=127))
                == expected_truncated
            )

    def reader(i):
        assert invoke(tok, victim, long=True) == expected_victim

    run_workers([mutator, reader], seconds=10 if victim == "encode" else 2)
    assert lock_waits, "no instance-lock acquisitions observed"
    record_property("longest_lock_wait_seconds", max(lock_waits))
    assert max(lock_waits) < 1, f"longest lock wait: {max(lock_waits):.6f}s"


class BackendProbe:
    """Delegate to Rust while observing lock ownership and an open encode."""

    def __init__(self, tok):
        self.real = tok._tokenizer
        self.tok = tok
        self.entered = threading.Event()
        self.release = threading.Event()
        self.contender_touched = threading.Event()
        self.calls = set()

    def __getattr__(self, name):
        return getattr(self.real, name)

    def checked(self, name, *args, **kwargs):
        self.calls.add(name)
        if threading.current_thread().name == "settings-contender":
            self.contender_touched.set()
        assert tokenizer_lock(
            self.tok
        )._is_owned(), f"{name} outside instance lock; fallback={USING_FALLBACK}"
        return getattr(self.real, name)(*args, **kwargs)

    def encode_batch(self, *args, **kwargs):
        if threading.current_thread().name == "encode-holder":
            self.entered.set()
            assert self.release.wait(1), "encode was never released"
        return self.checked("encode_batch", *args, **kwargs)

    def enable_padding(self, *args, **kwargs):
        return self.checked("enable_padding", *args, **kwargs)

    def no_padding(self, *args, **kwargs):
        return self.checked("no_padding", *args, **kwargs)

    def enable_truncation(self, *args, **kwargs):
        return self.checked("enable_truncation", *args, **kwargs)

    def no_truncation(self, *args, **kwargs):
        return self.checked("no_truncation", *args, **kwargs)


@pytest.mark.parametrize(
    "entry", ["call", "encode", "tokenize", "chat", "prepare_inputs", "setter"]
)
def test_settings_and_encode_are_one_reentrant_critical_section(
    tok, serialization, entry
):
    """Reproduces: every public path must lock settings together with encoding.

    The holder starts padded/truncated; the contender must not reach a backend
    mutator until that encode finishes. All five backend methods assert ownership.
    """
    serialization(tok)
    probe = BackendProbe(tok)
    tok._tokenizer = probe
    barrier = threading.Barrier(2)
    stop = threading.Event()
    errors = []
    attempted = threading.Event()

    def holder():
        try:
            barrier.wait(timeout=1)
            tok(BATCH, padding=True, truncation=True, max_length=32)
        except BaseException as exc:
            errors.append(("holder", type(exc).__name__, str(exc)))
            stop.set()

    def contender():
        try:
            barrier.wait(timeout=1)
            while not probe.entered.wait(0.005):
                if stop.is_set():
                    return
            attempted.set()
            if not stop.is_set():
                invoke(tok, entry)
        except BaseException as exc:
            errors.append(("contender", type(exc).__name__, str(exc)))
            stop.set()

    workers = [
        threading.Thread(target=holder, name="encode-holder", daemon=True),
        threading.Thread(target=contender, name="settings-contender", daemon=True),
    ]
    for worker in workers:
        worker.start()
    try:
        deadline = time.monotonic() + 1
        while not attempted.wait(0.005) and not stop.is_set():
            assert time.monotonic() < deadline, "holder never reached encode_batch"
        if not errors:
            assert attempted.is_set()
            assert not probe.contender_touched.wait(
                0.04
            ), "settings interleaved with encode"
    finally:
        probe.release.set()
        for worker in workers:
            worker.join(timeout=1)
        stop.set()
    assert not any(
        worker.is_alive() for worker in workers
    ), "deadlocked critical section"
    assert not errors, f"fallback={USING_FALLBACK}; thread exceptions={errors}"
    assert probe.contender_touched.is_set()
    assert probe.calls == {
        "encode_batch",
        "enable_padding",
        "no_padding",
        "enable_truncation",
        "no_truncation",
    }


@pytest.mark.parametrize(
    "padding,truncation", [(False, False), (True, False), (True, True)]
)
def test_serialization_preserves_uncontended_outputs(
    tok, serialization, padding, truncation
):
    """Preserves: ids and attention masks are unchanged for all encode strategies."""
    kwargs = dict(padding=padding, truncation=truncation)
    if truncation:
        kwargs["max_length"] = 7
    before = dict(tok(["alpha beta gamma delta", "beta"], **kwargs))
    serialization(tok)
    assert dict(tok(["alpha beta gamma delta", "beta"], **kwargs)) == before
