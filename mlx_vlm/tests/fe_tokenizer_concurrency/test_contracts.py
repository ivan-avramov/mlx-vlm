"""Object identity, lifetime, failure, and serialization contracts."""

import copy
import gc
import logging
import pickle
import weakref
from types import SimpleNamespace

import pytest
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer, TokenizerWrapper

from .conftest import make_tokenizer, run_workers

REQUIRED = ("_encode_plus", "set_truncation_and_padding")


class SlowTokenizer(PreTrainedTokenizer):
    vocab = {"[UNK]": 0, "alpha": 1}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, 0)

    def _convert_id_to_token(self, index):
        return {v: k for k, v in self.vocab.items()}.get(index, "[UNK]")


class RefusesBinding(PreTrainedTokenizerFast):
    def __setattr__(self, name, value):
        if vars(self).get("refuse_binding") and name in (
            *REQUIRED,
            "_mlx_vlm_serialized",
        ):
            raise AttributeError(f"refuses {name}")
        super().__setattr__(name, value)


class SyntheticProcessor(ProcessorMixin):
    # Real ProcessorMixin save_pretrained/to_dict traverses the detokenizer while
    # excluding the direct tokenizer, reproducing the copy-safety boundary.
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer)


def test_wrapper_first_identity_and_distinct_tokenizers(api, tok):
    """New behaviour: wrapper-first lookup, processor and tokenizer share one lock."""
    wrapper = TokenizerWrapper(tok)
    processor = SimpleNamespace(tokenizer=tok)
    lock = api.tokenizer_lock(api.resolve_fast_tokenizer(wrapper))
    assert api.resolve_fast_tokenizer(wrapper) is tok
    assert api.resolve_fast_tokenizer(processor) is tok
    assert lock is api.tokenizer_lock(api.resolve_fast_tokenizer(processor))
    assert lock is api.tokenizer_lock(tok)
    assert lock is not api.tokenizer_lock(make_tokenizer())
    api.serialize_fast_tokenizer(wrapper)
    assert set(vars(wrapper)) == {"_tokenizer", "_detokenizer"}
    assert api.is_serialized(wrapper) and api.is_serialized(processor)


def test_concurrent_first_lookup_returns_one_lock(api, tok):
    """New behaviour: simultaneous first lookups cannot allocate different locks."""
    locks = []
    run_workers(
        [lambda i: locks.append(api.tokenizer_lock(tok)) for _ in range(8)],
        seconds=1,
        iterations=1,
    )
    assert len(locks) == 8
    assert all(lock is locks[0] for lock in locks)


def test_idempotence_and_truthful_serialization_mark(api, tok):
    """New behaviour: binding is idempotent and a mark alone is insufficient."""
    assert not api.is_serialized(tok)
    tok._mlx_vlm_serialized = True
    assert not api.is_serialized(tok), "a mark without both bindings is not serialization"
    del tok._mlx_vlm_serialized
    assert api.serialize_fast_tokenizer(tok) is tok
    assert api.is_serialized(tok)
    bound = {name: vars(tok)[name] for name in REQUIRED}
    assert api.serialize_fast_tokenizer(tok) is tok
    for name, method in bound.items():
        assert vars(tok)[name] is method
        assert callable(method)
    tok.padding_side = "left"
    assert tok(["alpha beta", "alpha"], padding=True)["input_ids"] == [
        [2, 3],
        [1, 2],
    ]
    delattr(tok, REQUIRED[0])
    assert not api.is_serialized(tok)


def test_resolution_uses_instance_fields_not_forwarding_attributes(api, tok):
    """New behaviour: resolution is bounded, cycle-safe, and never uses getattr."""
    accesses = []

    class Forwarder:
        def __init__(self, inner):
            self._tokenizer = inner

        def __getattr__(self, name):
            accesses.append(name)
            return getattr(self._tokenizer, name)

    wrapper = Forwarder(tok)
    assert api.resolve_fast_tokenizer(wrapper) is tok
    assert api.serialize_fast_tokenizer(wrapper) is wrapper
    assert accesses == []
    assert set(vars(wrapper)) == {"_tokenizer"}
    assert (
        api.resolve_fast_tokenizer(
            SimpleNamespace(tokenizer=tok, _tokenizer=make_tokenizer())
        )
        is tok
    )
    cycle = SimpleNamespace()
    cycle.tokenizer = cycle
    assert api.resolve_fast_tokenizer(cycle) is None
    nested = tok
    for _ in range(6):
        nested = SimpleNamespace(tokenizer=nested)
    assert api.resolve_fast_tokenizer(nested) is None
    assert api.resolve_fast_tokenizer(object()) is None


def test_slow_and_tokenizer_free_objects_are_unchanged(serialization, caplog):
    """Preserves: genuine slow tokenizers and tokenizer-free objects are untouched."""
    slow = SlowTokenizer(unk_token="[UNK]")
    expected = slow.encode("alpha unknown")
    objects = [slow, SimpleNamespace(tokenizer=slow), SimpleNamespace(), object()]
    for obj in objects:
        before = dict(vars(obj)) if hasattr(obj, "__dict__") else None
        assert serialization(obj) is obj
        if before is not None:
            assert vars(obj) == before
    assert slow.encode("alpha unknown") == expected
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param(
            "_encode_plus",
            marks=pytest.mark.xfail(
                strict=True,
                reason="deleting the fast class's override exposes PreTrainedTokenizerBase._encode_plus, so the absence is not simulated",
            ),
        ),
        "set_truncation_and_padding",
    ],
)
def test_missing_required_class_method_logs_one_error(
    api, tok, monkeypatch, caplog, missing
):
    """New behaviour: dependency method removal is loud and returns the same object."""
    monkeypatch.delattr(PreTrainedTokenizerFast, missing)
    with caplog.at_level(logging.ERROR):
        assert api.serialize_fast_tokenizer(tok) is tok
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert missing in errors[0].getMessage()
    assert type(tok).__name__ in errors[0].getMessage()
    assert "tokenizer-concurrency" in errors[0].getMessage()
    assert not api.is_serialized(tok)


def test_attribute_refusal_is_logged_once(api, caplog):
    """New behaviour: an attribute-refusing fast tokenizer cannot fail silently."""
    tok = make_tokenizer(RefusesBinding)
    tok.refuse_binding = True
    with caplog.at_level(logging.ERROR):
        assert api.serialize_fast_tokenizer(tok) is tok
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    message = errors[0].getMessage()
    assert "RefusesBinding" in message
    assert any(name in message for name in REQUIRED)
    assert "tokenizer-concurrency" in message
    assert not api.is_serialized(tok)


def test_installed_fast_tokenizer_exposes_required_methods():
    """Preserves: the installed dependency exposes the specified interception points."""
    assert all(
        callable(getattr(PreTrainedTokenizerFast, name, None)) for name in REQUIRED
    )


def test_lock_is_reentrant_and_exception_releases_it(api, tok):
    """New behaviour: recursive encode works and Rust exceptions release the lock."""
    api.serialize_fast_tokenizer(tok)
    lock = api.tokenizer_lock(tok)

    def recursive(i):
        with lock:
            with lock:
                assert tok.encode("alpha beta") == [2, 3]

    run_workers([recursive], seconds=1, iterations=1)
    real = tok._tokenizer
    failure = RuntimeError("synthetic backend failure")

    class BrokenBackend:
        def __getattr__(self, name):
            return getattr(real, name)

        def encode_batch(self, *args, **kwargs):
            assert lock._is_owned()
            raise failure

    tok._tokenizer = BrokenBackend()
    with pytest.raises(RuntimeError) as caught:
        tok.encode("alpha beta")
    assert caught.value is failure
    tok._tokenizer = real
    # A new thread must acquire, rather than the owner merely re-entering.
    run_workers([recursive], seconds=1, iterations=1)


def test_decode_id_lookups_and_detokenizers_are_not_rebound(tok, serialization):
    """Preserves: decode, id lookup, add-tokens and detokenizer methods stay free."""
    wrapper = TokenizerWrapper(tok)
    detokenizer = wrapper.detokenizer
    names = (
        "decode",
        "_decode",
        "batch_decode",
        "convert_tokens_to_ids",
        "convert_ids_to_tokens",
        "_add_tokens",
    )
    methods = {name: getattr(tok, name).__func__ for name in names}
    det_methods = {
        name: getattr(detokenizer, name).__func__
        for name in ("add_token", "finalize", "reset")
    }
    before = dict(vars(tok))
    expected_decode = tok.decode([2, 3, 4])
    serialization(wrapper)
    for name, function in methods.items():
        assert getattr(tok, name).__func__ is function
        assert vars(tok).get(name) is before.get(name)
    assert wrapper.detokenizer is detokenizer
    for name, function in det_methods.items():
        assert getattr(detokenizer, name).__func__ is function
        assert name not in vars(detokenizer)
    assert tok.decode([2, 3, 4]) == expected_decode
    assert tok.batch_decode([[2, 3, 4]]) == [expected_decode]
    for token in (2, 3, 4):
        detokenizer.add_token(token)
    detokenizer.finalize()
    assert detokenizer.text == expected_decode
    added = set(vars(tok)) - set(before)
    assert added <= {*REQUIRED, "_mlx_vlm_serialized"}


@pytest.mark.parametrize("operation", ["deepcopy", "pickle"])
def test_tokenizer_copy_remains_usable(tok, serialization, operation, api):
    """Preserves: restored tokenizers stay serialized and honour their own settings."""
    serialization(tok)
    clone = (
        copy.deepcopy(tok)
        if operation == "deepcopy"
        else pickle.loads(pickle.dumps(tok))
    )
    assert clone is not tok
    assert api.is_serialized(clone)
    clone.padding_side = "left"
    assert tok.padding_side == "right"
    assert clone(["alpha beta", "alpha"], padding=True)["input_ids"] == [
        [2, 3],
        [1, 2],
    ]
    assert tok(["alpha beta", "alpha"], padding=True)["input_ids"] == [
        [2, 3],
        [2, 1],
    ]
    assert clone.encode("alpha beta") == tok.encode("alpha beta")


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            "deepcopy",
            marks=pytest.mark.xfail(
                strict=True,
                reason="pre-existing: NaiveStreamingDetokenizer cannot be deep-copied or unpickled (read-only text property); outside the tokenizer-concurrency change",
            ),
        ),
        pytest.param(
            "pickle",
            marks=pytest.mark.xfail(
                strict=True,
                reason="pre-existing: NaiveStreamingDetokenizer cannot be deep-copied or unpickled (read-only text property); outside the tokenizer-concurrency change",
            ),
        ),
    ],
)
def test_processor_copy_remains_usable(tok, serialization, operation):
    """Preserves: processor copying must traverse a retained detokenizer safely.

    This invariant already fails on baseline because NaiveStreamingDetokenizer's
    inherited text slot conflicts with its read-only text property.
    """
    processor = SyntheticProcessor(tok)
    serialization(processor)
    clone = (
        copy.deepcopy(processor)
        if operation == "deepcopy"
        else pickle.loads(pickle.dumps(processor))
    )
    assert clone.tokenizer is not tok
    assert clone.detokenizer._tokenizer is clone.tokenizer
    for name in REQUIRED:
        assert vars(clone.tokenizer)[name].tok is clone.tokenizer
    assert clone.tokenizer.encode("alpha beta") == [2, 3]


def test_tokenizer_save_remains_usable(tok, serialization, tmp_path):
    """Preserves: tokenizer save_pretrained succeeds and retains the token ids."""
    serialization(tok)
    tok.save_pretrained(tmp_path)
    restored = PreTrainedTokenizerFast.from_pretrained(tmp_path, local_files_only=True)
    assert restored.encode("alpha beta") == tok.encode("alpha beta")


@pytest.mark.xfail(
    strict=True,
    reason="pre-existing: NaiveStreamingDetokenizer cannot be deep-copied or unpickled (read-only text property); outside the tokenizer-concurrency change",
)
def test_processor_save_with_detokenizer_remains_usable(tok, serialization, tmp_path):
    """Preserves: real ProcessorMixin.save_pretrained traverses the detokenizer.

    Baseline already fails in ProcessorMixin.to_dict's deepcopy of that member.
    """
    processor = SyntheticProcessor(tok)
    serialization(processor)
    processor.save_pretrained(tmp_path)
    assert (tmp_path / "tokenizer.json").is_file()
    assert (tmp_path / "processor_config.json").is_file()


def test_copy_uses_a_distinct_lock_and_weak_registry_does_not_retain(api, tok):
    """New behaviour: copies get independent locks and the registry releases dead keys."""
    api.serialize_fast_tokenizer(tok)
    clone = copy.deepcopy(tok)
    assert api.is_serialized(clone)
    assert api.tokenizer_lock(clone) is not api.tokenizer_lock(tok)
    clone.padding_side = "left"
    assert tok.padding_side == "right"
    assert clone(["alpha beta", "alpha"], padding=True)["input_ids"] == [
        [2, 3],
        [1, 2],
    ]
    ref = weakref.ref(clone)
    lock = api.tokenizer_lock(clone)
    del clone
    gc.collect()
    assert ref() is None
    assert lock.acquire(timeout=0.1)
    lock.release()
