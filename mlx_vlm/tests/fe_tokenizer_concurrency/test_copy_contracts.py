"""Draft-5 restoration and unsupported shallow-copy contracts."""

import copy
import pickle

import pytest
from transformers import ProcessorMixin

from .conftest import BATCH, LONG, run_workers


class ProcessorWithoutDetokenizer(ProcessorMixin):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)


@pytest.mark.parametrize("operation", ["deepcopy", "pickle"])
def test_holder_without_detokenizer_copy_and_save(api, tok, tmp_path, operation):
    """New behaviour: a real holder restores protection and saves without a detokenizer."""
    holder = ProcessorWithoutDetokenizer(tok)
    api.serialize_fast_tokenizer(holder)
    clone = (
        copy.deepcopy(holder)
        if operation == "deepcopy"
        else pickle.loads(pickle.dumps(holder))
    )
    assert api.is_serialized(clone)
    assert clone.tokenizer is not tok
    assert api.tokenizer_lock(clone.tokenizer) is not api.tokenizer_lock(tok)
    clone.tokenizer.padding_side = "left"
    assert tok.padding_side == "right"
    assert clone.tokenizer(["alpha beta", "alpha"], padding=True)["input_ids"] == [
        [2, 3],
        [1, 2],
    ]
    clone.save_pretrained(tmp_path)
    assert (tmp_path / "tokenizer.json").is_file()
    assert (tmp_path / "processor_config.json").is_file()


def test_restored_tokenizers_stay_serialized_under_contention(
    api, tok, lock_waits, record_property
):
    """New behaviour: deep-copy and pickle restorations pass the full ten-second stress.

    Both pairs run concurrently to preserve the directory's thirty-second budget.
    No explicit serialization call is made on either restored object.
    """
    api.serialize_fast_tokenizer(tok)
    clones = [copy.deepcopy(tok), pickle.loads(pickle.dumps(tok))]
    actions = []
    for clone in clones:
        assert api.is_serialized(clone)
        assert api.tokenizer_lock(clone) is not api.tokenizer_lock(tok)
        expected = {
            pad: dict(clone(BATCH, padding=pad, add_special_tokens=False))
            for pad in (False, True)
        }
        expected_ids = clone.encode(LONG, add_special_tokens=False)

        def mutate(i, clone=clone, expected=expected):
            pad = bool(i % 2)
            assert (
                dict(clone(BATCH, padding=pad, add_special_tokens=False))
                == expected[pad]
            )

        def read(i, clone=clone, expected_ids=expected_ids):
            assert clone.encode(LONG, add_special_tokens=False) == expected_ids

        actions.extend([mutate, read])
    assert api.tokenizer_lock(clones[0]) is not api.tokenizer_lock(clones[1])
    run_workers(actions, seconds=10, iterations=5000)
    assert lock_waits
    record_property("restored_longest_lock_wait_seconds", max(lock_waits))
    assert max(lock_waits) < 1


def test_shallow_copy_is_explicitly_unsupported(api, tok):
    """New behaviour: shallow-copy calls keep using the original, even after serialization."""
    api.serialize_fast_tokenizer(tok)
    clone = copy.copy(tok)
    names = ("_encode_plus", "set_truncation_and_padding")
    stored = {name: vars(clone)[name] for name in names}
    assert all(stored[name] is vars(tok)[name] for name in names)
    assert clone.backend_tokenizer is tok.backend_tokenizer
    assert api.is_serialized(clone)
    clone.padding_side = "left"
    assert tok.padding_side == "right"
    batch = ["alpha beta", "alpha"]
    assert clone(batch, padding=True)["input_ids"] == [[2, 3], [2, 1]]
    assert api.serialize_fast_tokenizer(clone) is clone
    assert all(vars(clone)[name] is stored[name] for name in names)
    assert clone(batch, padding=True)["input_ids"] == [[2, 3], [2, 1]]
    tok.padding_side = "left"
    clone.padding_side = "right"
    assert clone(batch, padding=True)["input_ids"] == [[2, 3], [1, 2]]
