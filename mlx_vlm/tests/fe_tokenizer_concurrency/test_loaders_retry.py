"""Local-only loader integration and the retry logging tripwire."""

import json
import logging

import pytest

from mlx_vlm.prompt_utils import _encode_retrying_on_borrow_error
from mlx_vlm.tests.test_tokenizer_concurrency import _FlakyTokenizer
from mlx_vlm.tokenizer_utils import TokenizerWrapper, load_tokenizer
from mlx_vlm.utils import load_processor


@pytest.fixture
def local_model(tok, tmp_path):
    # bert's AutoProcessor mapping falls back to AutoTokenizer on transformers
    # 5.14.1 when only tokenizer files and model config exist. All inputs local.
    tok.backend_tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "model_max_length": 100000,
            }
        )
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))
    return tmp_path


def test_load_tokenizer_serializes_before_wrapping(local_model):
    """Reproduces: tokenizer-only loading currently returns an unprotected tokenizer."""
    wrapper = load_tokenizer(
        local_model, tokenizer_config_extra={"local_files_only": True}
    )
    assert isinstance(wrapper, TokenizerWrapper)
    inner = vars(wrapper)["_tokenizer"]
    assert vars(inner).get("_mlx_vlm_serialized") is True
    from mlx_vlm.tokenizer_lock import is_serialized

    assert is_serialized(inner)
    assert inner.encode("alpha beta") == [2, 3]


def test_load_processor_serializes_tokenizer_fallback(local_model):
    """Reproduces: AutoProcessor's bare-tokenizer fallback must be protected too."""
    processor = load_processor(
        local_model, add_detokenizer=False, local_files_only=True
    )
    assert processor.is_fast
    assert vars(processor).get("_mlx_vlm_serialized") is True
    from mlx_vlm.tokenizer_lock import is_serialized

    assert is_serialized(processor)
    assert processor.encode("alpha beta") == [2, 3]


def test_load_processor_serializes_a_processor_owned_tokenizer(
    local_model, tok, monkeypatch
):
    """New behaviour: the processor hook also covers an actual .tokenizer member."""
    from types import SimpleNamespace

    from mlx_vlm import utils

    processor = SimpleNamespace(tokenizer=tok)
    monkeypatch.setattr(
        utils.AutoProcessor, "from_pretrained", lambda *a, **k: processor
    )
    assert load_processor(local_model, add_detokenizer=False) is processor
    assert vars(tok).get("_mlx_vlm_serialized") is True


@pytest.mark.parametrize("fail_times", [1, 3])
def test_each_retried_attempt_warns_once_without_prompt(caplog, fail_times):
    """New behaviour: each successful retry logs its attempt and class, never user data."""
    prompt = "private synthetic prompt sentinel 9c76"
    tokenizer = _FlakyTokenizer(fail_times=fail_times)
    with caplog.at_level(logging.WARNING, logger="mlx_vlm.prompt_utils"):
        assert _encode_retrying_on_borrow_error(tokenizer, prompt, backoff=0) == [7]
    records = [
        r
        for r in caplog.records
        if r.name == "mlx_vlm.prompt_utils" and r.levelno == logging.WARNING
    ]
    assert tokenizer.calls == fail_times + 1
    assert len(records) == fail_times
    for attempt, record in enumerate(records, 1):
        message = record.getMessage()
        assert str(attempt) in message
        assert "_FlakyTokenizer" in message
        assert prompt not in message
        assert prompt not in repr(record.args)


def test_clean_encode_emits_no_retry_warning(caplog):
    """Preserves: an uncontended encode is quiet and retains its result."""
    tokenizer = _FlakyTokenizer(fail_times=0)
    with caplog.at_level(logging.WARNING, logger="mlx_vlm.prompt_utils"):
        assert _encode_retrying_on_borrow_error(tokenizer, "alpha", backoff=0) == [7]
    assert tokenizer.calls == 1
    assert not [r for r in caplog.records if r.name == "mlx_vlm.prompt_utils"]


def test_exhaustion_warns_only_for_attempts_that_are_retried(caplog):
    """New behaviour: the final propagated failure is not counted as another retry."""
    tokenizer = _FlakyTokenizer(fail_times=99)
    with caplog.at_level(logging.WARNING, logger="mlx_vlm.prompt_utils"):
        with pytest.raises(RuntimeError, match="Already borrowed"):
            _encode_retrying_on_borrow_error(
                tokenizer, "private exhausted prompt", attempts=3, backoff=0
            )
    records = [
        r
        for r in caplog.records
        if r.name == "mlx_vlm.prompt_utils" and r.levelno == logging.WARNING
    ]
    assert tokenizer.calls == 3
    assert len(records) == 2
    assert all("private exhausted prompt" not in r.getMessage() for r in records)


def test_unrelated_runtime_error_propagates_without_warning(caplog):
    """Preserves: unrelated exceptions propagate immediately without retry warnings."""
    tokenizer = _FlakyTokenizer(fail_times=99, message="synthetic corrupt vocabulary")
    with caplog.at_level(logging.WARNING, logger="mlx_vlm.prompt_utils"):
        with pytest.raises(RuntimeError, match="synthetic corrupt vocabulary"):
            _encode_retrying_on_borrow_error(tokenizer, "alpha", backoff=0)
    assert tokenizer.calls == 1
    assert not [r for r in caplog.records if r.name == "mlx_vlm.prompt_utils"]
