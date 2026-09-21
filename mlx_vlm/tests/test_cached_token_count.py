"""Fork-only (C91): ``StreamingToken.token_count`` on the session-cached path must be
the INCREMENT of dispatch's cumulative ``GenerationResult.generation_tokens``, not the
dataclass default of 1 per chunk.

Measured on the shipped ``Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed`` (C89
calibration, ``max_tokens: 1``): one generated token, ``finish_reason: length``, and
``usage.completion_tokens == 2``. ``generate/dispatch.py`` yields one chunk per token
with ``generation_tokens = n + 1`` and then a finalization chunk (detokenizer flush)
that REPEATS the last cumulative count on length termination; on EOS/stop the loop
breaks before yielding the EOS token, so the finalization chunk's count is genuinely
one higher. ``_process_cached_request`` gave every chunk ``token_count=1`` and
``server/openai.py`` sums those, so a length-terminated request over-counted by one
while a stop-terminated one happened to be right. ``_DiffusionBlockEmitter`` already
does the increment accounting; this brings the cached bridge in line with it.

Drives the REAL ``_process_cached_request`` with dispatch-shaped chunk sequences and
reads ``token_count`` off the queue, the same way the endpoint does.
"""

import sys
from queue import Queue
from types import SimpleNamespace

import pytest

import mlx_vlm.server as server
import mlx_vlm.server.generation as server_generation
from mlx_vlm.generate.common import GenerationResult

from test_cached_tokens_reporting import _bare_response_generator


def _chunk(text, token, generation_tokens, finish_reason=None):
    return GenerationResult(
        text=text,
        token=token,
        logprobs=None,
        prompt_tokens=10,
        generation_tokens=generation_tokens,
        total_tokens=10 + generation_tokens,
        prompt_tps=100.0,
        generation_tps=50.0,
        peak_memory=1.0,
        cached_tokens=0,
        finish_reason=finish_reason,
    )


def _drive(monkeypatch, chunks):
    def _gen(**kwargs):
        yield from chunks

    monkeypatch.setattr(sys.modules["mlx_vlm.generate"], "stream_generate", _gen)
    rg = _bare_response_generator()
    rqueue: Queue = Queue()
    rg._process_cached_request(
        rqueue=rqueue,
        prompt="hello",
        images=None,
        args=server.GenerationArguments(),
        prompt_tokens=10,
        prompt_cache_state=SimpleNamespace(),
    )
    rqueue.get_nowait()  # ctx
    tokens = []
    while True:
        item = rqueue.get_nowait()
        if item is None:
            break
        if isinstance(item, server_generation.KeepAlive):
            continue
        assert not isinstance(item, Exception), item
        tokens.append(item)
    return tokens


class TestCachedPathTokenCount:
    def test_max_tokens_one_length_reports_one_token(self, monkeypatch):
        """The saved C89/C84 calibration shape: one token, then a finalization chunk
        repeating ``generation_tokens=1`` with ``finish_reason='length'``."""
        tokens = _drive(monkeypatch, [_chunk("The", 100, 1), _chunk("", 100, 1, "length")])
        assert [t.token_count for t in tokens] == [1, 0]
        assert sum(t.token_count for t in tokens) == 1
        # Text and terminal metadata are preserved untouched.
        assert "".join(t.text for t in tokens) == "The"
        assert [t.finish_reason for t in tokens] == [None, "length"]

    def test_many_tokens_then_repeated_count_finalization(self, monkeypatch):
        chunks = [_chunk("a", 1, 1), _chunk("b", 2, 2), _chunk("c", 3, 3), _chunk("", 3, 3, "length")]
        tokens = _drive(monkeypatch, chunks)
        assert [t.token_count for t in tokens] == [1, 1, 1, 0]

    def test_stop_finalization_counts_the_newly_reported_eos(self, monkeypatch):
        """On EOS dispatch breaks before yielding the EOS token; its finalization chunk
        carries ``n + 1`` — a real increment, so it IS counted (no universal minus one)."""
        chunks = [_chunk("a", 1, 1), _chunk("b", 2, 2), _chunk("", 7, 3, "stop")]
        tokens = _drive(monkeypatch, chunks)
        assert [t.token_count for t in tokens] == [1, 1, 1]

    def test_zero_generated_tokens(self, monkeypatch):
        tokens = _drive(monkeypatch, [_chunk("", None, 0, "length")])
        assert [t.token_count for t in tokens] == [0]

    def test_decreasing_cumulative_count_is_not_fabricated_usage(self, monkeypatch):
        """Malformed producer: the count must never go negative or invent tokens."""
        chunks = [_chunk("a", 1, 1), _chunk("b", 2, 2), _chunk("", 2, 1, "length")]
        tokens = _drive(monkeypatch, chunks)
        assert [t.token_count for t in tokens] == [1, 1, 0]

    def test_chunk_without_generation_tokens_keeps_one_per_chunk(self, monkeypatch):
        """Minimal stand-ins used elsewhere in the suite carry no cumulative count;
        they keep the historical one-token-per-chunk behaviour."""
        chunks = [SimpleNamespace(text="x", token=5, logprobs=None, finish_reason=None, peak_memory=0.0),
                  SimpleNamespace(text="", token=5, logprobs=None, finish_reason="stop", peak_memory=0.0)]
        tokens = _drive(monkeypatch, chunks)
        assert [t.token_count for t in tokens] == [1, 1]

    def test_metrics_agree_with_the_bridge(self, monkeypatch):
        tokens = _drive(monkeypatch, [_chunk("The", 100, 1), _chunk("", 100, 1, "length")])
        metrics = server_generation.GenerationMetrics()
        for t in tokens:
            metrics.record_chunk(t)
        assert metrics.generated_tokens == 1
