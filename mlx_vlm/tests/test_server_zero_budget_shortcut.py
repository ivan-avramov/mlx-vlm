"""P369 round-6 F26: a zero-budget uncached request never reaches the GPU.

Fork-only file. The batch loop ran the vision/text embedding (``_gpu_embed``) and built
the generator before a zero-budget request was retired. It now answers immediately
(context, one token-less ``length`` chunk, terminator) without embedding or prefill.
Round 7 F29: it is answered before the ``Prefill started`` log line, too.
"""

import logging
from queue import Queue

import mlx.core as mx
import pytest
from test_backend_label import TestBackendLabelReportsThePathThatRan as _Harness

import mlx_vlm.server as server
import mlx_vlm.server.generation as server_generation


def _queued(rqueue, max_tokens):
    return server_generation.QueuedGenerationRequest(
        rqueue=rqueue,
        raw_inputs={"input_ids": mx.zeros((1, 3), dtype=mx.int32)},
        prompt_tokens=3,
        args=server.GenerationArguments(max_tokens=max_tokens),
        prompt_cache_state=None,
        prompt="hello",
    )


def test_f26_zero_budget_request_skips_embedding_and_prefill(monkeypatch):
    from queue import Queue

    calls = {"embed": 0}
    orig_setattr = monkeypatch.setattr

    def spy_setattr(target, name, value=None, *a, **k):
        if name == "_gpu_embed":  # count the embedding stub the harness installs
            inner = value

            def counted(*aa, **kk):
                calls["embed"] += 1
                return inner(*aa, **kk)

            value = counted
        return orig_setattr(target, name, value, *a, **k)

    rq = Queue()
    monkeypatch.setattr = spy_setattr  # type: ignore[method-assign]
    try:
        _Harness()._drive_run_once(monkeypatch, _queued(rq, 0))
    finally:
        monkeypatch.setattr = orig_setattr  # type: ignore[method-assign]
    ctx = rq.get_nowait()
    assert isinstance(ctx, server.GenerationContext)
    token = rq.get_nowait()
    assert token.finish_reason == "length" and token.token_count == 0
    assert rq.get_nowait() is None
    assert calls["embed"] == 0, "a zero-budget request ran the embedding"


@pytest.mark.parametrize(("max_tokens", "logged"), [(0, False), (2, True)])
def test_f29_only_a_request_that_prefills_logs_prefill_started(
    monkeypatch, caplog, max_tokens, logged
):
    caplog.set_level(logging.INFO, logger="mlx_vlm.server")
    _Harness()._drive_run_once(monkeypatch, _queued(Queue(), max_tokens))
    lines = [r.getMessage() for r in caplog.records]
    assert any("Prefill started" in line for line in lines) is logged
