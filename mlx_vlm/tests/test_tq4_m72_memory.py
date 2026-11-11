"""M72 1.3 settled constraint (C314 / C317): one layer's scratch may exist at a time; prefill is evaluated once per TQ4
full-attention layer per chunk; the scratch is the state's capacity (one size, so MLX's buffer cache reuses it).

Asserted behaviourally on a multi-layer synthetic chain through the model's own prefill: the allocator peak of one prefill chunk
above the resident arrays stays within one scratch (+ slack), against a control that shows the instrument sees a lazy graph
(chained prototype scratch calls evaluated once at the end hold several scratches), and the scratch is capacity-sized (the
peak is at least one scratch at a small T). No test reads the source.

Every test is classed in its comment: [new] exercises new behaviour, [invariant] instrument check (holds on today's code),
[fix] reproduces the failure the specification fixes (a lazy 16-layer graph holds ~6.4 scratches).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from test_tq4_m72_routing import build, counted, step
from test_tq4_m72_support import (
    BF16,
    MODES,
    NKV,
    SCALE,
    D,
    mx,
    needs_metal,
    queries_for,
    tq4_cache,
)

pytestmark = needs_metal

CAP = 65536
SCRATCH = (
    2 * NKV * CAP * D * 2
)  # K_rot + V_rot, fp16, capacity-sized: one layer's scratch in bytes
DEV = Path(__file__).resolve().parents[2] / "dev"
LAYERS = 20  # interval 2 -> 10 full-attention layers: 9 TQ4 + 1 native


def peak_above_resident(fn):
    fn()  # warm: kernels compiled, buffer cache primed with the working-set sizes
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    out = fn()
    mx.eval(out)
    mx.synchronize()
    return mx.get_peak_memory() - base


# ------------------------------------------------------------------ instrument (the dev prototype is today's code)
@pytest.fixture(scope="module")
def sp():
    spec = importlib.util.spec_from_file_location(
        "m72_scratch_prefill", DEV / "m72_scratch_prefill.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m72_scratch_prefill"] = mod
    spec.loader.exec_module(mod)
    return mod


def chain(sp, caches, per_layer_eval):
    q = queries_for(64, 5)
    for c in caches:
        q = sp.scratch_prefill_tq4(c, q, SCALE)
        if per_layer_eval:
            mx.eval(q)
    return q


# [invariant] the instrument tells a lazy chain (several scratches live) from a per-layer-evaluated one (one scratch)
def test_instrument_lazy_chain_holds_several_scratches_and_per_layer_eval_holds_one(sp):
    caches = [tq4_cache(1024, cap=CAP, seed=40 + i) for i in range(9)]
    lazy = peak_above_resident(lambda: chain(sp, caches, False))
    eager = peak_above_resident(lambda: chain(sp, caches, True))
    assert lazy >= 3 * SCRATCH, f"lazy {lazy / SCRATCH:.2f} scratches"
    assert eager <= 2 * SCRATCH, f"per-layer {eager / SCRATCH:.2f} scratches"


# ------------------------------------------------------------------ the production prefill through the model
def prefill_peak(mode, T0, chunk=512):
    m = build(T0, mode=mode, layers=LAYERS, cap=CAP)
    with counted() as c:
        extra = peak_above_resident(lambda: step(m, chunk).astype(BF16))
    return extra, c.delta


# [new] [fix] one chunk through 9 TQ4 layers peaks within two scratches (a lazy graph would hold ~9), at T_eff <= 2048
@pytest.mark.parametrize("mode", MODES)
def test_prefill_chunk_peak_is_one_scratch_not_one_per_layer(mode):
    extra, delta = prefill_peak(mode, 1024)
    assert delta.get("p2s", 0) >= 9, delta
    assert (
        extra <= 2 * SCRATCH
    ), f"{extra / SCRATCH:.2f} scratches above the resident arrays"


# [new] the same above 2,048 (the regime the pick lives in)
def test_prefill_chunk_peak_above_2048():
    extra, delta = prefill_peak("m72_v1", 2600)
    assert delta.get("p2s", 0) >= 9, delta
    assert extra <= 2 * SCRATCH, f"{extra / SCRATCH:.2f} scratches"


# [new] the scratch is the state's capacity, not T: at T_eff = 1,024 the peak still holds one capacity-sized scratch
def test_scratch_is_capacity_sized():
    extra, delta = prefill_peak("m72_v1", 1024)
    assert delta.get("p2s", 0) >= 9
    assert (
        extra >= 0.9 * SCRATCH
    ), f"{extra / SCRATCH:.2f} scratches: a T-sized scratch would be ~{1024 / CAP:.3f}"


# [new] one size across chunks: the buffer cache after several chunks at growing T holds no pile of stale scratches
def test_buffer_cache_does_not_pile_up_scratches_across_chunks():
    m = build(512, mode="m72_v1", layers=LAYERS, cap=CAP)
    step(m, 512)
    mx.synchronize()
    mx.clear_cache()
    for _ in range(
        4
    ):  # T_eff grows 1,024 -> 2,560: a T-sized scratch would be a new size each time
        mx.eval(step(m, 512))
    mx.synchronize()
    assert (
        mx.get_cache_memory() <= 3 * SCRATCH
    ), f"{mx.get_cache_memory() / SCRATCH:.2f} scratches cached"


# [new] the cache's own state is untouched by the scratch (no dequantized K/V kept: active memory returns to the resident set)
def test_active_memory_returns_after_a_chunk():
    m = build(1024, mode="m72_v1", layers=LAYERS, cap=CAP)
    step(m, 512)
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.eval(step(m, 512))
    mx.synchronize()
    mx.clear_cache()
    grown = mx.get_active_memory() - before
    # the TQ4 caches are preallocated (no growth); native layers + GDN state grow by a few MB, never by a scratch
    assert grown < 0.25 * SCRATCH, f"{grown / 1e6:.1f} MB retained after the chunk"
