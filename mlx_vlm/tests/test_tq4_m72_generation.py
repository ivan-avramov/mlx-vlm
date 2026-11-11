"""M72 1.3 A3 / A4 on the synthetic Qwen3.5-family model (the pick's attention shape, TQ4 + native layers):

A3 "Off is today": with the mode absent, fixed-seed generation is byte-identical in tokens and logits to the parent commit. The
digests below were recorded on the parent commit (before any 1.3 code existed) by ``scenario_*`` below, on the device and MLX
version named in ``GOLDEN_ENV``; on any other device / MLX build the golden tests skip with that reason (record again from the
parent commit there). They are the byte-identity oracle for the mode-off paths: any numerical change to a mode-off path flips a digest.

A4 "On is consistent" (same process): (a) mode-off prefill, snapshot (an identical independently prefilled state), 64 greedy
MTP-OFF tokens mode off vs mode on from the same state: token- and logit-identical (D2 exact), crossing the T = 2,048 edge;
(b) the last prompt token's logits after a mode-on prefill vs a mode-off prefill of the same prompt: within tolerance;
(c) ``m72_exact``: MTP ON emits exactly MTP OFF's tokens (greedy and seeded sampling, n = 256, k up to and past 5).

Every test is classed in its comment: [new] exercises new behaviour, [invariant] preserves an invariant of today's code.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_mtp_exactness import mtp_drafter, scripted
from test_tq4_m72_routing import build, counted, near, pair, stamp_policies
from test_tq4_m72_support import (
    MODES,
    TQ4_KV,
    device_arch,
    digest,
    fake_model,
    greedy,
    mode_module,
    model_cache,
    mtp_run,
    mx,
    needs_metal,
    prefill,
    prompt_ids,
    tiny_model,
)

from mlx_vlm.generate import ar

pytestmark = needs_metal

GOLDEN_ENV = {"arch": "applegpu_g17s", "mlx": "0.32.4"}
GOLDEN = {  # recorded on the parent commit under the suite's numerics (conftest pins MLX_ENABLE_TF32=0)
    "tq4_T600": "d30531cc2f13b676858db21251f27cf067ddfe446fcd64f851a489c239e9a0c2",
    "tq4_T2300": "532377f3f8ac2cd23169c14bf64514304cc308ebf8f55183af7c1ec21c723d05",
    "native_T2300": "a664592ff50861bd4c18cbc92f484ba545ad9b3e38b66e6ef7d356a7fbfacc7a",
    "tq4_mtp_T2100": "366749bd29484cb34d9fff671a8626ffd642354f08edd0aa6b608eb2482544a2",
}


def need_golden(name):
    if GOLDEN[name] is None:
        pytest.fail(f"golden digest {name!r} has not been recorded")
    if device_arch() != GOLDEN_ENV["arch"] or mx.__version__ != GOLDEN_ENV["mlx"]:
        pytest.skip(
            f"goldens were recorded on {GOLDEN_ENV['arch']} / mlx {GOLDEN_ENV['mlx']}; this is {device_arch()} / mlx {mx.__version__}"
        )
    return GOLDEN[name]


# ------------------------------------------------------------------ scenarios (mode absent; also used to record the goldens)
def scenario_plain(kv, T, n=24, chunk=512):
    lm = tiny_model(layers=4)
    cache = model_cache(lm, kv, 4096)
    last = prefill(lm, cache, T, chunk=chunk)
    toks, logits = greedy(lm, cache, last, n)
    return digest(toks, logits + [np.array(last.astype(mx.float32))])


def scenario_mtp_tq4(T=2100, n=64, block=4):
    lm = tiny_model(layers=4)
    prompt = prompt_ids(T)
    off, _ = mtp_run(lm, None, n=n, kv=TQ4_KV, block=block, prompt=prompt)
    drafter = mtp_drafter(lm, block=block)
    scripted(drafter, lambda p: off[p] if p < len(off) else None, high=True, vocab=64)
    on, _ = mtp_run(lm, drafter, n=n, kv=TQ4_KV, block=block, prompt=prompt)
    return digest(off + on, [])


# ================================================================== A3
# [invariant] A3: TQ4 prefill + decode below the D2 edge, mode absent, byte-identical to the parent commit
def test_a3_tq4_below_2048_is_byte_identical_to_the_parent_commit():
    assert scenario_plain("tq4", 600) == need_golden("tq4_T600")


# [invariant] A3: TQ4 above 2,048 (the D2 / p2s regimes when the mode is on): mode absent is still today's path
def test_a3_tq4_above_2048_is_byte_identical_to_the_parent_commit():
    assert scenario_plain("tq4", 2300) == need_golden("tq4_T2300")


# [invariant] A3: native16
def test_a3_native16_is_byte_identical_to_the_parent_commit():
    assert scenario_plain("native", 2300) == need_golden("native_T2300")


# [invariant] A3: TQ4 with MTP ON (full-acceptance scripted drafter) mode absent: the same tokens as the parent commit
def test_a3_tq4_mtp_is_byte_identical_to_the_parent_commit():
    assert scenario_mtp_tq4() == need_golden("tq4_mtp_T2100")


# [invariant] the scenarios themselves are deterministic in-process (so a digest difference is the code's, not noise)
def test_a3_scenarios_are_deterministic():
    assert scenario_plain("tq4", 600, n=6) == scenario_plain("tq4", 600, n=6)


# ================================================================== A4 (a)
# [new] A4(a): same state, MTP OFF, mode off vs mode on: identical tokens and bit-identical logits for 64 steps that cross
# the T_eff = 2,048 edge (single-pass -> D2) — both kernels are exercised
@pytest.mark.parametrize("mode", MODES)
def test_a4a_mode_on_decode_is_token_and_logit_identical(mode):
    a, b = pair(2030, mode)
    toks_b, logits_b = greedy(b.lm, b.cache, b.last, 64)
    with counted() as c:
        toks_a, logits_a = greedy(a.lm, a.cache, a.last, 64)
    assert c.delta.get("single_pass", 0) > 0 and c.delta.get("d2", 0) > 0, c.delta
    assert toks_a == toks_b
    for i, (x, y) in enumerate(zip(logits_a, logits_b)):
        assert x.tobytes() == y.tobytes(), f"logits differ at step {i}"


# ================================================================== A4 (b)
# [new] A4(b): last-prompt-token logits after a mode-on prefill vs a mode-off prefill of the same prompt (within A1's tolerance)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("T", (2600, 4000))
def test_a4b_prefill_logits_mode_on_vs_off(mode, T):
    on = build(0, mode=mode, apply_after_prefill=False)
    last_on = prefill(on.lm, on.cache, T)
    off = build(0, mode=None, apply_after_prefill=False)
    last_off = prefill(off.lm, off.cache, T)
    assert last_on.shape == last_off.shape and bool(
        mx.all(mx.isfinite(last_on.astype(mx.float32))).item()
    )
    assert near(last_on, last_off)
    # and the argmax token (what a greedy continuation starts from) agrees
    assert int(mx.argmax(last_on).item()) == int(mx.argmax(last_off).item())


# ================================================================== A4 (c)
SAMPLING = {
    "greedy": {},
    "seeded": dict(temperature=0.8, top_p=0.95, top_k=20, min_p=0.0, seed=11),
}


def run_off_on(mode, block, sampling, *, high, n=256, T=2100):
    lm = tiny_model(layers=4)
    stamp_policies(lm)
    mode_module().apply_to_model(lm, mode)
    prompt = prompt_ids(T)
    off, _ = mtp_run(lm, None, n=n, kv=TQ4_KV, block=block, prompt=prompt, **sampling)
    drafter = mtp_drafter(lm, block=block)
    script = scripted(
        drafter, lambda p: off[p] if p < len(off) else None, high=high, vocab=64
    )
    with counted() as c:
        on, _ = mtp_run(
            lm, drafter, n=n, kv=TQ4_KV, block=block, prompt=prompt, **sampling
        )
    return off, on, c.delta, script


# [new] A4(c) greedy: m72_exact: MTP ON emits exactly MTP OFF's tokens, high and no acceptance,
# widths 3 (k = 3), 5 (k = 5) and 7 (k = 7 > 5, the per-query loop)
@pytest.mark.parametrize("sampling", ["greedy"])
@pytest.mark.parametrize("high", (True, False), ids=["accepting", "rejecting"])
@pytest.mark.parametrize("block", (3, 5, 7))
def test_a4c_exact_mode_mtp_on_equals_mtp_off(sampling, high, block):
    off, on, delta, script = run_off_on(
        "m72_exact", block, SAMPLING[sampling], high=high
    )
    assert len(off) == len(on) == 256
    assert on == off
    assert script.rounds, "no speculative round ran"
    assert delta.get("mma", 0) == 0 and delta.get("mma_prefill", 0) == 0, delta
    if block <= 6:
        assert delta.get("d2_verify", 0) > 0, delta
    else:
        assert delta.get("per_query:k_gt_5", 0) > 0, delta


def from_snapshot(lm, drafter, *, n, block, T, mode, sampling):
    """Decode ``n`` tokens (MTP ON when ``drafter``) from a state built by a MODE-OFF prefill of the first T - 1 prompt tokens
    (the snapshot); the mode, when given, is applied only afterwards (spec A4: same snapshot, then apply the mode).
    """
    cache = model_cache(lm, "tq4", 4096)
    prefill(lm, cache, T - 1)
    if mode:
        mode_module().apply_to_model(lm, mode)
    extra = {}
    if drafter is not None:
        extra = dict(draft_model=drafter, draft_kind="mtp", draft_block_size=block)
    stream = ar.generate_step(
        prompt_ids(T)[:, -1:],
        fake_model(lm),
        None,
        None,
        max_tokens=n,
        prompt_cache=cache,
        prefill_step_size=512,
        **TQ4_KV,
        **extra,
        **sampling,
    )
    return [int(t) for t, _ in stream]


# [new] (delivery 3, spec erratum v1.1b) A4(c) seeded: m72_exact MTP ON emits exactly what MODE-ABSENT TQ4 MTP ON emits for the
# same seed (both verify exactly; MTP ON seeded is not asserted equal to MTP OFF: that fails with the mode absent too).
# Same proposals for both runs (scripted from a mode-absent MTP OFF run), widths 3 / 5 / 7, accepting and rejecting.
@pytest.mark.parametrize("high", (True, False), ids=["accepting", "rejecting"])
@pytest.mark.parametrize("block", (3, 5, 7))
def test_a4c_exact_mode_seeded_mtp_on_equals_mode_absent_mtp_on(high, block):
    n, T, seeded = 256, 2100, SAMPLING["seeded"]
    ref = tiny_model(layers=4)
    stamp_policies(ref)
    off = from_snapshot(ref, None, n=n, block=block, T=T, mode=None, sampling=seeded)

    def on_run(mode):
        lm = tiny_model(layers=4)
        stamp_policies(lm)
        drafter = mtp_drafter(lm, block=block)
        script = scripted(
            drafter, lambda p: off[p] if p < len(off) else None, high=high, vocab=64
        )
        with counted() as c:
            toks = from_snapshot(
                lm, drafter, n=n, block=block, T=T, mode=mode, sampling=seeded
            )
        return toks, c.delta, script

    absent, _, script_absent = on_run(None)
    exact, delta, script_exact = on_run("m72_exact")
    assert len(absent) == len(exact) == n
    assert script_absent.rounds and script_exact.rounds
    assert delta.get("mma", 0) == 0
    assert (
        delta.get("d2_verify", 0) > 0
        if block <= 6
        else delta.get("per_query:k_gt_5", 0) > 0
    ), delta
    assert exact == absent, next(
        i for i, (a, b) in enumerate(zip(exact, absent)) if a != b
    )


# [new] m72_v1 runs the matrix-unit verify end to end (not asserted exact: S21-P9, A6's diagnostic): it completes, emits n
# tokens, and routes through the mma kernel
@pytest.mark.parametrize("block", (3, 5))
def test_a4c_fast_mode_runs_mtp_through_the_matrix_unit_kernel(block):
    off, on, delta, script = run_off_on(
        "m72_v1", block, SAMPLING["greedy"], high=True, n=64
    )
    assert len(on) == len(off) == 64
    assert delta.get("mma", 0) > 0, delta
    assert delta.get("d2_verify", 0) == 0
