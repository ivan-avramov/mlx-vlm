"""M76 B2 (Sol P2): below 2,048 keys ``m72_exact`` counts ``per_query:short`` and returns None, and joint_v1 then records
verify fallback reason ``cache`` for that TurboQuant layer. The E1c rule of a tuple run accepts reason ``cache`` only at
the trajectory's ``per_query:short`` delta. Evidence here is real: the synthetic qwen3_5 target (``lm24``: TQ4 layer 1,
native last layer 3) with the shipped policies stamped, a served TurboQuant cache, the real ``Tq4Mode`` and
``JointV1Policy`` counters, and the worker's ``tq4_engagement`` on the real route counters (Metal GPU; no real model).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal GPU required for the TQ4 kernels"
)

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import orchestrate, worker  # noqa: E402
from test_tq4_impl_support import lm24, prefill, prompt, tq4_cache  # noqa: E402

from mlx_vlm import attention_policy as ap  # noqa: E402
from mlx_vlm import mtp_verify_scan as mv  # noqa: E402
from mlx_vlm import tq4_attention as t4  # noqa: E402
from mlx_vlm import tq4_fast as f  # noqa: E402

PROMPT = 1500  # every verify block's lengths[0] <= 2,048: the per_query:short side


@pytest.fixture
def served():
    """lm24 with fused_v1 + joint_v1 (on this device's class) + m72_exact stamped, a served cache prefilled to PROMPT."""
    lm = lm24()
    model = type("M", (), {"config": lm.config, "language_model": lm})()
    ap.apply_to_model(model, ap.FusedV1Policy())
    live = mv._device_class()
    vp = mv.JointV1Policy(domain=mv.Domain(device_classes=(live,)))
    mv.apply_to_model(model, vp)
    t4.apply_to_model(model, "m72_exact")
    cache = tq4_cache(lm)
    prefill(lm, cache, prompt(PROMPT))
    yield lm, vp, cache
    t4.apply_to_model(model, None)


def verify_block(lm, vp, cache, tokens):
    """One verify block through the model: (the policy's ``since`` timings, the worker's engagement record)."""
    snap, before = vp.counters(), f.route_counts()
    out = lm(mx.array([tokens]), cache=cache, speculative_verify=True)
    mx.eval(out.logits)
    timings = vp.since(snap)
    engagement = worker.tq4_engagement(
        before, f.route_counts(), prompt_tokens=PROMPT, generated=len(tokens)
    )
    return timings, engagement


def on_entry(timings, engagement):
    """The E1c ON entry around the real counters (the comparator / kernel evidence passing, a T1024 straddle cell)."""
    return dict(
        verdict={"pass": True},
        draft_rounds=1,
        prompt_end=orchestrate.straddle_target(1024),
        response=dict(timings=dict(timings, verify_blocks_straddle=1)),
        tq4_engagement=engagement,
    )


OFF = dict(rows={"pass": True}, states={"pass": True}, forced={"pass": True})


def reasons_problems(entry, *, tq4):
    v = orchestrate.e1c_verdict(
        entry,
        OFF,
        threshold=1024,
        verify_policy="joint_v1",
        **(dict(tq4=True) if tq4 else {}),
    )
    return [p for p in v["evidence_problems"] if p.startswith("fallback reasons")]


def test_per_query_short_is_joint_v1s_cache_reason_and_the_tuple_accepts_it(served):
    lm, vp, cache = served
    timings, engagement = verify_block(lm, vp, cache, [5, 9, 11])
    # the real routes: the TQ4 layer's block went to the per-query loop, joint_v1 counted it as reason ``cache``
    assert engagement["ok"] is True and engagement["expected"] == ["per_query:short"]
    assert engagement["per_query_short"] == 1 and engagement["d2_verify"] == 0
    assert timings[mv.REASONS_KEY] == {"cache": 1}
    entry = on_entry(timings, engagement)
    assert (
        reasons_problems(entry, tq4=True) == []
    )  # tuple run: cache == per_query:short
    # a non-tuple run is unchanged: any fallback reason fails
    assert reasons_problems(entry, tq4=False) == ["fallback reasons {'cache': 1}"]


def test_a_cache_reason_that_is_not_per_query_short_is_refused(served):
    """A verify block of six (``per_query:k_gt_5``) also reaches joint_v1 as reason ``cache`` on the TQ4 layer, with no
    per_query:short (and ``gqa_bound`` on the native layer: 6 x 6 > 32): the tuple's rule refuses it.
    """
    lm, vp, cache = served
    timings, engagement = verify_block(lm, vp, cache, [5, 9, 11, 2, 7, 4])
    reasons = timings[mv.REASONS_KEY]
    assert engagement["per_query_short"] == 0 and reasons["cache"] == 1
    assert reasons_problems(on_entry(timings, engagement), tq4=True) == [
        f"fallback reasons {reasons} (per_query:short delta 0)"
    ]


def test_two_short_blocks_count_two_on_both_sides(served):
    lm, vp, cache = served
    snap, before = vp.counters(), f.route_counts()
    for tokens in ([5, 9, 11], [2, 7]):
        out = lm(mx.array([tokens]), cache=cache, speculative_verify=True)
        mx.eval(out.logits)
    timings = vp.since(snap)
    engagement = worker.tq4_engagement(
        before, f.route_counts(), prompt_tokens=PROMPT, generated=5
    )
    assert (
        timings[mv.REASONS_KEY] == {"cache": 2} and engagement["per_query_short"] == 2
    )
    assert reasons_problems(on_entry(timings, engagement), tq4=True) == []
