"""P400 §5.3: the performance probe's pre-registered statistic and bounds (fork-only, CPU)."""

import sys
from pathlib import Path

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

import p400_perf_probe as probe  # noqa: E402


def _legs(a_tps, b_tps, a_peak, b_peak):
    def leg(arm, tps, peak):
        return dict(arm=arm, measured=[dict(tps=t, decode_peak=peak) for t in tps])

    return [
        leg("A", a_tps[:5], a_peak),
        leg("B", b_tps[:5], b_peak),
        leg("B", b_tps[5:], b_peak),
        leg("A", a_tps[5:], a_peak),
    ]


def test_the_order_is_abba_with_three_warmup_and_five_measured():
    assert probe.ORDER == ("A", "B", "B", "A")
    assert (probe.WARMUP, probe.MEASURED) == (3, 5)
    assert (probe.TARGET_PROMPT, probe.DECODE_TOKENS) == (8192, 256)


def test_arm_stats_pool_both_legs_and_take_the_median():
    stats = probe.arm_stats(_legs([10.0] * 9 + [100.0], [9.0] * 10, 1000, 1005))
    assert stats["A"] == dict(n=10, median_tps=10.0, median_peak=1000, max_peak=1000)
    assert stats["B"]["median_tps"] == 9.0 and stats["B"]["n"] == 10


def test_verdict_applies_both_pre_registered_bounds():
    ok = probe.verdict(probe.arm_stats(_legs([10.0] * 10, [9.75] * 10, 1000, 1009)))
    assert ok["status"] == "WITHIN_BOUNDS" and abs(ok["tps_regression"] - 0.025) < 1e-12
    slow = probe.verdict(probe.arm_stats(_legs([10.0] * 10, [9.6] * 10, 1000, 1000)))
    assert slow["status"] == "OUTSIDE_BOUNDS"  # 4 % slower
    fat = probe.verdict(probe.arm_stats(_legs([10.0] * 10, [10.0] * 10, 1000, 1020)))
    assert fat["status"] == "OUTSIDE_BOUNDS"  # 2 % more decode-phase peak
    short = probe.arm_stats(_legs([10.0] * 10, [10.0] * 10, 1000, 1000))
    short["B"]["n"] = 9
    assert probe.verdict(short)["status"] == "INCOMPLETE"
