"""P400 §5.3: the performance probe's pre-registered statistic and bounds (fork-only, CPU)."""

import json
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


# ------------------------------------------------------------------ code round 2 C3–C6
def test_every_request_must_decode_the_full_256_tokens_to_the_length_cap():
    """C3: an early EOS makes a shorter, cheaper request; it must never enter the statistic."""
    good = dict(tps=10.0, tokens=probe.DECODE_TOKENS, decode_peak=1)
    assert probe.request_problems(good, "length", [1, 2]) == []
    assert probe.request_problems(dict(good, tokens=200), "length", [2])
    assert probe.request_problems(good, "stop", [2])
    assert probe.request_problems(dict(good, tps=0.0), "length", [2])
    assert probe.request_problems({}, "length", [2])
    assert probe.request_problems(good, "length", [1])  # no width-2 verify block


def test_the_arms_are_two_distinct_clean_trees_at_the_expected_commits():
    """C4: an arm is bound to a commit and a clean tree, not a directory path."""
    trees = dict(A="/w/old", B="/w/new")
    refs = dict(A="a" * 40, B="b" * 40)
    att = {
        "/w/old": dict(head="a" * 40, clean=True),
        "/w/new": dict(head="b" * 40, clean=True),
    }
    assert probe.tree_problems(trees, refs, att) == []
    assert probe.tree_problems(
        dict(A="/w/old", B="/w/old"), refs, att
    )  # identical arms
    assert probe.tree_problems(trees, dict(refs, B="a" * 40), att)  # identical commits
    assert probe.tree_problems(
        trees, dict(refs, B="c" * 40), att
    )  # not the expected commit
    dirty = dict(att, **{"/w/new": dict(head="b" * 40, clean=False)})
    assert probe.tree_problems(trees, refs, dirty)


def test_every_leg_after_the_first_is_idle_first_across_picks():
    """C5: the ≥ 10 min idle precedes every globally consecutive leg, including a pick's first."""
    plan = probe.leg_schedule(["p", "q"])
    assert [(s["pick"], s["k"], s["arm"]) for s in plan] == [
        (pick, k, arm) for pick in ("p", "q") for k, arm in enumerate(probe.ORDER)
    ]
    assert [s["idle_before"] for s in plan] == [False] + [True] * 7


def test_the_idle_time_cannot_be_shortened_below_ten_minutes(capsys):
    import pytest

    argv = ["drive", "--stack-root", ".", "--old-tree", "a", "--new-tree", "b"]
    argv += ["--old-ref", "x", "--new-ref", "y", "--out", "o", "--python", "p"]
    with pytest.raises(SystemExit):
        probe.main(argv + ["--idle-s", "599"])
    assert "at least 600" in capsys.readouterr().err


class _RG:
    def __init__(self, bits, scheme, **changes):
        self.kv_bits, self.kv_quant_scheme = bits, scheme
        self.kv_key_bits = self.kv_value_bits = None
        self.kv_key_scheme = self.kv_value_scheme = None
        self.kv_group_size, self.quantized_kv_start = 64, 0
        self.__dict__.update(changes)


def test_the_leg_asserts_the_served_tuple_and_kv_codec_its_argv_asks_for(monkeypatch):
    """C6 + P405 F3: the leg proves the deployed MTP tuple (drafter, verify policy) and EVERY resolved KV field its argv
    names (the shared kvmanifest resolution)."""
    calls = []

    class Worker:
        @staticmethod
        def served_state():
            return dict(state=1)

        @staticmethod
        def assert_served(state, expect):
            calls.append((state, expect))

    monkeypatch.setenv("TQ_KV_QUANT_MODE", "mse")
    pick = "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed"
    argv = ["--model", "m", "--kv-bits", "4", "--kv-quant-scheme", "turboquant"]
    argv += ["--quantized-kv-start", "0"]
    got = probe.served_check(Worker, _RG(4.0, "turboquant"), argv, pick)
    from p369_phase2 import kvmanifest

    want = kvmanifest.kv_from_argv(argv)
    assert calls == [
        (dict(state=1), dict(drafter=True, verify_policy="joint_v1", kv=want))
    ]
    assert got["kv"]["kv_quant_scheme"] == "turboquant" and got["state"] == dict(
        state=1
    )
    import pytest

    for rg in (
        _RG(4.0, "uniform"),  # the P405 shape: the CLI asked TurboQuant, affine served
        _RG(None, "uniform"),
        _RG(4.0, "turboquant", kv_group_size=32),
        _RG(4.0, "turboquant", quantized_kv_start=5000),
    ):
        with pytest.raises(RuntimeError, match="KV"):
            probe.served_check(Worker, rg, argv, pick)
    monkeypatch.setenv("TQ_KV_QUANT_MODE", "prod")
    with pytest.raises(RuntimeError, match="KV"):
        probe.served_check(Worker, _RG(4.0, "turboquant"), argv, pick)


# ------------------------------------------------------------------ code round 3 C8 C9
def test_arm_refs_must_be_full_resolved_commit_ids():
    """C8: a prefix or an empty ref could match any HEAD; refs are resolved to full ids before comparison."""
    trees = dict(A="/w/old", B="/w/new")
    att = {
        "/w/old": dict(head="a" * 40, clean=True),
        "/w/new": dict(head="a" * 40, clean=True),
    }
    for refs in (dict(A="", B="a" * 40), dict(A="aaaa", B="a" * 40)):
        assert any(
            "full commit id" in p for p in probe.tree_problems(trees, refs, att)
        ), refs


def test_two_spellings_of_one_commit_resolve_to_the_same_id(tmp_path):
    """C8 regression: a short and a full spelling of one commit are the same arm."""
    import subprocess

    git = ["git", "-C", str(tmp_path)]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(
        git
        + [
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    full = probe.resolve_ref(tmp_path, "HEAD")
    assert len(full) == 40 and probe.resolve_ref(tmp_path, full[:8]) == full
    trees = dict(A=str(tmp_path), B="/w/other")
    att = {
        str(tmp_path): dict(head=full, clean=True),
        "/w/other": dict(head=full, clean=True),
    }
    refs = dict(A=probe.resolve_ref(tmp_path, full[:8]), B=full)
    assert any("same commit" in p for p in probe.tree_problems(trees, refs, att))
    import pytest

    with pytest.raises(SystemExit, match="resolve"):
        probe.resolve_ref(tmp_path, "")


def test_non_finite_or_non_positive_measurements_are_refused():
    """C9: inf / nan / zero / negative telemetry must never satisfy the bounds."""
    good = dict(tps=10.0, tokens=probe.DECODE_TOKENS, decode_peak=1000)
    for bad in (
        dict(tps=float("inf")),
        dict(tps=float("nan")),
        dict(tps=-1.0),
        dict(decode_peak=0),
        dict(decode_peak=-5),
        dict(decode_peak=float("nan")),
    ):
        assert probe.request_problems(dict(good, **bad), "length", [2]), bad
    stats = probe.arm_stats(_legs([10.0] * 10, [float("inf")] * 10, 1000, 1000))
    assert probe.verdict(stats)["status"] == "INVALID"
    stats = probe.arm_stats(_legs([10.0] * 10, [10.0] * 10, 1000, 0))
    assert probe.verdict(stats)["status"] == "INVALID"


# ------------------------------------------------------------------ P405 R5 (code round 1 K5): codec-only ABBA
REGISTRY = """models:
  - name: Qwen3.8-27B-mlx-uniform-4bit
    kv_quant_scheme: turboquant
    kv_bits: 4                     # turboquant 4-bit KV
  - name: other
    kv_quant_scheme: turboquant
"""


def test_codec_overlay_changes_exactly_the_picks_scheme_line():
    out = probe.codec_overlay(REGISTRY, "Qwen3.8-27B-mlx-uniform-4bit", "uniform")
    assert out.count("kv_quant_scheme: uniform") == 1
    assert out.split("- name: other")[1].count("kv_quant_scheme: turboquant") == 1
    assert (
        probe.codec_overlay(REGISTRY, "Qwen3.8-27B-mlx-uniform-4bit", "turboquant")
        == REGISTRY
    )
    import pytest

    with pytest.raises(ValueError, match="kv_quant_scheme"):
        probe.codec_overlay(
            REGISTRY.replace(
                "    kv_quant_scheme: turboquant\n    kv_bits", "    kv_bits"
            ),
            "Qwen3.8-27B-mlx-uniform-4bit",
            "uniform",
        )
    with pytest.raises(ValueError, match="no registry block"):
        probe.codec_overlay(REGISTRY, "absent", "uniform")


def test_codec_schedules_are_the_pre_registered_ones():
    assert probe.CODEC_ARMS == dict(A="uniform", B="turboquant")
    assert probe.CODEC_SCHEDULES == dict(
        short=dict(prompt=8192, tokens=256, warmup=3, measured=5),
        long=dict(prompt=261120, tokens=64, warmup=0, measured=2),
    )


def test_codec_requests_need_the_schedules_token_count_and_positive_prefill():
    good = dict(tps=10.0, tokens=64, decode_peak=1, prefill_tps=300.0, prefill_peak=2)
    assert probe.codec_request_problems(good, "length", 64) == []
    assert probe.codec_request_problems(dict(good, tokens=63), "length", 64)
    assert probe.codec_request_problems(good, "stop", 64)
    for bad in (
        dict(prefill_tps=0.0),
        dict(prefill_peak=float("nan")),
        dict(tps=float("inf")),
    ):
        assert probe.codec_request_problems(dict(good, **bad), "length", 64), bad


def test_codec_stats_are_descriptive_per_arm_and_schedule():
    def leg(arm, schedule, x):
        return dict(
            arm=arm,
            schedule=schedule,
            measured=[
                dict(
                    tps=x,
                    prefill_tps=10 * x,
                    decode_peak=100 * x,
                    prefill_peak=1000 * x,
                )
            ]
            * 2,
        )

    legs = [
        leg("A", "long", 1.0),
        leg("B", "long", 2.0),
        leg("B", "long", 4.0),
        leg("A", "long", 3.0),
    ]
    stats = probe.codec_stats(legs)
    assert (
        stats["long"]["A"]["median"]["tps"] == 2.0
        and stats["long"]["B"]["median"]["tps"] == 3.0
    )
    assert stats["long"]["A"]["range"]["prefill_peak"] == [1000.0, 3000.0]
    assert (
        stats["long"]["A"]["n"] == 4 and "verdict" not in stats["long"]
    )  # descriptive: no bound, no verdict


# ------------------------------------------------------------------ P405 code round 2 K7 / K9
TQ = [
    "--model",
    "m",
    "--kv-bits",
    "4",
    "--kv-quant-scheme",
    "turboquant",
    "--quantized-kv-start",
    "0",
]
AFFINE = [
    "--model",
    "m",
    "--kv-bits",
    "4",
    "--kv-quant-scheme",
    "uniform",
    "--quantized-kv-start",
    "0",
]


def _layers(kinds):
    return [dict(path=str(i), kind=k, cls=k, batch=False) for i, k in enumerate(kinds)]


def test_every_request_checks_the_live_cache_against_its_arms_argv():
    """K7: configuration can say TurboQuant while the cache is affine (P405 itself); the leg checks the LIVE cache."""
    template = _layers(["ArraysCache", "native", "ArraysCache", "native"])
    affine = [
        dict(t, kind="affine", bits=4, group_size=64) if t["kind"] == "native" else t
        for t in template
    ]
    assert probe.leg_cache_problems(affine, template, AFFINE) == []
    assert probe.leg_cache_problems(
        affine, template, TQ
    )  # the arms disagree with their live leaves
    assert probe.leg_cache_problems(None, template, TQ) == [
        "no live target cache captured"
    ]


def test_codec_mode_defaults_to_quantized_picks_and_refuses_native_arms():
    """K9: Fable is native KV — changing only the scheme cannot make affine / TurboQuant arms."""
    assert probe.CODEC_PICKS == ("Qwen3.8-27B-mlx-uniform-4bit",)
    assert probe.codec_arm_problems(AFFINE, TQ) == []
    native = ["--model", "m", "--kv-bits", "0", "--kv-quant-scheme", "turboquant"]
    assert probe.codec_arm_problems(native, native)
    assert probe.codec_arm_problems(TQ, TQ)  # both arms TurboQuant
    assert probe.codec_arm_problems(AFFINE, AFFINE)


def test_the_live_cache_manifest_is_kept_verbatim_per_request():
    """P405 code round 3 K10: the report carries each request's full validated manifest (paths, kinds, widths,
    group size, mode), not only the kinds — materially different manifests must not record identically.
    """
    live = _layers(["ArraysCache", "native"])
    rep = probe.request_record(dict(tps=1.0), "length", [2], live, warmup=False)
    assert (
        rep["live_cache"] == live and json.loads(json.dumps(rep))["live_cache"] == live
    )


def test_p429_a_completed_leg_exits_without_interpreter_finalization(monkeypatch):
    """P429 (R5 Qwen3.6 leg 1-B, rc -11): a background thread's MLX compile-cache destructor re-entered a finalized
    interpreter at exit and segfaulted AFTER the leg's result was written. A completed leg flushes and ``os._exit(0)``s.
    """
    calls = []
    monkeypatch.setattr(probe, "leg", lambda job: calls.append(("leg", job)))
    monkeypatch.setattr(
        probe.os,
        "_exit",
        lambda code: calls.append(("_exit", code))
        or (_ for _ in ()).throw(SystemExit(code)),
    )
    try:
        probe.main(["leg", "job.json"])
    except SystemExit as exc:
        assert exc.code == 0
    assert calls == [("leg", "job.json"), ("_exit", 0)]


def test_p429_a_failing_leg_still_raises_and_never_exits_zero(monkeypatch):
    def boom(job):
        raise RuntimeError("request 3: problems")

    monkeypatch.setattr(probe, "leg", boom)
    monkeypatch.setattr(
        probe.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(AssertionError("_exit called")),
    )
    import pytest

    with pytest.raises(RuntimeError, match="request 3"):
        probe.main(["leg", "job.json"])
