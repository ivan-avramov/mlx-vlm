"""P369 phase 2: the orchestrator's pure parts (the overlay, the worker env, the bounds, the E1c rule).

Fork-only file. The orchestrator runs under the STACK venv (mlx-serve, psutil); its pure functions are
tested here without either.
"""

import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import orchestrate  # noqa: E402

FABLE, UNIFORM = list(orchestrate.PICKS)
MARK = "# P369 INTERIM (C177 2026-10-17): MTP OFF on the daily driver"

REGISTRY = f"""models:
  - name: {UNIFORM}
    type: vision
    kv_bits: 4
    # draft_kind: mtp  {MARK}
    # draft_model: caslca/{UNIFORM}-mtp-drafter  # UPLOADED 2026-08-26 (public)
    presentation:
      role: main

  - name: Other-Model
    type: vision
    # draft_kind: mtp  {MARK}
    # draft_model: caslca/other-drafter

  - name: {FABLE}
    type: vision
    kv_bits: 0
    # mtp_verify_scan: joint_v1  {MARK} (joint_v1 requires draft_kind mtp)
    generation_defaults:
      temperature: 0.5
    # draft_kind: mtp  {MARK}
    # draft_model: caslca/{FABLE}-mtp-drafter
    presentation:
      role: main
"""


@pytest.mark.parametrize(
    ("pick", "want"),
    [
        (
            FABLE,
            [
                "mtp_verify_scan: joint_v1",
                "draft_kind: mtp",
                f"draft_model: caslca/{FABLE}-mtp-drafter",
            ],
        ),
        (UNIFORM, ["draft_kind: mtp", f"draft_model: caslca/{UNIFORM}-mtp-drafter"]),
    ],
)
def test_overlay_restores_exactly_the_c177_lines_of_the_pick(pick, want):
    text, restored = orchestrate.overlay_registry(REGISTRY, pick)
    assert restored == want
    before, after = REGISTRY.split("\n"), text.split("\n")
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after) and len(changed) == len(want)
    assert all(b.strip() in want for _, b in changed)
    other = [b for b in after if "other-drafter" in b]
    assert other == [
        "    # draft_model: caslca/other-drafter"
    ]  # another pick untouched


def test_overlay_refuses_a_registry_without_the_marked_lines():
    restored_text, _ = orchestrate.overlay_registry(REGISTRY, FABLE)
    with pytest.raises(ValueError, match="expected 3 C177 lines"):
        orchestrate.overlay_registry(restored_text, FABLE)
    unmarked = REGISTRY.replace(MARK, "# some other note")
    with pytest.raises(ValueError, match="expected 2 C177 lines"):
        orchestrate.overlay_registry(unmarked, UNIFORM)
    with pytest.raises(ValueError, match="no registry block"):
        orchestrate.overlay_registry(REGISTRY, "Missing-Model")


def test_overlay_on_the_live_stack_registry():
    live = DEV.parents[1] / "mlx_local_stack" / "main_models.yaml"
    if not live.exists():
        pytest.skip(
            "no sibling stack checkout: the live C177 registry is not available here"
        )
    text = live.read_text()
    for pick in orchestrate.PICKS:
        new, restored = orchestrate.overlay_registry(text, pick)
        assert any(r.startswith("draft_model: caslca/") for r in restored)


def test_worker_env_drops_conditional_knobs_and_sets_the_session_max():
    base = {
        "PATH": "/bin",
        "HF_HUB_OFFLINE": "1",
        "APC_ENABLED": "1",
        "MLX_VLM_DRAFT_KIND": "suffix",
        "MLX_VLM_PRELOAD_MODEL": "x",
        "MLX_VLM_MTP_VERIFY_SCAN": "joint_v1",
        "MLX_VLM_CACHE_SESSION_MAX": "2",
    }
    env = orchestrate.worker_env(base)
    assert env == {
        "PATH": "/bin",
        "HF_HUB_OFFLINE": "1",
        "MLX_VLM_CACHE_SESSION_MAX": "1",
    }


# ------------------------------------------------------------------ code round 1 H3 H9 H11 H14 + power
def test_worker_env_is_an_allowlist():
    base = {
        "PATH": "/bin",
        "HOME": "/home/u",
        "HF_TOKEN": "t",
        "HF_HUB_OFFLINE": "1",
        "LC_ALL": "C",
        "MLX_EPICACHE_BUDGET": "4096",
        "MIN_OUTPUT_TOKENS": "64",
        "MLX_VLM_SPEC_BATCH_COALESCE_MS": "5",
        "KV_BITS": "4",
        "TQ_FUSED_PREFILL": "1",
        "MLX_SDPA_BLOCKS": "1",
        "SOME_FUTURE_KNOB": "x",
        "PYTHONPATH": "/elsewhere",
    }
    env = orchestrate.worker_env(base)
    assert env == {
        "PATH": "/bin",
        "HOME": "/home/u",
        "HF_TOKEN": "t",
        "HF_HUB_OFFLINE": "1",
        "LC_ALL": "C",
        "MLX_VLM_CACHE_SESSION_MAX": "1",
    }
    assert orchestrate.dropped_env(base) == sorted(k for k in base if k not in env)


@pytest.mark.parametrize(
    ("argv", "refused"),
    [
        (["/v/bin/python", "-m", "mlx_vlm.generate", "--model", "m"], True),
        (["python3", "-I", "-m", "mlx_vlm.server"], True),
        (["/v/bin/python", "-m", "pytest", "-q"], True),
        (["/v/bin/python", "-m", "mlx_lm.server"], True),
        (["uv", "run", "mlx-serve", "start"], True),
        (["/v/bin/mlx-serve", "start"], True),
        (["opencode", "run", "x"], True),
        (["/v/bin/python", "/s/benchmark/bench/run_tg1_probe.py"], True),
        (["/v/bin/python", "-m", "bench.runner"], True),
        (["/v/bin/python", "-I", "/f/dev/p369_phase2/worker.py", "job.json"], True),
        (["zsh", "-c", "cd x && pytest -q && mlx-serve start"], False),
        (["/usr/bin/vim", "notes_mlx_vlm.txt"], False),
        (["/v/bin/python", "-c", "import mlx_vlm"], False),
        (["cursor-agent", "-p", "review mlx_vlm"], False),
        ([], False),
    ],
)
def test_refused_process_inspects_argv_structurally(argv, refused):
    assert bool(orchestrate.refused_process(argv)) is refused, argv


def test_tree_status_refuses_any_tracked_or_untracked_change():
    assert orchestrate.tree_status_problems("") == []
    assert orchestrate.tree_status_problems(" M dev/p369_phase2/worker.py\n")
    assert orchestrate.tree_status_problems("?? dev/p369_phase2/new.py\n")


PMSET_AC = """ Wattage = 140W
 Current = 4990mA
 Voltage = 28000mV
 AdapterID = 28697
 Adapter Name = 140W USB-C Power Adapter
 Hardware Version = 1.0
"""


def test_parse_power_reads_watts_volts_and_battery():
    batt = " -InternalBattery-0 (id=1)\t64%; charging; (no estimate) present: true\n"
    got = orchestrate.parse_power(PMSET_AC, batt)
    assert (got["watts"], got["volts"], got["battery"]) == (140, 28.0, 64)
    assert got["ac_ok"] and got["battery_ok"]
    low_volts = orchestrate.parse_power(PMSET_AC.replace("28000mV", "20000mV"), batt)
    assert not low_volts["ac_ok"]
    assert not orchestrate.parse_power("No adapter attached.\n", batt)["ac_ok"]
    assert not orchestrate.parse_power(PMSET_AC, batt.replace("64%", "19%"))[
        "battery_ok"
    ]


def _binding(**changes):
    pick = dict(
        overlay_sha256="o",
        argv_on=["--model", "m", "--draft-model", "d"],
        argv_off=["--model", "m"],
        params=dict(temperature=0.5),
        target=dict(revision="r", files={"a.safetensors": "1"}),
        drafter=dict(revision="r2", files={"b.safetensors": "2"}),
    )
    pick.update(changes)
    return dict(
        production=orchestrate.PRODUCTION,
        fork_head="h",
        harness_tree="t",
        picks={FABLE: pick},
    )


def test_binding_mismatch_refuses_the_full_run():
    manifest = _binding()
    smoke = dict(binding=orchestrate.binding(manifest))
    assert orchestrate.binding_problems(smoke, manifest) == []
    for key, value in [
        ("argv_on", ["--model", "other"]),
        ("target", dict(revision="r", files={"a.safetensors": "9"})),
        ("overlay_sha256", "x"),
        ("params", dict(temperature=0.6)),
    ]:
        assert orchestrate.binding_problems(smoke, _binding(**{key: value})), key
    head = dict(_binding(), fork_head="other")
    assert orchestrate.binding_problems(smoke, head)
    picks = dict(_binding(), picks={})
    assert orchestrate.binding_problems(smoke, picks)
    assert orchestrate.binding_problems(
        {}, manifest
    )  # a smoke summary without a binding


def test_lock_staleness_needs_a_dead_or_reused_owner():
    record = dict(pid=4242, create_time=100.0)
    assert orchestrate.lock_is_stale(record, lambda pid: None)  # owner gone
    assert orchestrate.lock_is_stale(record, lambda pid: 200.0)  # pid reused
    assert not orchestrate.lock_is_stale(record, lambda pid: 100.0)  # owner alive
    assert not orchestrate.lock_is_stale(
        dict(pid="x"), lambda pid: None
    )  # unreadable: keep


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [(0, 20, 0.0, 0.1684), (20, 20, 0.8316, 1.0), (5, 20, 0.0866, 0.4910)],
)
def test_clopper_pearson_matches_known_values(k, n, lo, hi):
    got = orchestrate.clopper_pearson(k, n)
    assert got[0] == pytest.approx(lo, abs=1e-4) and got[1] == pytest.approx(
        hi, abs=1e-4
    )


def _entries(**changes):
    on = dict(
        verdict=dict(**{"pass": True}),
        draft_rounds=12,
        prompt_end=orchestrate.straddle_target(8192),
        straddle=True,
        response=dict(
            timings=dict(verify_blocks_straddle=3, verify_fallback_reasons={})
        ),
    )
    off = dict(rows=dict(**{"pass": True}), states=dict(**{"pass": True}))
    for key, value in changes.items():
        (off if key in ("rows", "states") else on)[key] = value
    return on, off


@pytest.mark.parametrize("policy", ["joint_v1", None])
def test_e1c_verdict_passes_a_conforming_request(policy):
    on, off = _entries()
    assert orchestrate.e1c_verdict(on, off, threshold=8192, verify_policy=policy)[
        "pass_"
    ]


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (dict(rows=dict(**{"pass": False})), "rows"),
        (dict(states=dict(**{"pass": False})), "states"),
        (dict(verdict=dict(**{"pass": False})), "kernels"),
        (dict(draft_rounds=0), "no MTP rounds"),
        (dict(prompt_end=8000), "prompt_end"),
    ],
)
def test_e1c_verdict_flags_each_failure(change, problem):
    on, off = _entries(**change)
    got = orchestrate.e1c_verdict(on, off, threshold=8192, verify_policy="joint_v1")
    assert not got["pass_"] and any(p.startswith(problem) for p in got["problems"])


def test_e1c_verdict_needs_the_policy_specific_straddle_evidence():
    on, off = _entries(response=dict(timings=dict(verify_blocks_straddle=0)))
    assert not orchestrate.e1c_verdict(
        on, off, threshold=8192, verify_policy="joint_v1"
    )["pass_"]
    on, off = _entries(straddle=False)
    assert not orchestrate.e1c_verdict(on, off, threshold=8192, verify_policy=None)[
        "pass_"
    ]
    on, off = _entries(
        response=dict(
            timings=dict(verify_blocks_straddle=2, verify_fallback_reasons={"x": 1})
        )
    )
    assert not orchestrate.e1c_verdict(
        on, off, threshold=8192, verify_policy="joint_v1"
    )["pass_"]


# ------------------------------------------------------------------ round 5 precedence (Q37)
def _v(pass_=True, c1=True, c2=True):
    return dict(pass_=pass_, c1=dict(identical=c1), c2=dict(identical=c2), problems=[])


def test_pick_outcome_precedence():
    assert orchestrate.pick_outcome({"a": _v(), "b": _v()}) == dict(
        e1c="PASS", run_e2_e7=True, cross_load="identical"
    )
    assert orchestrate.pick_outcome({"a": _v(), "b": _v(c2=False)}) == dict(
        e1c="PASS", run_e2_e7=True, cross_load="unestablished"
    )
    assert orchestrate.pick_outcome({"a": _v(pass_=False), "b": _v()}) == dict(
        e1c="FAIL", run_e2_e7=False, cross_load="identical"
    )
    assert orchestrate.pick_outcome(
        {"a": _v(pass_=False, c1=False), "b": _v()}
    ) == dict(e1c="ABORT", run_e2_e7=False, cross_load="identical")
