"""M76 B2 acceptance 3: ``orchestrate.py --tuple m76-x`` (P369 phase 2 E1c + E2 for the TQ4 ``m72_exact`` arm) (fork-only, dev; CPU).

Written from SPEC v1 (``m76-b2-fork-dev-tools.md``) before the implementation, by a test writer who did not write it.  Each test
carries a ``# class:`` comment: (a) reproduces a failure the specification fixes, (b) preserves an invariant (passes on the
baseline and must keep passing), (c) new behaviour.

The public entry point is driven: ``orchestrate.main([... "--plan-only"])`` with ``preflight`` replaced by a fixed answer (it
probes the host: processes, ports, power).  The stack roots are temp git repos committing the arm registry as the stack does; their
``.venv/bin/python`` answers ``worker_argv`` from mlx-serve's REAL ``_build_command`` and their ``.venv-bench/bin/python`` answers
the C84 reconstruction with 20 fake items (see ``test_m76_b2_fe_common.make_stack_root``).  Nothing starts a worker.

The spec says plan-only "produces the ON / OFF argv pair, the cell list and the item list" without fixing the print format; these
tests read every JSON document the run prints and look for the argv lists (lists of strings holding ``--model``), the integers
(cell thresholds / targets) and the item ids anywhere in it, never at a fixed key.
"""

# isort: skip_file

import copy
import json

import pytest

from mlx_vlm.tests.test_m76_b2_fe_common import (  # also puts dev/ on sys.path
    FIXTURE,
    FORK,
    ITEMS,
    PICK,
    UNIFORM,
    V_TEXT,
    VO_TEXT,
    X_TEXT,
    XO_TEXT,
    kv_text,
    make_stack_root,
    real_command,
    registry_states,
)
from p369_phase2 import (
    orchestrate,
)  # noqa: E402  (dev/ is on sys.path by the import above)

PREFLIGHT = dict(problems=[], power=dict(ac_ok=True, battery_ok=True), free_gb=123.5)
# GOLDEN: what ``main(["--plan-only", ...])`` prints for every existing pick on the baseline (json.dumps(..., indent=1) + "\n")
GOLDEN_PLAN_ONLY = """{
 "preflight": {
  "problems": [],
  "power": {
   "ac_ok": true,
   "battery_ok": true
  },
  "free_gb": 123.5
 }
}
"""
# GOLDEN: sha256 (12 hex) of ``json.dumps(orchestrate.PICKS, sort_keys=True, default=str)`` on the baseline
GOLDEN_PICKS = "cca6c0ef9e28"


class Run:
    def __init__(self, code, out, err):
        self.code, self.out, self.err = code, out, err

    @property
    def ok(self):
        return self.code in (0, None)

    def docs(self):
        text, dec, i, found = self.out, json.JSONDecoder(), 0, []
        while i < len(text):
            if text[i] not in "{[":
                i += 1
                continue
            try:
                obj, end = dec.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            found.append(obj)
            i = end
        return found


@pytest.fixture
def plan(tmp_path, monkeypatch, capsys):
    workdir = tmp_path / "workdir"
    monkeypatch.setenv("STACK_WORKDIR", str(workdir))
    monkeypatch.setattr(
        orchestrate, "preflight", lambda args, out: copy.deepcopy(PREFLIGHT)
    )
    counter = [0]

    def run(*extra, phase="smoke"):
        counter[0] += 1
        out = workdir / "p369" / "phase2" / f"run-{counter[0]}"
        argv = [
            "--fork-root",
            str(FORK),
            "--out",
            str(out),
            "--phase",
            phase,
            "--plan-only",
            *map(str, extra),
        ]
        capsys.readouterr()
        code = None
        try:
            orchestrate.main(argv)
        except SystemExit as exc:
            code = exc.code
        cap = capsys.readouterr()
        return Run(code, cap.out, cap.err)

    run.tmp = tmp_path
    return run


def walk(obj):
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)


def argv_lists(res):
    """Every distinct list of strings holding ``--model`` anywhere in what the run printed."""
    out = []
    for doc in res.docs():
        for node in walk(doc):
            if (
                isinstance(node, list)
                and "--model" in node
                and all(isinstance(a, str) for a in node)
            ):
                if node not in out:
                    out.append(node)
    return out


def ints(res):
    return {n for doc in res.docs() for n in walk(doc) if type(n) is int}


@pytest.fixture
def roots(plan):
    """The X (MTP-ON) and XO arm worktrees of the stack."""
    return make_stack_root(plan.tmp, "x-root", X_TEXT), make_stack_root(
        plan.tmp, "xo-root", XO_TEXT
    )


def tuple_run(plan, on, off, *extra):
    return plan("--tuple", "m76-x", "--stack-root", on, "--off-stack-root", off, *extra)


def served(root):
    """mlx-serve's worker argv (``_build_command(...)[1:]``) of the pick on the root's committed registry."""
    return real_command(root, root / "main_models.yaml", PICK)[1:]


# ================================================================== the tuple table
# class: (c)
def test_tuples_holds_m76_x_exactly_as_specified():
    t = orchestrate.TUPLES["m76-x"]
    assert t["pick"] == PICK and t["kv"] == "tq4" and t["verify_policy"] == "joint_v1"
    assert t["tq4_attention"] == "m72_exact" and t["items_from_kv"] == "native16"
    assert t["drafter"] == orchestrate._DENSE
    assert isinstance(t["production"], str) and 7 <= len(t["production"]) <= 40
    assert set(orchestrate.TUPLES) == {"m76-x"}


# class: (b) (PICKS is byte-identical: one native16 / tq4 tuple per model, no new key, none for the new tuple)
def test_picks_is_unchanged():
    import hashlib

    got = hashlib.sha256(
        json.dumps(orchestrate.PICKS, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    assert got == GOLDEN_PICKS, got
    assert "m76-x" not in orchestrate.PICKS
    assert (
        orchestrate.PICKS[PICK]["kv"] == "native16"
        and orchestrate.PICKS[PICK]["verify_policy"] == "joint_v1"
    )


# class: (b) (the production pin of the existing picks)
def test_production_of_the_existing_picks_is_unchanged():
    assert orchestrate.PRODUCTION == "b26fa7c9"


# ================================================================== the CLI
# class: (c) (--tuple and --pick are mutually exclusive: argparse refuses, exit 2)
def test_tuple_and_pick_are_mutually_exclusive(plan, capsys):
    res = plan(
        "--tuple",
        "m76-x",
        "--pick",
        PICK,
        "--stack-root",
        plan.tmp,
        "--off-stack-root",
        plan.tmp,
    )
    assert (
        not res.ok and "--model" not in res.out
    )  # argparse (exit 2) or a refusal message: the spec fixes neither


# class: (c)
def test_an_unknown_tuple_is_refused(plan, roots):
    on, off = roots
    assert tuple_run(plan, on, off).ok  # control
    res = plan("--tuple", "m76-nope", "--stack-root", on, "--off-stack-root", off)
    assert not res.ok and "--model" not in res.out


# class: (c) (the cross-load OFF needs its own root)
def test_a_tuple_without_an_off_stack_root_is_refused(plan, roots):
    on, off = roots
    assert tuple_run(plan, on, off).ok  # control
    res = plan("--tuple", "m76-x", "--stack-root", on)
    assert not res.ok and "--model" not in res.out


# ================================================================== existing picks: plan-only is byte-identical (golden)
# class: (b)
@pytest.mark.parametrize("pick", sorted(orchestrate.PICKS))
def test_plan_only_of_an_existing_pick_is_byte_identical(plan, pick):
    res = plan("--pick", pick, "--stack-root", plan.tmp)
    assert res.ok and res.out == GOLDEN_PLAN_ONLY, res.out


# class: (b) (the full phase: refused for want of its smoke summary, as today, tuple or not)
def test_plan_only_full_without_a_smoke_summary_is_refused_as_today(plan):
    res = plan(
        "--pick",
        PICK,
        "--stack-root",
        plan.tmp,
        "--reviewed-head",
        "HEAD",
        phase="full",
    )
    assert not res.ok and res.out == ""


# ================================================================== the tuple's plan
# class: (c)
def test_plan_only_for_m76_x_prints_the_on_off_argv_pair(plan, roots):
    on, off = roots
    res = tuple_run(plan, on, off)
    assert res.ok, (res.code, res.err)
    lists = argv_lists(res)
    assert len(lists) == 2, lists
    on_argv = next(a for a in lists if "--draft-kind" in a)
    off_argv = next(a for a in lists if "--draft-kind" not in a)
    # ON from the X registry, OFF from the XO registry, both mlx-serve's own spelling: no overlay, no derivation
    assert on_argv == served(on) and off_argv == served(off)
    flag = lambda argv, f: argv[argv.index(f) + 1] if f in argv else None  # noqa: E731
    assert (
        flag(on_argv, "--mtp-verify-scan") == "joint_v1"
        and flag(on_argv, "--tq4-attention") == "m72_exact"
    )
    assert flag(on_argv, "--draft-kind") == "mtp" and "--draft-model" in on_argv
    assert (
        flag(off_argv, "--tq4-attention") == "m72_exact"
        and flag(off_argv, "--kv-bits") == "4"
    )
    assert "--mtp-verify-scan" not in off_argv and "--draft-model" not in off_argv
    assert (
        flag(on_argv, "--kv-bits") == "4"
        and flag(on_argv, "--kv-quant-scheme") == "turboquant"
    )


# class: (c) (the plan after the preflight; the preflight block is today's)
def test_the_tuple_plan_still_prints_the_preflight_first(plan, roots):
    on, off = roots
    res = tuple_run(plan, on, off)
    assert res.ok, (res.code, res.err)
    assert (
        res.docs()[0]["preflight"] == PREFLIGHT
    )  # the print format is open (Errata v1.1): the preflight is today's block


# class: (c) (the cell list: the native-layer MLX thresholds and the scale cells, sized by the prompt sizer)
@pytest.mark.parametrize("threshold", (1024, 65536))
def test_the_smoke_plan_lists_the_smoke_cells(plan, roots, threshold):
    # AMENDED (delivery 2): a smoke plan lists the smoke cells (T = 1,024 and 65,536: acceptance 5), as P369's smoke always has;
    # the frozen version expected the full-run cell list from a smoke plan, which the spec does not say.
    on, off = roots
    res = tuple_run(plan, on, off)
    assert res.ok, (res.code, res.err)
    sized = {threshold} | {threshold - o for o in orchestrate.STRADDLE_OFFSETS}
    assert sized & ints(res), (threshold, sorted(ints(res))[:40])


# class: (c) (the full-run cell list: the native-layer MLX thresholds, the TQ4 thresholds and the scale cells)
# STRUCTURAL: ``tuple_cells`` is read directly (a full-phase plan needs a passed smoke summary); the spec names no such function.
def test_the_full_cell_list_has_the_specified_thresholds_and_scale_cells():
    cells = orchestrate.tuple_cells("m76-x", smoke=False)
    thresholds = {c["threshold"] for c in cells if isinstance(c.get("threshold"), int)}
    assert {
        1024,
        2048,
        8192,
        32768,
        65536,
    } <= thresholds  # native-layer MLX thresholds + the TQ4 verify threshold
    scale = {
        c["prompt_tokens"]
        for c in cells
        if c.get("threshold") is None and isinstance(c.get("prompt_tokens"), int)
    }
    assert {131072, 260096} <= scale
    for c in cells:
        if isinstance(c.get("threshold"), int):  # sized by the existing prompt sizer
            assert c["prompt_tokens"] == orchestrate.straddle_target(c["threshold"])
    smoke = orchestrate.tuple_cells("m76-x", smoke=True)
    assert {c["threshold"] for c in smoke} == set(orchestrate.SMOKE_THRESHOLDS)


# class: (b) (the existing picks' cell thresholds are untouched)
def test_the_existing_thresholds_are_unchanged():
    assert orchestrate.THRESHOLDS == (1024, 8192, 32768, 65536)
    assert orchestrate.SMOKE_THRESHOLDS == (1024, 65536)


# class: (c) (the item list: the 20 C84 items of the pick, read from its NATIVE16 bundles)
def test_the_plan_lists_the_c84_items_from_the_native16_bundles(plan, roots):
    on, off = roots
    res = tuple_run(plan, on, off)
    assert res.ok, (res.code, res.err)
    for item in ITEMS:
        assert item["id"] in res.out, item
    assert len(ITEMS) == 20


# class: (c) (never the tuple's own kv label: a root whose bundles are only the TQ4 ones has no items; refused)
def test_tq4_bundles_are_no_source_of_the_items(plan, roots):
    import subprocess

    on, off = roots
    assert tuple_run(plan, on, off).ok  # control
    bad = make_stack_root(plan.tmp, "x-tq4-bundles", X_TEXT)
    for path in (bad / "benchmark" / "results" / PICK).glob("*native16*"):
        path.rename(path.with_name(path.name.replace("native16", "tq4")))
    for args in (
        ["add", "-A"],
        [
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "commit",
            "-q",
            "--no-verify",
            "-m",
            "tq4",
        ],
    ):
        subprocess.run(["git", "-C", str(bad), *args], check=True, capture_output=True)
    refused_plan(tuple_run(plan, bad, off))


# class: (c) (no overlay is built; the stack roots stay byte-clean)
def test_the_tuple_plan_builds_no_overlay_and_leaves_the_roots_clean(plan, roots):
    import subprocess

    on, off = roots
    assert tuple_run(plan, on, off).ok
    assert not list(plan.tmp.rglob("overlay-*"))
    for root in roots:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
        ).stdout
        assert status == "", status


# class: (c) (the all-OFF-root rule and ``classify`` are not applied to a tuple run: X and XO are not among the 16 states)
def test_the_tuple_roots_are_not_sixteen_state_registries(roots):
    for root in roots:
        text = (root / "main_models.yaml").read_text()
        assert registry_states.classify(text, FIXTURE) is None


# ================================================================== refusals
def refused_plan(res):
    assert not res.ok, res.out
    assert "--model" not in res.out  # no argv was planned


STACK_ROOT_CASES = {
    "xo-as-on": XO_TEXT,  # the pick is MTP-OFF: nothing to run ON
    "v-commit": V_TEXT,  # a mode of another tuple
    "fixture-native16": FIXTURE,
    "af4": kv_text(PICK, 4, "uniform"),
    "tq4-no-mode": kv_text(PICK, 4, "turboquant"),
    "mode-with-tq8": kv_text(PICK, 8, "turboquant", mode="m72_exact"),
    "unknown-mode": kv_text(PICK, 4, mode="m72_v9"),
    "false-mode": kv_text(PICK, 4, mode="false"),
    "x-plus-step": X_TEXT.replace(
        "    prefill_step_size: 512", "    prefill_step_size: 1024", 1
    ),
    "mode-on-uniform": kv_text(UNIFORM, 4, "turboquant", mode="m72_exact"),
}


# class: (c) (a stack root that is not an X commit is refused)
@pytest.mark.parametrize("label", sorted(STACK_ROOT_CASES))
def test_a_stack_root_that_is_not_an_x_commit_is_refused(plan, roots, label):
    on, off = roots
    assert tuple_run(plan, on, off).ok  # control
    text = STACK_ROOT_CASES[label]
    assert text != X_TEXT
    bad = make_stack_root(plan.tmp, f"bad-{label}", text)
    refused_plan(tuple_run(plan, bad, off))


OFF_ROOT_CASES = {
    "x-as-off": X_TEXT,  # the pick still serves MTP
    "fixture-native16": FIXTURE,
    "entry-off-native16": registry_states.apply(
        FIXTURE, {PICK}
    ),  # MTP OFF but not the X arm
    "v-off": VO_TEXT,  # MTP OFF, the other mode
    "xo-plus-step": XO_TEXT.replace(
        "    prefill_step_size: 512", "    prefill_step_size: 1024", 1
    ),
    "xo-without-mode": kv_text(
        PICK, 4, "turboquant", base=registry_states.apply(FIXTURE, {PICK})
    ),
    "af4-off": kv_text(PICK, 4, "uniform", base=registry_states.apply(FIXTURE, {PICK})),
}


# class: (c) (an off root that is not an XO commit is refused)
@pytest.mark.parametrize("label", sorted(OFF_ROOT_CASES))
def test_an_off_root_that_is_not_an_xo_commit_is_refused(plan, roots, label):
    on, off = roots
    assert tuple_run(plan, on, off).ok  # control
    text = OFF_ROOT_CASES[label]
    assert text != XO_TEXT
    bad = make_stack_root(plan.tmp, f"bad-off-{label}", text)
    refused_plan(tuple_run(plan, on, bad))


# class: (c) (the same worktree as ON and OFF cannot be both)
def test_the_same_root_for_on_and_off_is_refused(plan, roots):
    on, off = roots
    assert tuple_run(plan, on, off).ok  # control
    refused_plan(tuple_run(plan, on, on))
    refused_plan(tuple_run(plan, off, off))


# class: (c) (the two roots exchanged)
def test_the_roots_exchanged_are_refused(plan, roots):
    on, off = roots
    control = tuple_run(plan, on, off)
    assert control.ok, (control.code, control.err[-800:], control.out[-800:])  # control
    refused_plan(tuple_run(plan, off, on))
