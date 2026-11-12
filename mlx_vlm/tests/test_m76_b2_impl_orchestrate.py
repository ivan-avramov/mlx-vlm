"""M76 B2 acceptance 3: the P369 campaign tuple ``m76-x`` (ON = the X arm commit, cross-load OFF = the XO arm commit) —
``--plan-only``, ``Campaign.prepare``, the jobs, the cells, the verdicts — and the worker's TQ4 additions (CPU only).

The arm worktrees are temp git repos committing the arm registries of ``test_m76_b2_impl_arms`` (the B1 edit model on the
pinned fixture); mlx-serve's ``_build_command`` is ``test_m76_b2_impl_qualify.serve_argv`` (pinned to its real output).
"""

import argparse
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import orchestrate, registry_states, worker  # noqa: E402

from mlx_vlm.tests import test_m76_b2_impl_arms as arms  # noqa: E402
from mlx_vlm.tests import test_m76_b2_impl_qualify as q  # noqa: E402

PICK = arms.PICK
FIXTURE = arms.FIXTURE
TUPLE = "m76-x"
FULL_CELLS = [
    "T1024",
    "T2048",
    "T8192",
    "T32768",
    "T65536",
    "S131072",
    "S260096",
    "short",
]
ITEMS = [f"{axis}/{i}" for axis in orchestrate.C84_AXES for i in range(5)]


def _git(cwd, *a):
    return subprocess.run(
        ["git", "-C", str(cwd), *a], capture_output=True, text=True, check=True
    ).stdout.strip()


def worktree(path, text, *, bundles=True):
    """An arm worktree: a git commit holding ``main_models.yaml`` (and the pick's native16 C84 bundles)."""
    path.mkdir(parents=True)
    _git(path.parent, "init", "-q", str(path))
    (path / "main_models.yaml").write_bytes(text.encode("utf-8"))
    if bundles:
        results = path / "benchmark" / "results" / PICK
        results.mkdir(parents=True)
        for axis in orchestrate.C84_AXES:
            rows = [
                dict(id=i, sampler_seed=100 + i, prompt_tokens=200 + i)
                for i in range(5)
            ]
            (results / f"{axis}.c84bundle-after-native16-20260914.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
    _git(path, "add", "-A")
    _git(
        path,
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "user.name=t",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "--no-verify",
        "-m",
        "arm",
    )
    return path


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """X and XO worktrees, mlx-serve's argv spelling, the host's preflight mocked."""
    built = []

    def worker_argv(stack_root, registry, pick):
        text = Path(registry).read_bytes().decode("utf-8")
        built.append((Path(stack_root).name, Path(registry).parent.name))
        return q.serve_argv(registry_states.settings(text)[pick], "/s")[1:]

    monkeypatch.setattr(orchestrate, "worker_argv", worker_argv)
    monkeypatch.setattr(
        orchestrate,
        "preflight",
        lambda args, out: dict(problems=[], power={}, free_gb=99.0),
    )
    monkeypatch.setenv("STACK_WORKDIR", str(tmp_path / "wd"))
    trees = {
        arm: worktree(tmp_path / arm, arms.arm_text(arm)) for arm in ("X", "XO", "V")
    }
    trees["N16"] = worktree(tmp_path / "N16", FIXTURE)
    return tmp_path, trees, built


def plan(tmp_path, stack, off, *extra, phase="smoke"):
    argv = [
        "--stack-root",
        str(stack),
        "--fork-root",
        str(DEV.parent),
        "--out",
        str(tmp_path / "wd" / "p369" / "phase2" / "run"),
        "--phase",
        phase,
        "--plan-only",
        *extra,
    ]
    if off is not None:
        argv += ["--off-stack-root", str(off)]
    out = io.StringIO()
    with redirect_stdout(out):
        orchestrate.main(argv)
    return json.loads(out.getvalue())


def refused_plan(tmp_path, stack, off, *extra):
    with pytest.raises(SystemExit) as exc:
        plan(tmp_path, stack, off, *extra)
    assert exc.value.code not in (0, None)
    return str(exc.value.code)


# ================================================================== --plan-only
def test_plan_only_for_the_tuple_gives_the_argv_pair_cells_and_items(roots):
    tmp_path, trees, built = roots
    got = plan(tmp_path, trees["X"], trees["XO"], "--tuple", TUPLE)
    p = got["plan"]
    assert got["preflight"] == dict(problems=[], power={}, free_gb=99.0)
    assert (
        p["tuple"] == TUPLE and p["pick"] == PICK and p["tq4_attention"] == "m72_exact"
    )
    assert p["production"] == orchestrate.M77_PRODUCTION
    assert p["argv_on"] == q.REAL_ARGV["X"] and p["argv_off"] == q.REAL_ARGV["XO"]
    assert built == [
        ("X", "X"),
        ("XO", "XO"),
    ]  # ON from the X commit, OFF from the XO commit, no overlay
    assert p["stack_head"] == _git(trees["X"], "rev-parse", "HEAD")
    assert p["off_stack_head"] == _git(trees["XO"], "rev-parse", "HEAD")
    assert [c["name"] for c in p["cells"]] == ["T1024", "T65536"]  # the smoke's
    assert [i["name"] for i in p["items"]] == ITEMS
    assert p["items"][0] == dict(name="math500/0", seed=100, prompt_tokens=200)
    assert p["e2"] == ITEMS[:2]


def test_plan_only_full_has_the_tq4_and_native_thresholds_and_the_scale_cells(
    roots, monkeypatch
):
    tmp_path, trees, _ = roots
    monkeypatch.setattr(orchestrate, "smoke_authorization_problems", lambda s, **k: [])
    monkeypatch.setattr(orchestrate, "reviewed_head_problems", lambda phase, r, h: [])
    smoke = tmp_path / "smoke.json"
    smoke.write_text("{}")
    p = plan(
        tmp_path,
        trees["X"],
        trees["XO"],
        "--tuple",
        TUPLE,
        "--smoke-summary",
        str(smoke),
        phase="full",
    )["plan"]
    cells = {c["name"]: c for c in p["cells"]}
    assert list(cells) == FULL_CELLS
    assert cells["T2048"] == dict(
        name="T2048", threshold=2048, prompt_tokens=1984, route="tq4"
    )
    assert cells["T65536"]["route"] == "mlx" and cells["T1024"]["prompt_tokens"] == 960
    assert cells["S131072"] == dict(
        name="S131072", threshold=None, prompt_tokens=131072, route="scale"
    )
    assert cells["S260096"]["prompt_tokens"] == 260096
    assert p["e2"] == ITEMS + FULL_CELLS[:-1]


@pytest.mark.parametrize(
    "on, off, why",
    (
        ("XO", "XO", "--stack-root: the registry does not serve"),
        ("V", "XO", "is not the m76-x arm"),
        ("N16", "XO", "is not the m76-x arm"),
        ("X", "X", "--off-stack-root: the registry does not serve"),
        ("X", "V", "--off-stack-root"),
        ("X", "N16", "--off-stack-root"),
    ),
    ids=("on-is-XO", "on-is-V", "on-is-N16", "off-is-X", "off-is-V", "off-is-N16"),
)
def test_plan_only_refuses_roots_that_are_not_x_and_xo(roots, on, off, why):
    tmp_path, trees, built = roots
    text = refused_plan(tmp_path, trees[on], trees[off], "--tuple", TUPLE)
    assert why in text, text
    assert built == []  # refused before any argv is built


def test_plan_only_refuses_an_off_root_that_differs_beyond_the_mtp_keys(roots):
    tmp_path, trees, _ = roots
    other = registry_states.apply(FIXTURE, {"Ornith-1.0-35B-mlx-uniform-4bit"})
    xo = worktree(tmp_path / "XO2", arms.arm_text("XO", base=other))
    assert "beyond" in refused_plan(tmp_path, trees["X"], xo, "--tuple", TUPLE)


def test_plan_only_refuses_an_off_root_that_is_mtp_off_without_the_kv_change(roots):
    tmp_path, trees, _ = roots
    off = worktree(tmp_path / "OFF16", registry_states.apply(FIXTURE, {PICK}))
    assert "--off-stack-root" in refused_plan(
        tmp_path, trees["X"], off, "--tuple", TUPLE
    )


def test_plan_only_refuses_an_uncommitted_registry(roots):
    tmp_path, trees, _ = roots
    (trees["X"] / "main_models.yaml").write_bytes(
        ((trees["X"] / "main_models.yaml").read_bytes() + b"# edited\n")
    )
    assert "not committed as is" in refused_plan(
        tmp_path, trees["X"], trees["XO"], "--tuple", TUPLE
    )


def test_plan_only_refuses_an_off_root_that_is_not_a_git_commit(roots, monkeypatch):
    tmp_path, trees, _ = roots
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "main_models.yaml").write_text(arms.arm_text("XO"))
    monkeypatch.setenv(
        "GIT_CEILING_DIRECTORIES", str(tmp_path)
    )  # no enclosing repository is found
    assert "not a git commit" in refused_plan(
        tmp_path, trees["X"], loose, "--tuple", TUPLE
    )


def test_plan_only_refuses_a_bad_tuple_selection(roots):
    tmp_path, trees, _ = roots
    assert "mutually exclusive" in refused_plan(
        tmp_path, trees["X"], trees["XO"], "--tuple", TUPLE, "--pick", PICK
    )
    assert "--off-stack-root" in refused_plan(
        tmp_path, trees["X"], None, "--tuple", TUPLE
    )
    assert "needs --tuple" in refused_plan(
        tmp_path, trees["X"], trees["XO"], "--pick", PICK
    )
    with pytest.raises(SystemExit):  # argparse: an unknown tuple
        plan(tmp_path, trees["X"], trees["XO"], "--tuple", "m76-v")


def test_plan_only_refuses_when_a_bundle_is_missing(roots):
    tmp_path, trees, _ = roots
    x = worktree(tmp_path / "Xnb", arms.arm_text("X"), bundles=False)
    assert "C84 bundle" in refused_plan(tmp_path, x, trees["XO"], "--tuple", TUPLE)


# ================================================================== the argv rule
def test_the_tuple_argv_rule():
    on, off = list(q.REAL_ARGV["X"]), list(q.REAL_ARGV["XO"])
    assert orchestrate.tuple_argv_problems(TUPLE, on, off) == []
    assert orchestrate.strip_mtp_flags(on) == off
    # ON without the scan / mode / drafter, OFF with a scan or a drafter, OFF without the mode, OFF otherwise different
    cases = {
        "scan": (
            orchestrate.strip_mtp_flags(on)
            + ["--draft-kind", "mtp", "--draft-model", "d"],
            off,
        ),
        "mode-on": ([a for a in on if a not in ("--tq4-attention", "m72_exact")], off),
        "v-mode": (list(q.REAL_ARGV["V"]), off),
        "drafter-off": (on, off + ["--draft-kind", "mtp"]),
        "scan-off": (on, off + ["--mtp-verify-scan", "joint_v1"]),
        "mode-off": (on, [a for a in off if a not in ("--tq4-attention", "m72_exact")]),
        "other-off": (on, [a if a != "512" else "1024" for a in off]),
    }
    for label, (a, b) in cases.items():
        assert orchestrate.tuple_argv_problems(TUPLE, a, b), label


def test_the_tuple_table():
    spec = orchestrate.TUPLES[TUPLE]
    assert spec == dict(
        pick=PICK,
        kv="tq4",
        verify_policy="joint_v1",
        tq4_attention="m72_exact",
        drafter=orchestrate._DENSE,
        items_from_kv="native16",
        production=orchestrate.M77_PRODUCTION,
    )
    assert orchestrate.tuple_thresholds(TUPLE) == (1024, 2048, 8192, 32768, 65536)
    assert orchestrate.tuple_selection_problems(TUPLE, None, "/x") == []
    assert (
        PICK in orchestrate.PICKS and orchestrate.PICKS[PICK]["kv"] == "native16"
    )  # PICKS unchanged


# ================================================================== Campaign.prepare / jobs under the tuple
def campaign(tmp_path, trees, monkeypatch, *, phase="smoke"):
    args = argparse.Namespace(
        stack_root=str(trees["X"]),
        off_stack_root=str(trees["XO"]),
        fork_root=str(DEV.parent),
        out=str(tmp_path / "run"),
        pick=None,
        tuple=TUPLE,
        phase=phase,
        wall_limit_s=None,
        reviewed_head=None,
        smoke_summary=None,
    )
    for name, value in dict(
        stack_bench_problems=lambda *a, **k: [],
        stack_bench_state=lambda *a, **k: dict(tree="t", bundles={}),
        fork_versions=lambda *a, **k: {"mlx": "0.32.4"},
        weights_record=lambda m: dict(hf_path=m, revision="r", files={"a": "b"}),
        pins_problems=lambda *a, **k: [],
    ).items():
        monkeypatch.setattr(orchestrate, name, value)
    return orchestrate.Campaign(args)


def test_prepare_builds_on_and_off_from_the_two_commits_without_classify_or_overlay(
    roots, monkeypatch
):
    tmp_path, trees, built = roots

    def no(*a, **k):
        raise AssertionError("not under a tuple")

    monkeypatch.setattr(orchestrate.registry_states, "classify", no)
    monkeypatch.setattr(orchestrate, "overlay_registry", no)
    monkeypatch.setattr(orchestrate, "registry_off_problems", no)
    camp = campaign(tmp_path, trees, monkeypatch)
    manifest = camp.prepare()
    assert built == [("X", "X"), ("XO", "XO")]
    assert camp.arms[PICK] == dict(on=q.REAL_ARGV["X"], off=q.REAL_ARGV["XO"])
    assert (
        manifest["tuple"] == TUPLE
        and manifest["production"] == orchestrate.M77_PRODUCTION
    )
    assert manifest["stack_head"] == _git(trees["X"], "rev-parse", "HEAD")
    assert manifest["off_stack_head"] == _git(trees["XO"], "rev-parse", "HEAD")
    assert manifest["registry_off"] == [] and manifest["off_registry_off"] == [PICK]
    entry = manifest["picks"][PICK]
    assert entry["overlay_sha256"] is None and entry["restored"] == []
    assert entry["registry_on_sha256"] == orchestrate.sha256_file(
        trees["X"] / "main_models.yaml"
    )
    assert entry["registry_off_sha256"] == orchestrate.sha256_file(
        trees["XO"] / "main_models.yaml"
    )
    assert not list(camp.out.glob("overlay-*"))
    assert json.loads((camp.out / "manifest.json").read_text()) == manifest
    bound = orchestrate.binding(manifest)
    assert bound["tuple"] == TUPLE and bound["stack_head"] == manifest["stack_head"]
    assert set(bound["picks"][PICK]) >= {"registry_on_sha256", "registry_off_sha256"}
    assert camp.summary["tuple"] == TUPLE
    # a smoke bound to another X commit (or XO registry) does not authorize this full run
    reviewed = dict(manifest, reviewed_head="r")
    smoke = orchestrate.binding(reviewed)
    assert orchestrate.binding_problems(dict(binding=smoke), reviewed) == []
    moved = dict(binding=dict(smoke, stack_head="t" * 40))
    assert orchestrate.binding_problems(moved, reviewed) == [
        f"stack_head: smoke {'t' * 40!r} != now {manifest['stack_head']!r}"
    ]
    other = json.loads(json.dumps(smoke))
    other["picks"][PICK]["registry_off_sha256"] = "0" * 64
    assert orchestrate.binding_problems(dict(binding=other), reviewed) == [
        f"{PICK} registry_off_sha256 differs from the smoke"
    ]


def test_prepare_refuses_swapped_roots(roots, monkeypatch):
    tmp_path, trees, _ = roots
    trees = dict(trees, X=trees["XO"], XO=trees["X"])
    with pytest.raises(orchestrate.Refused, match="does not serve"):
        campaign(tmp_path, trees, monkeypatch).prepare()


def test_tuple_jobs_expect_the_mode_on_both_arms(roots, monkeypatch):
    tmp_path, trees, _ = roots
    camp = campaign(tmp_path, trees, monkeypatch)
    camp.prepare()
    on = camp.job(PICK, "j", "e1c_pair", "on", [dict(name="r", timeout_s=1)])
    off = camp.job(PICK, "j", "e1c_off", "off", [dict(name="r", timeout_s=1)])
    assert (
        on["expect"]["tq4_attention"] == off["expect"]["tq4_attention"] == "m72_exact"
    )
    assert (
        on["expect"]["verify_policy"] == "joint_v1"
        and off["expect"]["verify_policy"] is None
    )
    assert on["argv"] == q.REAL_ARGV["X"] and off["argv"] == q.REAL_ARGV["XO"]


def test_the_campaign_refuses_a_tuple_with_a_pick_or_without_an_off_root(tmp_path):
    for pick, off in (([PICK], "/x"), (None, None)):
        args = argparse.Namespace(
            stack_root="s",
            fork_root="f",
            out=str(tmp_path / "o"),
            pick=pick,
            tuple=TUPLE,
            off_stack_root=off,
            phase="smoke",
            wall_limit_s=None,
        )
        with pytest.raises(SystemExit, match="refused"):
            orchestrate.Campaign(args)


# ================================================================== run_pick / e1c / e2 under the tuple
def stub_camp(tuple_name):
    camp = object.__new__(orchestrate.Campaign)
    camp.stack, camp.bench_paths = None, []
    camp.arms = {PICK: dict(on=list(q.REAL_ARGV["X"]), off=list(q.REAL_ARGV["XO"]))}
    if tuple_name is not None:
        camp.tuple, camp.tq4 = tuple_name, "m72_exact"
    return camp


@pytest.mark.parametrize("smoke", (True, False))
def test_run_pick_sizes_the_tuple_cells_and_hands_e2_the_scale_prompts(
    monkeypatch, smoke
):
    camp = stub_camp(TUPLE)
    seen = {}
    monkeypatch.setattr(orchestrate, "c84_items", lambda stack, pick: [])
    camp.check_bench = lambda pick: None
    camp.size = lambda pick, thresholds: seen.update(sized=thresholds) or {
        t: f"p{t}" for t in thresholds
    }
    camp.size_scale = lambda pick, totals: seen.update(scale=totals) or {
        f"S{n}": f"s{n}" for n in totals
    }
    camp.check_reconstruction = lambda pick, items, panel=None: {}

    def e1c(pick, prompts, *, smoke, scale=None):
        seen.update(e1c=sorted(prompts), e1c_scale=scale)
        return {"T1024": dict(pass_=True)}

    camp.e1c = e1c
    monkeypatch.setattr(
        orchestrate, "pick_outcome", lambda v: dict(e1c="PASS", run_e2_e7=True)
    )
    camp.e2 = lambda pick, items, prompts, smoke: seen.update(e2=prompts) or dict(
        verdict="PASS"
    )
    camp.e7a = lambda pick, items: dict(status="OBSERVED")
    camp.e7b = lambda pick, items, sessions=5: dict(status="OBSERVED", sessions=[])
    camp.summary = dict(picks={})
    camp.write_summary = lambda: None
    camp.run_pick(PICK, smoke=smoke)
    if smoke:
        assert seen["sized"] == orchestrate.SMOKE_THRESHOLDS
        assert "scale" not in seen and seen["e1c_scale"] is None
    else:
        assert seen["sized"] == (1024, 2048, 8192, 32768, 65536)
        assert seen["scale"] == orchestrate.SCALE_CELLS
        assert seen["e1c_scale"] == {"S131072": "s131072", "S260096": "s260096"}
        assert seen["e2"]["S260096"] == "s260096" and seen["e2"][2048] == "p2048"


def _cell_camp(tmp_path, names, *, engaged=True, straddle_counter=True):
    """``e1c`` on prepared req files: one ON entry per cell, all valid, the C2 entries passing."""
    camp = stub_camp(TUPLE)
    camp.out = tmp_path
    camp.job = lambda pick, name, mode, arm, requests: name
    camp.run = lambda name: None
    pair = tmp_path / PICK / "e1c-pair-a0"
    c2 = tmp_path / PICK / "e1c-c2"
    pair.mkdir(parents=True)
    c2.mkdir(parents=True)
    for i, (key, tokens) in enumerate(names):
        on = dict(
            final=dict(token_ids=[1] * 70),
            straddle=isinstance(key, int),
            verdict={"pass": True},
            draft_rounds=5,
            prompt_end=(
                orchestrate.straddle_target(key) if isinstance(key, int) else tokens
            ),
            response=dict(
                timings=dict(
                    verify_blocks_straddle=1 if straddle_counter else 0,
                    verify_fallback_reasons={},
                )
            ),
            pair=dict(
                rows={"pass": True}, states={"pass": True}, forced={"pass": True}
            ),
            c1=dict(acceptance={"pass": True, "problems": []}),
            tq4_engagement=dict(
                ok=engaged, problems=[] if engaged else ["d2_verify did not move"]
            ),
        )
        (pair / f"req-{i}.json").write_text(json.dumps(on))
        (c2 / f"req-{i}.json").write_text(
            json.dumps(
                dict(rows={"pass": True}, states={"pass": True}, forced={"pass": True})
            )
        )
    return camp


def test_e1c_runs_the_scale_cells_as_non_straddle_cells(tmp_path):
    camp = _cell_camp(tmp_path, [(1024, None), (2048, None), ("S131072", 131072)])
    got = camp.e1c(PICK, {1024: "a", 2048: "b"}, smoke=True, scale={"S131072": "c"})
    assert list(got) == ["T1024", "T2048", "S131072"]
    assert all(v["pass_"] for v in got.values()), got
    assert camp.accepted_prompts[PICK] == {1024: "a", 2048: "b", "S131072": "c"}


def test_e1c_tuple_cells_need_the_tq4_engagement(tmp_path):
    camp = _cell_camp(tmp_path, [(1024, None), ("S131072", 131072)], engaged=False)
    got = camp.e1c(PICK, {1024: "a"}, smoke=True, scale={"S131072": "c"})
    for v in got.values():
        assert v["pass_"] is False and v["neutral"] is True
        assert any("tq4 engagement" in p for p in v["evidence_problems"])
    assert orchestrate.pick_outcome(got)["e1c"] == "INCOMPLETE"


def _on_entry(*, valid=True, engaged=True, neutral=True, c1=True):
    return dict(
        final=dict(token_ids=[1] * 70),
        straddle=valid,
        verdict={"pass": True},
        draft_rounds=5,
        prompt_end=orchestrate.straddle_target(1024),
        response=dict(
            timings=dict(verify_blocks_straddle=1, verify_fallback_reasons={})
        ),
        pair=dict(rows={"pass": neutral}, states={"pass": True}, forced={"pass": True}),
        c1=dict(acceptance={"pass": c1, "problems": [] if c1 else ["c1"]}),
        tq4_engagement=dict(
            ok=engaged,
            problems=[] if engaged else ["d2_verify did not move"],
            per_query_short=0,
        ),
    )


def _attempt_camp(tmp_path, attempts, *, tuple_name=TUPLE):
    """``e1c`` on one T1024 cell whose attempts a0, a1, ... are ``attempts`` (``_on_entry`` keyword sets); the last is
    the valid one, every earlier one invalid (no straddle) and so rejected."""
    camp = stub_camp(tuple_name)
    camp.out = tmp_path
    camp.job = lambda pick, name, mode, arm, requests: name
    camp.run = lambda name: None
    for a, changes in enumerate(attempts):
        d = tmp_path / PICK / f"e1c-pair-a{a}"
        d.mkdir(parents=True)
        entry = _on_entry(valid=a == len(attempts) - 1, **changes)
        if tuple_name is None:
            entry.pop("tq4_engagement")
        (d / "req-0.json").write_text(json.dumps(entry))
    c2 = tmp_path / PICK / "e1c-c2"
    c2.mkdir(parents=True)
    (c2 / "req-0.json").write_text(
        json.dumps(
            dict(rows={"pass": True}, states={"pass": True}, forced={"pass": True})
        )
    )
    return camp


def test_a_rejected_attempts_engagement_failure_makes_e1c_incomplete(tmp_path):
    camp = _attempt_camp(tmp_path, [dict(engaged=False), dict()])
    got = camp.e1c(PICK, {1024: "a"}, smoke=True)
    cell = got["T1024"]
    assert cell["pass_"] is True  # the accepted attempt itself passes...
    (rejected,) = cell["rejected"]
    assert rejected["tq4_engaged"] is False  # ...the rejected one's evidence is kept
    assert rejected["tq4_engagement"]["problems"] == ["d2_verify did not move"]
    assert [a["tq4_engaged"] for a in cell["attempts"]] == [False, True]
    assert orchestrate.pick_outcome(got)["e1c"] == "INCOMPLETE"


def test_engaged_rejected_attempts_leave_e1c_passing(tmp_path):
    camp = _attempt_camp(tmp_path, [dict(), dict(), dict()])
    got = camp.e1c(PICK, {1024: "a"}, smoke=True)
    assert [a["tq4_engaged"] for a in got["T1024"]["rejected"]] == [True, True]
    assert orchestrate.pick_outcome(got)["e1c"] == "PASS"


@pytest.mark.parametrize(
    "first, outcome",
    (
        (dict(engaged=False, c1=False), "ABORT"),
        (dict(engaged=False, neutral=False), "FAIL"),
    ),
    ids=("c1-first", "neutrality-first"),
)
def test_the_engagement_of_rejected_attempts_comes_after_c1_and_neutrality(
    tmp_path, first, outcome
):
    camp = _attempt_camp(tmp_path, [first, dict()])
    assert (
        orchestrate.pick_outcome(camp.e1c(PICK, {1024: "a"}, smoke=True))["e1c"]
        == outcome
    )


def test_without_a_tuple_attempts_carry_no_engagement(tmp_path):
    camp = _attempt_camp(tmp_path, [dict(), dict()], tuple_name=None)
    got = camp.e1c(PICK, {1024: "a"}, smoke=True)
    assert all("tq4_engaged" not in a for a in got["T1024"]["attempts"])
    assert orchestrate.pick_outcome(got)["e1c"] == "PASS"


def test_a_malformed_rejected_engagement_flag_raises():
    v = dict(
        pass_=True,
        neutral=True,
        evidence_ok=True,
        c1=dict(identical=True),
        c2=dict(identical=True),
        rejected=[dict(neutral=True, c1=True, tq4_engaged="no")],
    )
    with pytest.raises(orchestrate.MalformedEvidence, match="tq4_engaged"):
        orchestrate.pick_outcome({"T1024": v})


def _reasons_verdict(reasons, short, *, tq4=True):
    on = _on_entry()
    on["response"]["timings"]["verify_fallback_reasons"] = reasons
    on["tq4_engagement"]["per_query_short"] = short
    if not tq4:
        on.pop("tq4_engagement")
    return orchestrate.e1c_verdict(
        on, on["pair"], threshold=1024, verify_policy="joint_v1", tq4=tq4
    )["evidence_problems"]


def test_the_cache_reason_is_accepted_only_as_per_query_shorts_count():
    assert _reasons_verdict({"cache": 30}, 30) == []
    assert _reasons_verdict({}, 30) == []
    for reasons, short in (
        ({"cache": 31}, 30),
        ({"cache": 29}, 30),
        ({"cache": 30, "domain": 1}, 30),
        ({"domain": 1}, 0),
        ({"cache": 1}, 0),
    ):
        assert _reasons_verdict(reasons, short) == [
            f"fallback reasons {reasons} (per_query:short delta {short})"
        ], reasons
    # non-tuple runs: today's rule, every reason fails
    assert _reasons_verdict({"cache": 30}, 30, tq4=False) == [
        "fallback reasons {'cache': 30}"
    ]
    with pytest.raises(orchestrate.MalformedEvidence, match="per_query:short"):
        _reasons_verdict({"cache": 3}, "3")


# ------------------------------------------------------------------ the tuple's smoke projection (Sol P6)
STEPS = dict(prep=600.0, e1c=900.0, e2=300.0, e7b=240.0)


def test_the_tuple_projection_reads_the_full_tuple_cells():
    full = [
        c["prompt_tokens"]
        for c in orchestrate.tuple_cells(TUPLE, smoke=False)
        if c["prompt_tokens"] is not None
    ]
    assert full == [
        orchestrate.straddle_target(t) for t in (1024, 2048, 8192, 32768, 65536)
    ] + [131072, 260096]
    smoke = sum(orchestrate.straddle_target(t) for t in orchestrate.SMOKE_THRESHOLDS)
    e1c = STEPS["e1c"] * sum(full) / smoke
    want = (
        STEPS["prep"] * 1.5 + 2 * e1c + STEPS["e2"] * 10 + STEPS["e7b"] * (20 / 3 + 5)
    )
    got = orchestrate.tuple_phase_projection(STEPS, TUPLE)
    assert got["nominal_s"] == pytest.approx(want)
    # far more than the native16 projection, which knows neither T2048 nor the scale cells
    assert got["nominal_s"] > 2 * orchestrate.phase_projection(STEPS)["nominal_s"]


def _gate_picks(steps=STEPS):
    return {
        PICK: dict(outcome=dict(e1c="PASS"), e1c={}, step_s=dict(steps), wall_s=1.0)
    }


def test_the_tuple_smoke_gate_projects_the_tuple_and_fits_eight_hours():
    plain = orchestrate.smoke_gate(_gate_picks())
    gate = orchestrate.smoke_gate(_gate_picks(), tuple_name=TUPLE)
    assert "tuple" not in plain and "limit_s" not in plain  # non-tuple: today's gate
    assert plain["projected_full_s"] == orchestrate.phase_projection(STEPS)["nominal_s"]
    projected = orchestrate.tuple_phase_projection(STEPS, TUPLE)["nominal_s"]
    assert gate["projected_full_s"] == projected
    assert gate["tuple"] == TUPLE and gate["limit_s"] == 8 * 3600
    assert gate["fits"] is (projected <= 8 * 3600)


def _smoke_summary(steps, tuple_name=TUPLE):
    gate = orchestrate.smoke_gate(_gate_picks(steps), tuple_name=tuple_name)
    status = (
        ("SMOKE_PASS", 0) if gate["pass_"] and gate["fits"] else ("SMOKE_NOT_PASS", 2)
    )
    return (
        dict(
            phase="smoke",
            tuple=tuple_name,
            status=status[0],
            exit_code=status[1],
            smoke_gate=gate,
            picks=_gate_picks(steps),
        ),
        gate,
    )


@pytest.fixture
def gate_ok(monkeypatch):
    """The smoke gate's evidence checks pass on the stub pick (only the projection decides)."""
    real = orchestrate.smoke_gate

    def gate(picks, tuple_name=None):
        out = real(picks, tuple_name=tuple_name)
        return dict(out, problems=[], pass_=True)

    monkeypatch.setattr(orchestrate, "smoke_gate", gate)


def test_a_tuple_smoke_over_eight_hours_needs_the_operators_approval(gate_ok):
    long = dict(STEPS, e1c=3600.0)
    summary, gate = _smoke_summary(long)
    hours = gate["projected_full_s"] / 3600
    assert gate["fits"] is False and hours > 8
    auth = orchestrate.smoke_authorization_problems
    assert any("did not pass / fit" in p for p in auth(summary))  # refused without
    assert auth(summary, operator_approved_hours=hours + 0.01) == []
    short_of = auth(summary, operator_approved_hours=hours - 0.5)
    assert any("< the projected full run" in p for p in short_of), short_of


def test_a_tuple_smoke_within_eight_hours_needs_no_approval(gate_ok):
    summary, gate = _smoke_summary(dict(STEPS, e1c=750.0))
    # past the native16 3.5 h limit, within the tuple's 8 h: authorized as a passing smoke
    assert orchestrate.ESTIMATE_LIMIT_S < gate["projected_full_s"] <= 8 * 3600
    assert gate["fits"] is True and summary["status"] == "SMOKE_PASS"
    assert orchestrate.smoke_authorization_problems(summary) == []


def test_the_approval_is_for_tuple_runs_only(gate_ok):
    summary, _ = _smoke_summary(dict(STEPS, e1c=3600.0))
    summary.pop("tuple")
    summary["smoke_gate"] = orchestrate.smoke_gate(summary["picks"])
    got = orchestrate.smoke_authorization_problems(
        summary, operator_approved_hours=99.0
    )
    assert got == ["--operator-approved-hours approves a campaign-tuple run only"]


@pytest.mark.parametrize(
    "argv",
    (
        ["--phase", "full", "--pick", PICK, "--operator-approved-hours", "9"],
        [
            "--phase",
            "smoke",
            "--tuple",
            TUPLE,
            "--off-stack-root",
            "/o",
            "--operator-approved-hours",
            "9",
        ],
    ),
    ids=("no-tuple", "smoke-phase"),
)
def test_main_refuses_the_approval_outside_a_tuple_full_run(tmp_path, argv):
    base = ["--stack-root", "/s", "--fork-root", "/f", "--out", str(tmp_path / "o")]
    with pytest.raises(
        SystemExit, match="--operator-approved-hours needs --tuple --phase full"
    ):
        orchestrate.main(base + argv)


def test_the_tuple_projection_needs_the_full_cells_not_the_native_thresholds():
    """A smoke projected from the native16 thresholds alone would under-count the tuple's full run."""
    tokens = sum(
        c["prompt_tokens"]
        for c in orchestrate.tuple_cells(TUPLE, smoke=False)
        if c["prompt_tokens"] is not None
    )
    assert tokens > 4 * sum(orchestrate.THRESHOLDS)


def test_the_tq4_threshold_cell_reads_the_observed_straddle_not_the_scan_counter(
    tmp_path,
):
    camp = _cell_camp(tmp_path, [(2048, None), (1024, None)], straddle_counter=False)
    got = camp.e1c(PICK, {2048: "a", 1024: "b"}, smoke=True)
    assert got["T2048"]["pass_"] is True
    assert got["T1024"]["evidence_problems"] == [
        "no straddle counter"
    ]  # the native layer's threshold: today's rule


def test_e1c_verdict_without_a_tuple_is_todays(tmp_path):
    on = dict(
        verdict={"pass": True},
        draft_rounds=3,
        prompt_end=orchestrate.straddle_target(2048),
        straddle=True,
        response=dict(timings=dict(verify_blocks_straddle=0)),
    )
    off = dict(rows={"pass": True}, states={"pass": True}, forced={"pass": True})
    v = orchestrate.e1c_verdict(on, off, threshold=2048, verify_policy="joint_v1")
    assert v["evidence_problems"] == [
        "no straddle counter"
    ]  # no TQ4 rule, no engagement check
    engaged = dict(on, tq4_engagement=dict(ok=True, problems=[]))
    v = orchestrate.e1c_verdict(
        engaged, off, threshold=None, verify_policy="joint_v1", tq4=True, prompt_end=5
    )
    assert v["evidence_problems"] == [f"prompt_end {on['prompt_end']} != 5"]
    with pytest.raises(orchestrate.MalformedEvidence, match="tq4_engagement"):
        orchestrate.e1c_verdict(
            on, off, threshold=None, verify_policy="joint_v1", tq4=True
        )


def _e2(engaged):
    turn = lambda **k: dict(  # noqa: E731
        final=dict(token_ids=[1, 2], finish_reason="stop"),
        mtp_rounds=k.get("rounds", 0),
        prompt=dict(sha256="a" * 64),
        telemetry=dict(speculative_argmax_from_hidden=k.get("argmax", 0)),
    )
    pair = {
        "x": dict(
            turns=[turn(rounds=3, argmax=2)],
            off=[turn(), turn()],
            verdict={"pass": True},
        )
    }
    if engaged is not None:
        pair["x"]["tq4_engagement"] = dict(ok=engaged, problems=[])
    return pair, {"x": dict(turns=[turn()])}


def test_e2_under_the_tuple_needs_the_tq4_engagement():
    pair, cross = _e2(True)
    assert (
        orchestrate.e2_summary(pair, cross, digests={}, tq4=True)["verdict"] == "PASS"
    )
    pair, cross = _e2(False)
    assert (
        orchestrate.e2_summary(pair, cross, digests={}, tq4=True)["verdict"]
        == "INCOMPLETE"
    )
    assert (
        orchestrate.e2_summary(pair, cross, digests={})["verdict"] == "PASS"
    )  # no tuple: today's
    pair, cross = _e2(None)
    with pytest.raises(orchestrate.MalformedEvidence):
        orchestrate.e2_summary(pair, cross, digests={}, tq4=True)


def test_the_summary_manifest_of_a_tuple_run_records_its_production_and_arm_commits(
    tmp_path, monkeypatch
):
    camp = stub_camp(TUPLE)
    camp.args = argparse.Namespace(smoke_summary=None, phase="full")
    camp.out, camp.picks, camp.fork, camp.head = tmp_path, [PICK], tmp_path, "h"
    camp.production = orchestrate.M77_PRODUCTION
    camp.summary = dict(picks={}, tuple=TUPLE)
    camp.write_summary = lambda: None
    camp.run_pick = lambda pick, smoke: None
    monkeypatch.setattr(orchestrate, "tree_guard", lambda f, h: None)
    monkeypatch.setattr(orchestrate, "Monitor", _NullMonitor)
    manifest = dict(
        fork_head="f" * 40, tuple=TUPLE, stack_head="s" * 40, off_stack_head="o" * 40
    )
    camp._execute(manifest)
    assert camp.summary["manifest"] == dict(
        fork_head="f" * 40,
        production=orchestrate.M77_PRODUCTION,
        tuple=TUPLE,
        stack_head="s" * 40,
        off_stack_head="o" * 40,
    )


class _NullMonitor:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ================================================================== worker: the hook identity and the route engagement
class _Hook:
    def __init__(self, name):
        self.name = name


class _RG:
    def __init__(self):
        self.model, self.mtp_verify_policy = object(), object()
        self.draft_model, self.draft_kind = "drafter", "mtp"
        self.tq4_attention = _Hook("m72_exact")


def test_the_tq4_identity_holds_the_hook_and_its_name():
    rg = _RG()
    ident = worker.instance_identity(rg, tq4=True)
    assert len(ident) == 4 and worker.same_instance(rg, ident)
    assert worker.instance_identity(rg) == ident[:2]  # without tq4: today's pair
    rg.tq4_attention = _Hook("m72_exact")  # a replaced hook with the same name
    assert not worker.same_instance(rg, ident)
    rg = _RG()
    ident = worker.instance_identity(rg, tq4=True)
    rg.tq4_attention.name = "m72_v1"  # the same object, renamed
    assert not worker.same_instance(rg, ident)
    rg = _RG()
    ident = worker.instance_identity(rg, tq4=True)
    rg.tq4_attention = None
    assert not worker.same_instance(rg, ident)
    assert worker.same_instance(
        rg, worker.instance_identity(rg)
    )  # the plain identity ignores the hook


LONG = dict(
    prompt_tokens=65472, generated=160
)  # every verify block past tq4_fast.SINGLE_PASS_MAX (2048)
SHORT = dict(prompt_tokens=960, generated=160)  # every block at or below it
CROSS = dict(prompt_tokens=1984, generated=160)  # the T2048 straddle: both sides
EDGE = dict(prompt_tokens=2045, generated=4)  # neither side certain


def test_the_limits_are_tq4_fasts():
    from mlx_vlm import tq4_fast

    assert (
        worker._tq4_limits()
        == (tq4_fast.SINGLE_PASS_MAX, tq4_fast.MAX_VERIFY)
        == (2048, 5)
    )


def test_the_route_engagement_rule_past_the_single_pass_limit():
    ok = worker.tq4_engagement(
        {"d2_verify": 3, "p2s": 1},
        {"d2_verify": 5, "p2s": 9, "fallback:mask": 0},
        **LONG,
    )
    assert ok["ok"] is True and ok["d2_verify"] == 2 and ok["expected"] == ["d2_verify"]
    assert ok["fallbacks"] == {"fallback:mask": 0}
    still = worker.tq4_engagement(
        {"d2_verify": 3}, {"d2_verify": 3, "per_query:short": 9}, **LONG
    )
    assert still["ok"] is False and still["problems"] == ["d2_verify did not move"]
    fell = worker.tq4_engagement(
        {"d2_verify": 3, "fallback:mask": 1},
        {"d2_verify": 6, "fallback:mask": 2},
        **LONG,
    )
    assert fell["ok"] is False and fell["problems"] == [
        "fallbacks {'fallback:mask': 1}"
    ]
    # a fallback counted before the trajectory (another request) does not fail this one
    assert worker.tq4_engagement(
        {"d2_verify": 0, "fallback:x": 4}, {"d2_verify": 1, "fallback:x": 4}, **LONG
    )["ok"]


def test_the_route_engagement_rule_below_and_across_the_limit():
    short = worker.tq4_engagement({}, {"per_query:short": 7}, **SHORT)
    assert short["ok"] is True and short["expected"] == ["per_query:short"]
    assert worker.tq4_engagement({}, {"d2_verify": 7}, **SHORT)["problems"] == [
        "per_query:short did not move"
    ]
    cross = worker.tq4_engagement({}, {"per_query:short": 2, "d2_verify": 30}, **CROSS)
    assert cross["ok"] is True and cross["expected"] == ["d2_verify", "per_query:short"]
    assert worker.tq4_engagement({}, {"per_query:short": 2}, **CROSS)["problems"] == [
        "d2_verify did not move"
    ]
    assert worker.tq4_engagement({}, {"d2_verify": 1}, **EDGE)["ok"] is True
    assert worker.tq4_engagement({}, {"p2s": 1}, **EDGE)["problems"] == [
        "neither d2_verify nor per_query:short moved"
    ]


@pytest.mark.parametrize("mode", ("m72_v1", "", None, "M72_EXACT"))
def test_the_route_engagement_supports_m72_exact_only(mode):
    with pytest.raises(ValueError, match="m72_exact"):
        worker.tq4_engagement({}, {"mma": 3}, mode=mode, **SHORT)


LIMIT, BLOCK = (
    2048,
    5,
)  # tq4_fast.SINGLE_PASS_MAX, MAX_VERIFY (test_the_limits_are_tq4_fasts)


def span(lo, hi):
    return dict(prompt_tokens=lo, generated=hi - lo)


def wanted(lo, hi):
    return worker.tq4_engagement(
        {}, {"d2_verify": 1, "per_query:short": 1}, **span(lo, hi)
    )["expected"]


def test_a_crossing_of_two_blocks_needs_both_routes():
    """Errata v1.1: a crossing wider than one draft block around 2,048 needs both routes."""
    got = span(
        LIMIT - BLOCK, LIMIT + BLOCK + 1
    )  # both edges just outside the one-block band
    assert wanted(LIMIT - BLOCK, LIMIT + BLOCK + 1) == ["d2_verify", "per_query:short"]
    assert worker.tq4_engagement({}, {"d2_verify": 4}, **got)["problems"] == [
        "per_query:short did not move"
    ]
    assert worker.tq4_engagement({}, {"per_query:short": 4}, **got)["problems"] == [
        "d2_verify did not move"
    ]
    assert worker.tq4_engagement({}, {"d2_verify": 1, "per_query:short": 1}, **got)[
        "ok"
    ]


@pytest.mark.parametrize(
    "lo, hi",
    ((LIMIT - 2, LIMIT + 3), (LIMIT - BLOCK + 1, LIMIT + BLOCK), (LIMIT, LIMIT + 1)),
    ids=("one-block", "band-edges", "at-the-limit"),
)
def test_a_crossing_within_one_block_needs_either_route(lo, hi):
    assert wanted(lo, hi) == []
    for moved in ({"d2_verify": 1}, {"per_query:short": 1}):
        assert worker.tq4_engagement({}, moved, **span(lo, hi))["ok"] is True
    assert worker.tq4_engagement({}, {"p2s": 9}, **span(lo, hi))["problems"] == [
        "neither d2_verify nor per_query:short moved"
    ]


def test_the_band_is_exactly_one_block_on_each_edge():
    # the upper edge: in the band up to limit + block, outside one past it (was 2 blocks)
    assert wanted(LIMIT - 1, LIMIT + BLOCK) == []
    assert wanted(LIMIT - 1, LIMIT + BLOCK + 1) == ["d2_verify"]
    assert wanted(LIMIT - 1, LIMIT + 2 * BLOCK) == ["d2_verify"]
    # the lower edge: in the band above limit - block, outside at it
    assert wanted(LIMIT - BLOCK + 1, LIMIT + 1) == []
    assert wanted(LIMIT - BLOCK, LIMIT + 1) == ["per_query:short"]
    # entirely above / at or below: one side's route, whatever the band
    assert wanted(LIMIT + 1, LIMIT + 2) == ["d2_verify"]
    assert wanted(LIMIT - 1, LIMIT) == ["per_query:short"]


def test_the_served_mode_check():
    rg = _RG()
    assert worker.tq4_mode_problems(rg, "m72_exact") == []
    assert worker.tq4_mode_problems(rg, "m72_v1") == [
        "tq4_attention='m72_exact' expected 'm72_v1'"
    ]
    rg.tq4_attention = None
    assert worker.tq4_mode_problems(rg, "m72_exact") == [
        "tq4_attention=None expected 'm72_exact'"
    ]


@pytest.fixture
def triple(monkeypatch):
    """``e2_pair_entry`` with the production calls faked (as test_p369_phase2_worker) plus the route counters."""
    from p369_phase2 import telemetry

    calls, rg = [], _RG()
    counts = {"d2_verify": 0}
    state = dict(swap_at=None, on_moves=2)

    def turn(client, capture, tel, payload, key, timeout):
        calls.append(("turn", key))
        on = rg.draft_model is not None
        if on:
            counts["d2_verify"] += state["on_moves"]
        if key == state["swap_at"]:
            rg.tq4_attention = _Hook("m72_exact")
        return dict(
            final=dict(token_ids=[1, 2], cached_tokens=0, finish_reason="length"),
            prompt=dict(tokens=4096, sha256="a" * 64),  # past the single-pass limit
            chat_id=f"chat-{len(calls)}",
            mtp_rounds=3 if on else 0,
            verify_scope_calls=5 if on else 0,
            telemetry=dict(speculative_argmax_from_hidden=4 if on else 0),
        )

    monkeypatch.setattr(worker, "greedy_turn", turn)
    monkeypatch.setattr(worker, "wait_idle", lambda rg_, m: calls.append(("idle",)))
    monkeypatch.setattr(
        worker, "release_session", lambda cid: calls.append(("release", cid))
    )
    monkeypatch.setattr(
        worker, "tq4_route_counts", lambda: calls.append(("routes",)) or dict(counts)
    )
    monkeypatch.setattr(telemetry, "KernelTelemetry", lambda lm: object())
    monkeypatch.setattr(telemetry, "verdict", lambda tel, manifest: {"pass": True})
    spec = dict(name="a", payload=dict(model="m"))

    def run(**k):
        return worker.e2_pair_entry(rg, None, None, None, None, {}, spec, 10.0, **k)

    return run, calls, state


def test_e2_pair_entry_records_the_on_trajectorys_route_engagement(triple):
    run, calls, _ = triple
    entry = run(tq4="m72_exact")
    assert calls[:4] == [("routes",), ("turn", "a"), ("idle",), ("routes",)]
    assert (
        calls.count(("routes",)) == 2
    )  # before / after ON only, never around the replays
    assert (
        entry["tq4_engagement"]["ok"] is True
        and entry["tq4_engagement"]["d2_verify"] == 2
    )


def test_e2_pair_entry_without_tq4_reads_no_counter(triple):
    run, calls, _ = triple
    entry = run()
    assert ("routes",) not in calls and "tq4_engagement" not in entry


def test_e2_pair_entry_refuses_a_replaced_hook(triple):
    run, _, state = triple
    state["swap_at"] = "a#OFF1"
    with pytest.raises(
        RuntimeError, match="tq4_attention hook changed during the E2 triple"
    ):
        run(tq4="m72_exact")


def test_e2_pair_entry_records_a_non_engaged_on_trajectory(triple):
    run, _, state = triple
    state["on_moves"] = 0
    assert run(tq4="m72_exact")["tq4_engagement"]["ok"] is False
