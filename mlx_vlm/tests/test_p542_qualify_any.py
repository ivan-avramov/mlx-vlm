"""P542 (C254): `qualify.py` for ANY entry of the stack root's committed registry, in either predictor state.

Written from the spec (P542_SPEC.md v1) before the implementation. The driver tests run the REAL ``qualify.main`` →
``drive`` path against synthetic registries built from the pinned golden fixture (``registry_states``, real), with the
host side mocked (lock, preflight, job, power, console script). Post-review r1 (spec v1.2): the fork root is a temp git
repo, so production refs resolve for real. Pure / temp-git (CPU only).
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import orchestrate, qualify, registry_states  # noqa: E402

from mlx_vlm.tests import test_p415_qualify as t415  # noqa: E402

FIXTURE = registry_states.fixture_text(Path(__file__).resolve().parent / "fixtures")
NAMES = [m["name"] for m in registry_states.parse(FIXTURE)["models"]]
PICK_A = "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed"
UNIFORM = "Qwen3.8-27B-mlx-uniform-4bit"
PICK_C = "Qwen3.6-27B-Opus-Distill-OptiQ-4bit"
PICK_D = "Ornith-1.0-35B-mlx-uniform-4bit"
SERVED_X = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
MTP_KEYS = (
    "draft_kind",
    "draft_model",
    "mtp_verify_scan",
)  # C177 / bench.p426_leg.MTP_KEYS
CANDIDATE = "Candidate-Model-9B-4bit"
CANDIDATE_BLOCK = """  - name: Candidate-Model-9B-4bit
    type: vision
    on_demand: true
    hf_path: example/Candidate-Model-9B-4bit
    max_kv_cache_size: 131072
    kv_prealloc_tokens: 131072
    kv_bits: 0
    prefill_step_size: 512
"""
CANDIDATE_MTP_BLOCK = CANDIDATE_BLOCK + (
    "    draft_kind: mtp\n    draft_model: example/Candidate-Model-9B-4bit-mtp-drafter\n"
)
OLD_PRODUCTION = "7207179b"  # the removed qualify.PRODUCTION constant
F = "f" * 40


def _with(text, *blocks):
    """``text`` (a registry) with extra ``models`` items appended (a new candidate entry)."""
    return text.rstrip("\n") + "\n\n" + "".join(blocks)


def _off(*names):
    return registry_states.apply(FIXTURE, set(names))


def _models(text):
    return {m["name"]: m for m in registry_states.parse(text)["models"]}


def _argv(text, name):
    """The worker argv (``_build_command(...)[1:]``, as P369's ``argv_on``) for ``name`` in registry ``text``."""
    entry = {m["name"]: m for m in yaml.safe_load(text)["models"]}[name]
    argv = []
    for flag, value in qualify.expected_config(entry)["flags"].items():
        argv += [flag] if flag in qualify.VALUELESS else [flag, value]
    return argv + ["--host", "127.0.0.1", "--port", "8091"]


def _edit_block(text, name, old, new=None):
    """``text`` with the one line ``old`` inside ``name``'s block replaced by ``new`` (deleted when None)."""
    lines = text.split("\n")
    start, end = registry_states._span(lines, name)
    hits = [i for i in range(start, end) if lines[i] == old]
    assert len(hits) == 1, (name, old, hits)
    if new is None:
        del lines[hits[0]]
    else:
        lines[hits[0]] = new
    return "\n".join(lines)


def _git(cwd, *a):
    return subprocess.run(
        ["git", "-C", str(cwd), *a], capture_output=True, text=True, check=True
    ).stdout.strip()


class _Result:
    def __init__(self, code, report, out):
        self.code, self.report, self.out = code, report, out
        self.refused = code if isinstance(code, str) else None

    def state(self, name):
        return (self.report or {}).get("entries", {}).get(name) or {}

    def completed(self, *names):
        return (
            self.code == 0
            and self.report is not None
            and all(self.state(n).get("status") == "COMPLETED" for n in names)
        )

    def text(self):
        return (self.refused or "") + json.dumps(self.report, default=str)


class _Rig:
    """A stack root with a committed ``main_models.yaml``, a campaign root for P369 summaries, and the host mocked."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp, self.mp = tmp_path, monkeypatch
        self.stack = tmp_path / "stack"
        (self.stack / ".venv" / "bin").mkdir(parents=True)
        self.fork = tmp_path / "fork"
        self.fork.mkdir()
        # r1: a real fork history P0 -> P -> R (the reviewed head); the default production is P's short sha, as the real
        # orchestrate.PRODUCTION is
        _git(tmp_path, "init", "-q", str(self.fork))
        shas = []
        for label in ("p0", "p", "r"):
            (self.fork / "f.txt").write_text(label)
            _git(self.fork, "add", "-A")
            _git(
                self.fork,
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
                label,
            )
            shas.append(_git(self.fork, "rev-parse", "HEAD"))
        self.P0, self.P, self.R = shas
        monkeypatch.setattr(orchestrate, "PRODUCTION", self.P[:8])
        self.workdir = tmp_path / "workdir"
        monkeypatch.setenv("STACK_WORKDIR", str(self.workdir))
        self.runs, self.served, self.jobs, self.expected = 0, [], [], []
        self.job_dirs, self.request_extra, self.hooks = [], [], {}
        self.classify_calls = []
        self.registry(FIXTURE)
        real_classify = registry_states.classify

        def classify(text, fixture):
            self.classify_calls.append(1)
            return real_classify(text, fixture)

        monkeypatch.setattr(registry_states, "classify", classify)
        monkeypatch.setattr(registry_states, "fixture_text", lambda d: FIXTURE)

        def run_job(job, *a, **k):
            name = job["name"].rsplit("/", 1)[0]
            self.jobs.append(name)
            self.job_dirs.append(job["out_dir"])
            i = NAMES.index(name) if name in NAMES else len(NAMES)
            rec = dict(
                peaks=dict(prefill=30e9 + i, decode=31e9 + i, retire=None),
                rates=dict(prefill_tps=700.0 + i, decode_tps=12.0 + i),
            )
            return dict(requests=[dict(qualify=rec)])

        def full_command(stack, path, name):
            text = Path(path).read_bytes().decode("utf-8")
            cmd = [str(Path(stack) / ".venv" / "bin" / "mlx_vlm.server")]
            cmd += _argv(text, name)
            self.served.append(dict(name=name, path=Path(path), text=text))
            return cmd, "m"

        def request_problems(rec, expected):
            self.expected.append(expected)
            return list(self.request_extra)

        def hooked(hook, value):
            def call(*a, **k):
                fn = self.hooks.pop(hook, None)
                if fn is not None:
                    fn()
                return value(*a, **k)

            return call

        def binding_problems(stack, module_file, registry_sha, sha256_file):
            # the real check's registry half (the submodule half needs the stack's git)
            same = sha256_file(Path(stack) / "main_models.yaml") == registry_sha
            return [] if same else ["main_models.yaml changed during the run"]

        for name, value in dict(
            acquire_lock=hooked("acquire_lock", lambda lock: 3),
            release_lock=lambda fd: None,
            lock_path=lambda env: tmp_path / ".lock",
            preflight=hooked("preflight", lambda args, out: dict(problems=[])),
            fork_versions=lambda f: {"mlx": "0.32.4"},
            weights_record=lambda m: dict(revision="r", files={}),
            run_job=run_job,
            worker_env=lambda env: env,
            Monitor=t415._NullMonitor,
        ).items():
            monkeypatch.setattr(orchestrate, name, value)
        for name, value in dict(
            full_command=full_command,
            request_problems=request_problems,
            load_references=lambda p, **k: {},
            _tree_state=lambda fork, stack: dict(
                fork_head=self.R, fork_clean=True, stack_head="s" * 40
            ),
            console_script_info=lambda stack: dict(
                shebang=str(Path(stack) / ".venv" / "bin" / "python"),
                entry="mlx_vlm.server:main",
                stack_mlx_vlm="x",
            ),
            _last_tick_age=lambda p: 1.0,
            _pmset_log=lambda: (0, t415.PMSET),
            interference=hooked("interference", lambda o: dict(ok=True, why=None)),
            binding_problems=binding_problems,
            _is_commit=lambda path, ref: True,
            _is_ancestor=lambda path, a, ref: True,
        ).items():
            monkeypatch.setattr(qualify, name, value)
        monkeypatch.setattr("time.sleep", lambda s: None)

    def registry(self, text):
        self.committed = self.stack / "main_models.yaml"
        self.committed.write_bytes(text.encode("utf-8"))
        return self.committed

    def certified_argv(self, name):
        """The argv the P369 run certified for ``name`` MTP-ON: the committed block when it serves a drafter, else the
        16-state overlay (None when neither gives one)."""
        text = self.committed.read_bytes().decode("utf-8")
        try:
            if qualify.expected_config(_models(text)[name])["drafter"]:
                return _argv(text, name)
            return _argv(orchestrate.overlay_registry(text, name, FIXTURE)[0], name)
        except (KeyError, ValueError):
            return None

    def summary(self, name, production=None, argv_on="auto", **changes):
        """A P369 FULL-run PASS for ``name`` as a real campaign artifact (beside its manifest.json, which carries
        ``picks[name].argv_on`` unless ``argv_on`` is None)."""
        prod = orchestrate.PRODUCTION if production is None else production
        if argv_on == "auto":
            argv_on = self.certified_argv(name)
        self.runs += 1
        run = self.workdir / "p369" / "phase2" / f"run-{self.runs}"
        run.mkdir(parents=True)
        s = dict(
            phase="full",
            status="PASS",
            binding=dict(production=prod, fork_head=F, reviewed_head=F),
            manifest=dict(production=prod, fork_head=F),
            picks={name: dict(outcome=dict(e1c="PASS"), e2=dict(verdict="PASS"))},
        )
        s.update(changes)
        (run / "summary.json").write_text(json.dumps(s))
        manifest = dict(fork_head=F, production=prod)
        if argv_on is not None:
            manifest["picks"] = {name: dict(argv_on=argv_on)}
        (run / "manifest.json").write_text(json.dumps(manifest))
        return run / "summary.json"

    def run(self, *extra):
        self.runs += 1
        out = self.tmp / f"out-{self.runs}"
        self.served, self.jobs, self.expected, self.job_dirs = [], [], [], []
        self.classify_calls.clear()
        argv = [
            "--stack-root",
            str(self.stack),
            "--fork-root",
            str(self.fork),
            "--out",
            str(out),
            "--reviewed-head",
            self.R,
            *[str(a) for a in extra],
        ]
        code = None
        try:
            qualify.main(argv)
        except SystemExit as exc:
            code = exc.code
        path = out / "report.json"
        report = json.loads(path.read_text()) if path.exists() else None
        return _Result(code, report, out)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return _Rig(tmp_path, monkeypatch)


def _under(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


# ------------------------------------------------------------------ change 1: any registry entry; default_entries
def test_c1_the_fixed_entries_table_is_removed():
    assert not hasattr(qualify, "ENTRIES")


@pytest.mark.parametrize(
    "registry, name",
    [(FIXTURE, SERVED_X), (_with(FIXTURE, CANDIDATE_BLOCK), CANDIDATE)],
    ids=["served-x", "new-candidate"],
)
def test_c1_any_registry_name_is_accepted(rig, registry, name):
    rig.registry(registry)
    got = rig.run("--entry", name)
    assert got.completed(name), got.text()
    assert rig.jobs == [name]


def test_c1_a_name_outside_the_registry_refuses_listing_the_registry_names(rig):
    rig.registry(_with(FIXTURE, CANDIDATE_BLOCK))
    got = rig.run("--entry", "Not-A-Registry-Model")
    assert got.code != 0 and rig.jobs == []
    message = got.refused or got.text()
    assert "Not-A-Registry-Model" in message
    for name in NAMES + [CANDIDATE]:
        assert name in message, (name, message)


def test_c1_default_entries_are_the_picks_in_the_registry(rig):
    """No ``--entry``: every registry entry in ``orchestrate.PICKS`` (the four MTP entries), never the served-only entry or a new one."""
    rig.registry(_with(FIXTURE, CANDIDATE_BLOCK))
    got = rig.run("--predictor", "off")  # OFF needs no P369 gate
    want = set(orchestrate.PICKS) & set(NAMES)
    assert len(want) == 4
    assert set((got.report or {}).get("entries", {})) == want, got.text()
    assert got.completed(*want) and sorted(rig.jobs) == sorted(want)


# ------------------------------------------------------------------ change 2: the arm per entry (arm_for)
@pytest.mark.parametrize(
    "registry, name, predictor, arm",
    [
        (FIXTURE, PICK_A, None, "served_on"),
        (FIXTURE, PICK_A, "on", "served_on"),
        (FIXTURE, PICK_A, "off", "mtp_off"),
        (FIXTURE, PICK_D, "off", "mtp_off"),
        (FIXTURE, SERVED_X, None, "served_off"),
        (FIXTURE, SERVED_X, "off", "served_off"),
        (_off(PICK_A), PICK_A, None, "served_off"),
        (_off(PICK_A), PICK_A, "off", "served_off"),
    ],
    ids=[
        "pick-a-on-none",
        "pick-a-on-on",
        "pick-a-on-off",
        "ornith-on-off",
        "served-x-none",
        "served-x-off",
        "pick-a-off-none",
        "pick-a-off-off",
    ],
)
def test_c2_arm_for_derives_the_arm_from_the_committed_block(
    registry, name, predictor, arm
):
    got = qualify.arm_for(
        name, predictor, registry_entry=_models(registry)[name], registry_state=None
    )
    assert got == arm


@pytest.mark.parametrize(
    "registry, name",
    [(FIXTURE, SERVED_X), (_with(FIXTURE, CANDIDATE_BLOCK), CANDIDATE)],
    ids=["served-x", "new-candidate"],
)
def test_c2_arm_for_refuses_on_without_a_drafter_to_restore(registry, name):
    with pytest.raises(ValueError, match=f"{name}: no MTP drafter to restore"):
        qualify.arm_for(
            name, "on", registry_entry=_models(registry)[name], registry_state=None
        )


def test_c2_served_on_serves_the_committed_registry(rig):
    got = rig.run("--entry", PICK_A, "--p369-summary", rig.summary(PICK_A))  # bare path
    assert got.completed(PICK_A), got.text()
    assert got.state(PICK_A)["arm"] == "served_on"
    assert [s["path"] for s in rig.served] == [rig.committed]
    assert rig.expected[0]["drafter"] is True


def test_c2_predictor_on_for_an_entry_served_on_serves_it_as_committed(rig):
    got = rig.run(
        "--entry", PICK_A, "--predictor", "on", "--p369-summary", rig.summary(PICK_A)
    )
    assert got.completed(PICK_A), got.text()
    assert got.state(PICK_A)["arm"] == "served_on"
    assert [s["path"] for s in rig.served] == [rig.committed]


@pytest.mark.parametrize("predictor", [None, "off"], ids=["no-predictor", "off"])
def test_c2_an_entry_served_off_is_served_as_committed_without_a_gate(rig, predictor):
    rig.registry(_off(PICK_A))
    got = rig.run("--entry", PICK_A, *(["--predictor", predictor] if predictor else []))
    assert got.completed(PICK_A), got.text()
    assert got.state(PICK_A)["arm"] == "served_off"
    assert [s["path"] for s in rig.served] == [rig.committed]
    assert rig.expected[0]["drafter"] is False


@pytest.mark.parametrize("name", [PICK_A, UNIFORM, PICK_D, PICK_C])
def test_c2_mtp_off_removes_only_the_mtp_keys_and_records_the_derived_sha(rig, name):
    got = rig.run("--entry", name, "--predictor", "off")
    assert got.completed(name), got.text()
    assert got.state(name)["arm"] == "mtp_off"
    [served] = rig.served
    assert served["path"] != rig.committed and _under(served["path"], got.out)
    committed = registry_states.parse(FIXTURE)
    derived = registry_states.parse(served["text"])
    assert {k: v for k, v in derived.items() if k != "models"} == {
        k: v for k, v in committed.items() if k != "models"
    }
    assert [m["name"] for m in derived["models"]] == NAMES
    for want, have in zip(committed["models"], derived["models"]):
        if want["name"] == name:
            assert any(k in want for k in MTP_KEYS)
            assert have == {k: v for k, v in want.items() if k not in MTP_KEYS}
        else:
            assert have == want
    digest = hashlib.sha256(served["path"].read_bytes()).hexdigest()
    assert digest in json.dumps(got.report), "the derived registry's sha256 is recorded"
    assert rig.expected[0]["drafter"] is False
    assert rig.committed.read_bytes() == FIXTURE.encode("utf-8")  # never edited


def test_c2_mtp_on_restored_serves_the_16_state_overlay(rig):
    committed = _off(PICK_A)
    rig.registry(committed)
    got = rig.run(
        "--entry", PICK_A, "--predictor", "on", "--p369-summary", rig.summary(PICK_A)
    )
    assert got.completed(PICK_A), got.text()
    assert got.state(PICK_A)["arm"] == "mtp_on_restored"
    [served] = rig.served
    assert _under(served["path"], got.out)
    assert served["text"] == orchestrate.overlay_registry(committed, PICK_A, FIXTURE)[0]
    assert rig.expected[0]["drafter"] is True


@pytest.mark.parametrize(
    "registry, name",
    [(FIXTURE, SERVED_X), (_with(FIXTURE, CANDIDATE_BLOCK), CANDIDATE)],
    ids=["served-x", "new-candidate"],
)
def test_c2_predictor_on_without_a_restorable_drafter_refuses(rig, registry, name):
    rig.registry(registry)
    got = rig.run("--entry", name, "--predictor", "on")
    assert not got.completed(name) and rig.jobs == []
    assert f"{name}: no MTP drafter to restore" in got.text()


# ------------------------------------------------------------------ change 3: the MTP-ON gate (generalized C199)
def test_c3_served_on_without_a_p369_summary_refuses(rig):
    got = rig.run("--entry", PICK_A)
    assert not got.completed(PICK_A) and rig.jobs == []


def test_c3_mtp_on_restored_without_a_p369_summary_refuses(rig):
    rig.registry(_off(PICK_A))
    got = rig.run("--entry", PICK_A, "--predictor", "on")
    assert not got.completed(PICK_A) and rig.jobs == []


def test_c3_a_summary_for_another_entry_refuses(rig):
    got = rig.run("--entry", PICK_A, "--p369-summary", rig.summary(UNIFORM))
    assert not got.completed(PICK_A) and rig.jobs == []


def test_c3_a_summary_on_the_old_production_constant_refuses(rig):
    got = rig.run(
        "--entry", PICK_A, "--p369-summary", rig.summary(PICK_A, OLD_PRODUCTION)
    )
    assert not got.completed(PICK_A) and rig.jobs == []
    s = rig.summary(PICK_A, manifest=dict(production=OLD_PRODUCTION, fork_head=F))
    got = rig.run("--entry", PICK_A, "--p369-summary", s)  # manifest.production too
    assert not got.completed(PICK_A) and rig.jobs == []


def test_c3_a_summary_on_the_default_production_passes(rig):
    """The default attested production is ``orchestrate.PRODUCTION`` — not qualify's old ``7207179b``. Post-review r1:
    realistic shapes — the default is a SHORT sha of a fork commit, the P369 run recorded the FULL sha of it.
    """
    assert orchestrate.PRODUCTION == rig.P[:8] != OLD_PRODUCTION
    got = rig.run("--entry", PICK_C, "--p369-summary", rig.summary(PICK_C, rig.P))
    assert got.completed(PICK_C), got.text()
    assert got.state(PICK_C)["arm"] == "served_on"


def test_c3_a_summary_must_match_the_runs_production_ref(rig):
    ref = (
        rig.P0
    )  # a real strict ancestor of the reviewed head (resolve_production_ref is not mocked)
    got = rig.run(
        "--entry",
        PICK_C,
        "--production-ref",
        ref,
        "--p369-summary",
        rig.summary(PICK_C, ref),
    )
    assert got.completed(PICK_C), got.text()
    assert got.report["production"] == ref
    got = rig.run(
        "--entry",
        PICK_C,
        "--production-ref",
        ref,
        "--p369-summary",
        rig.summary(PICK_C),  # orchestrate.PRODUCTION, not this run's
    )
    assert not got.completed(PICK_C) and rig.jobs == []


def test_c3_per_entry_summaries_name_equals_path(rig):
    sum_a, sum_b = rig.summary(PICK_A), rig.summary(UNIFORM)
    got = rig.run(
        "--entry",
        PICK_A,
        "--entry",
        UNIFORM,
        "--p369-summary",
        f"{PICK_A}={sum_a}",
        "--p369-summary",
        f"{UNIFORM}={sum_b}",
    )
    assert got.completed(PICK_A, UNIFORM), got.text()
    assert {got.state(n)["arm"] for n in (PICK_A, UNIFORM)} == {"served_on"}
    got = rig.run(  # swapped: each entry gets the other's summary
        "--entry",
        PICK_A,
        "--entry",
        UNIFORM,
        "--p369-summary",
        f"{PICK_A}={sum_b}",
        "--p369-summary",
        f"{UNIFORM}={sum_a}",
    )
    assert not got.completed(PICK_A, UNIFORM) and rig.jobs == []


def test_c3_mtp_on_needs_the_entry_in_picks_even_with_a_summary(rig):
    """(a): a registry entry served MTP-ON outside ``orchestrate.PICKS`` has no drafter shape — refused; its OFF arm runs."""
    rig.registry(_with(FIXTURE, CANDIDATE_MTP_BLOCK))
    assert CANDIDATE not in orchestrate.PICKS
    got = rig.run("--entry", CANDIDATE, "--predictor", "off")
    assert got.completed(CANDIDATE), got.text()
    assert got.state(CANDIDATE)["arm"] == "mtp_off"
    got = rig.run("--entry", CANDIDATE, "--p369-summary", rig.summary(CANDIDATE))
    assert not got.completed(CANDIDATE) and rig.jobs == []


# ------------------------------------------------------------------ change 4: the classifier only for a restore
def test_c4_served_and_off_arms_never_classify(rig):
    rig.run("--entry", PICK_A, "--p369-summary", rig.summary(PICK_A))
    assert rig.classify_calls == [] and rig.jobs == [PICK_A]
    rig.run("--entry", UNIFORM, "--predictor", "off")
    assert rig.classify_calls == [] and rig.jobs == [UNIFORM]
    rig.run("--entry", SERVED_X)
    assert rig.classify_calls == [] and rig.jobs == [SERVED_X]


def test_c4_a_restore_classifies(rig):
    rig.registry(_off(PICK_A))
    got = rig.run(
        "--entry", PICK_A, "--predictor", "on", "--p369-summary", rig.summary(PICK_A)
    )
    assert got.completed(PICK_A), got.text()
    assert got.state(PICK_A)["arm"] == "mtp_on_restored"
    assert rig.classify_calls


@pytest.mark.parametrize(
    "registry, name, extra, arm",
    [
        (FIXTURE, PICK_A, (), "served_on"),
        (FIXTURE, UNIFORM, ("--predictor", "off"), "mtp_off"),
        (_off(PICK_A), PICK_A, (), "served_off"),
        (FIXTURE, CANDIDATE, (), "served_off"),
    ],
    ids=["served_on", "mtp_off", "served_off", "the-new-entry"],
)
def test_c4_a_registry_with_an_extra_entry_is_accepted_for_served_and_off_arms(
    rig, registry, name, extra, arm
):
    rig.registry(_with(registry, CANDIDATE_BLOCK))
    if arm == "served_on":
        extra = ("--p369-summary", rig.summary(name))
    got = rig.run("--entry", name, *extra)
    assert got.completed(name), got.text()
    assert got.state(name)["arm"] == arm


def test_c4_a_registry_with_an_extra_entry_is_refused_for_a_restore(rig):
    rig.registry(_with(_off(PICK_A), CANDIDATE_BLOCK))
    got = rig.run("--entry", PICK_A)  # the same registry serves the entry OFF fine
    assert (
        got.completed(PICK_A) and got.state(PICK_A)["arm"] == "served_off"
    ), got.text()
    got = rig.run(
        "--entry", PICK_A, "--predictor", "on", "--p369-summary", rig.summary(PICK_A)
    )
    assert not got.completed(PICK_A) and rig.jobs == []


# ------------------------------------------------------------------ change 5: references are optional
def test_c5_load_references_takes_any_entry_name(tmp_path):
    ref = dict(
        peak_gb=28.0,
        source="docs/x.md:1",
        source_path="docs/x.md",
        source_blob="0" * 40,
        fingerprint={},
    )
    table = {CANDIDATE: None, SERVED_X: ref}
    got = qualify.load_references(tmp_path, table=table, blob_exists=lambda b, p: True)
    assert got == table
    with pytest.raises(ValueError):  # the row checks themselves are unchanged
        qualify.load_references(
            tmp_path,
            table={CANDIDATE: dict(peak_gb=28.0)},
            blob_exists=lambda b, p: True,
        )


def test_c5_an_entry_without_a_reference_reports_peak_and_guidance_only(rig):
    got = rig.run("--entry", SERVED_X)
    assert got.completed(SERVED_X), got.text()
    state = got.state(SERVED_X)
    assert state["capacity_gb"] == pytest.approx(31e9 / 1e9, abs=1e-3)
    assert state["vs_guidance_gb"] == pytest.approx(
        state["capacity_gb"] - 48.0, abs=1e-3
    )
    assert state.get("reference") is None


# ------------------------------------------------------------------ change 6: rates and arm in the final state
def test_c6_the_final_state_carries_rates_and_arm_per_entry(rig):
    got = rig.run("--entry", SERVED_X, "--entry", PICK_D, "--predictor", "off")
    assert got.completed(SERVED_X, PICK_D), got.text()
    for name, arm in ((SERVED_X, "served_off"), (PICK_D, "mtp_off")):
        state = got.state(name)
        assert state["arm"] == arm
        assert state["rates"] == state["record"]["rates"]
        i = NAMES.index(name)
        assert state["rates"] == dict(prefill_tps=700.0 + i, decode_tps=12.0 + i)


# ------------------------------------------------------------------ change 7: an unknown registry key still refuses
def test_c7_an_unknown_registry_key_on_a_new_entry_still_refuses(rig):
    rig.registry(_with(FIXTURE, CANDIDATE_BLOCK + "    new_knob: 1\n"))
    got = rig.run("--entry", CANDIDATE)
    assert not got.completed(CANDIDATE) and rig.jobs == []
    assert "new_knob" in got.text()


# ================================================================== post-review r1 (spec v1.2)
PICK_A_DRAFT_MODEL = (
    "    draft_model: caslca/Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed-mtp-drafter"
)
CERTIFIED = "the served configuration differs from the one its P369 run certified"


def _set_flag(argv, flag, value):
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv += [flag, value]
    return argv


def _without_lines(text, name, drop):
    """``text`` with the lines ``drop`` (exact) removed from ``name``'s block — the expected textual removal."""
    for line in drop:
        text = _edit_block(text, name, line)
    return text


# ------------------------------------------------------------------ r1 v1.2: one registry snapshot
@pytest.mark.parametrize("hook", ["acquire_lock", "preflight", "interference"])
def test_r1_a_registry_flipped_on_after_the_gate_refuses(rig, hook):
    """The gate saw the entry OFF (no P369 summary needed); the file then turns it ON before execution."""
    rig.registry(_off(PICK_A))
    rig.hooks[hook] = lambda: rig.registry(FIXTURE)
    got = rig.run("--entry", PICK_A)
    assert not got.completed(PICK_A), got.text()
    assert not any(e["drafter"] for e in rig.expected), "an ungated MTP-ON job ran"


def test_r1_a_registry_changed_between_the_gate_and_execution_refuses(rig):
    rig.registry(FIXTURE)
    rig.hooks["preflight"] = lambda: rig.registry(
        _off(UNIFORM)
    )  # another entry's block
    got = rig.run("--entry", SERVED_X)
    assert not got.completed(SERVED_X), got.text()


# ------------------------------------------------------------------ r1 v1.2: MTP-ON bound to the certified argv
def test_r1_an_mtp_on_arm_runs_when_its_argv_equals_the_certified_one(rig):
    s = rig.summary(PICK_A)
    assert json.loads((s.parent / "manifest.json").read_text())["picks"][PICK_A][
        "argv_on"
    ] == _argv(FIXTURE, PICK_A)
    got = rig.run("--entry", PICK_A, "--p369-summary", s)
    assert got.completed(PICK_A), got.text()


@pytest.mark.parametrize(
    "registry, name, predictor, flag, mutate",
    [
        (FIXTURE, UNIFORM, None, "--kv-bits", lambda a: _set_flag(a, "--kv-bits", "8")),
        (FIXTURE, PICK_A, None, "--kv-bits", lambda a: _set_flag(a, "--kv-bits", "4")),
        (
            FIXTURE,
            PICK_A,
            None,
            "--draft-model",
            lambda a: _set_flag(a, "--draft-model", "caslca/another-drafter"),
        ),
        (
            _off(PICK_A),
            PICK_A,
            "on",
            "--draft-model",
            lambda a: _set_flag(a, "--draft-model", "caslca/another-drafter"),
        ),
    ],
    ids=[
        "uniform-kv-bits",
        "pick-a-kv-bits-added",
        "pick-a-drafter",
        "restored-drafter",
    ],
)
def test_r1_an_mtp_on_argv_differing_from_the_certified_one_refuses_naming_the_flag(
    rig, registry, name, predictor, flag, mutate
):
    rig.registry(registry)
    s = rig.summary(name, argv_on=mutate(rig.certified_argv(name)))
    got = rig.run(
        "--entry",
        name,
        "--p369-summary",
        s,
        *(["--predictor", predictor] if predictor else []),
    )
    assert not got.completed(name) and rig.jobs == []
    assert f"{name}: {CERTIFIED}" in got.text() and flag in got.text(), got.text()


def test_r1_a_p369_manifest_without_the_entrys_argv_refuses(rig):
    got = rig.run(
        "--entry", PICK_A, "--p369-summary", rig.summary(PICK_A, argv_on=None)
    )
    assert not got.completed(PICK_A) and rig.jobs == []


# ------------------------------------------------------------------ r1 v1.2: production resolved to full shas
def test_r1_a_short_summary_matches_a_full_run_ref(rig):
    s = rig.summary(PICK_C, rig.P0[:8])
    got = rig.run("--entry", PICK_C, "--production-ref", rig.P0, "--p369-summary", s)
    assert got.completed(PICK_C), got.text()


def test_r1_a_full_summary_matches_a_short_run_ref(rig):
    s = rig.summary(PICK_C, rig.P0)
    got = rig.run(
        "--entry", PICK_C, "--production-ref", rig.P0[:8], "--p369-summary", s
    )
    assert got.completed(PICK_C), got.text()


def test_r1_a_summary_on_another_commit_refuses(rig):
    got = rig.run("--entry", PICK_C, "--p369-summary", rig.summary(PICK_C, rig.P0))
    assert not got.completed(PICK_C) and rig.jobs == []


def test_r1_an_unresolvable_default_production_refuses(rig, monkeypatch):
    """The default is resolved too: a short sha that is no fork commit refuses, even when the summary repeats it."""
    monkeypatch.setattr(orchestrate, "PRODUCTION", "0123abcd")
    got = rig.run("--entry", PICK_C, "--p369-summary", rig.summary(PICK_C, "0123abcd"))
    assert not got.completed(PICK_C) and rig.jobs == []


@pytest.mark.parametrize(
    "extra, production",
    [(("--production-ref", "no-such-ref"), None), ((), "0000000")],
    ids=["run-ref", "summary"],
)
def test_r1_an_unresolvable_production_refuses(rig, extra, production):
    s = rig.summary(PICK_C, production)
    got = rig.run("--entry", PICK_C, "--p369-summary", s, *extra)
    assert not got.completed(PICK_C) and rig.jobs == []


# ------------------------------------------------------------------ r1 v1.2: served MTP-ON = expected_config's drafter
@pytest.mark.parametrize(
    "registry",
    [
        _edit_block(
            FIXTURE, PICK_A, PICK_A_DRAFT_MODEL
        ),  # draft_kind + scan, no drafter
        _edit_block(
            _edit_block(FIXTURE, PICK_A, PICK_A_DRAFT_MODEL),
            PICK_A,
            "    draft_kind: mtp",
        ),  # the scan alone
        _edit_block(FIXTURE, PICK_A, "    draft_kind: mtp", "    draft_kind: suffix"),
    ],
    ids=["no-draft-model", "scan-only", "non-mtp-kind"],
)
def test_r1_a_partial_or_non_mtp_draft_block_refuses(rig, registry):
    rig.registry(registry)
    got = rig.run("--entry", PICK_A, "--p369-summary", rig.summary(PICK_A))
    assert not got.completed(PICK_A) and rig.jobs == []
    assert f"{PICK_A}: a partial or non-MTP draft block" in got.text(), got.text()


# ------------------------------------------------------------------ r1 v1.2: mtp_off through the YAML node marks
INDENTED = "Candidate-Indented-9B-4bit"
INDENTED_BLOCK = """  -   name: Candidate-Indented-9B-4bit
      type: vision
      on_demand: true
      hf_path: example/Candidate-Indented-9B-4bit
      max_kv_cache_size: 131072
      kv_prealloc_tokens: 131072
      kv_bits: 0
      prefill_step_size: 512
      draft_kind: mtp
      draft_model: example/Candidate-Indented-9B-4bit-mtp-drafter
"""
FLOW = "Candidate-Flow-9B-4bit"
FLOW_BLOCK = (
    "  - {name: Candidate-Flow-9B-4bit, type: vision, on_demand: true, hf_path: example/Candidate-Flow-9B-4bit, "
    "max_kv_cache_size: 131072, kv_prealloc_tokens: 131072, kv_bits: 0, prefill_step_size: 512, draft_kind: mtp, "
    "draft_model: example/flow-drafter}\n"
)
QUOTED = _edit_block(
    _edit_block(FIXTURE, PICK_A, "    draft_kind: mtp", '    "draft_kind": mtp'),
    PICK_A,
    "    mtp_verify_scan: joint_v1",
    "    'mtp_verify_scan': joint_v1",
)


@pytest.mark.parametrize(
    "registry, name, drop",
    [
        (
            QUOTED,
            PICK_A,
            [
                '    "draft_kind": mtp',
                PICK_A_DRAFT_MODEL,
                "    'mtp_verify_scan': joint_v1",
            ],
        ),
        (
            _with(FIXTURE, INDENTED_BLOCK),
            INDENTED,
            [
                "      draft_kind: mtp",
                "      draft_model: example/Candidate-Indented-9B-4bit-mtp-drafter",
            ],
        ),
    ],
    ids=["quoted-keys", "indented-block"],
)
def test_r1_mtp_off_removes_quoted_and_differently_indented_keys(
    rig, registry, name, drop
):
    rig.registry(registry)
    got = rig.run("--entry", name, "--predictor", "off")
    assert got.completed(name), got.text()
    [served] = rig.served
    assert served["text"] == _without_lines(
        registry, name, drop
    )  # every other byte kept
    assert (
        got.state(name)["overlay_sha256"]
        == hashlib.sha256(served["path"].read_bytes()).hexdigest()
    )


def test_r1_an_mtp_key_that_cannot_be_removed_names_the_key(rig):
    """A flow-style block keeps its keys on the entry's own line: no complete line to remove — refused, key named."""
    rig.registry(_with(FIXTURE, FLOW_BLOCK))
    got = rig.run("--entry", FLOW, "--predictor", "off")
    assert not got.completed(FLOW) and rig.jobs == []
    assert "draft_kind" in got.text(), got.text()


# ------------------------------------------------------------------ r1 v1.2: safe filesystem ids under --out
@pytest.mark.parametrize(
    "name", ["org/Model-X-4bit", "/abs/Model-Y-4bit", "../Model-Z-4bit"]
)
@pytest.mark.parametrize("mtp", [False, True], ids=["served_off", "mtp_off"])
def test_r1_per_entry_paths_use_a_safe_id_under_out(rig, name, mtp):
    block = (CANDIDATE_MTP_BLOCK if mtp else CANDIDATE_BLOCK).replace(CANDIDATE, name)
    rig.registry(_with(FIXTURE, block))
    got = rig.run("--entry", name, *(["--predictor", "off"] if mtp else []))
    assert got.completed(name), got.text()
    assert got.state(name)["arm"] == ("mtp_off" if mtp else "served_off")
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    paths = [Path(d) for d in rig.job_dirs] + (
        [s["path"] for s in rig.served] if mtp else []
    )
    assert paths
    for path in paths:
        rel = (
            Path(path).resolve().relative_to(got.out.resolve())
        )  # raises when outside --out
        assert len(rel.parts) == 1 and digest in rel.name, rel
        assert re.fullmatch(r"[A-Za-z0-9._-]+", rel.name), rel


# ------------------------------------------------------------------ r1 v1.2: arm_for and registry_state
def test_r1_arm_for_refuses_a_restore_when_the_name_is_not_off_in_registry_state():
    entry = _models(_off(PICK_A))[PICK_A]
    assert (
        qualify.arm_for(
            PICK_A, "on", registry_entry=entry, registry_state=frozenset({PICK_A})
        )
        == "mtp_on_restored"
    )
    for state in (frozenset({UNIFORM}), frozenset()):
        with pytest.raises(ValueError, match=re.escape(PICK_A)):
            qualify.arm_for(PICK_A, "on", registry_entry=entry, registry_state=state)


# ------------------------------------------------------------------ r1 v1.2: rates and arm on a FAILED entry
def test_r1_a_failed_entry_with_a_record_carries_rates_and_arm(rig):
    rig.request_extra = ["synthetic request problem"]
    got = rig.run("--entry", SERVED_X)
    state = got.state(SERVED_X)
    assert got.code == 1 and state["status"] == "FAILED", got.text()
    assert state["arm"] == "served_off"
    assert state["rates"] == state["record"]["rates"]
    i = NAMES.index(SERVED_X)
    assert state["rates"] == dict(prefill_tps=700.0 + i, decode_tps=12.0 + i)
    assert "capacity_gb" not in state
