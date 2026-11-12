"""M76 B2 goldens: today's fingerprints, KV-arm classifications, qualify reports and P369 plans / manifests / jobs for the
existing picks, recorded BEFORE the B2 change (fork ``f9bf743f``) and pinned byte-for-byte (sha256 of canonical JSON).

CPU only: the host, the worker and mlx-serve are mocked exactly as the P542 / P475a rigs do.
"""

import argparse
import hashlib
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import orchestrate, qualify, registry_states  # noqa: E402

from mlx_vlm.tests import test_p475a_kv_arm as kv  # noqa: E402
from mlx_vlm.tests import test_p475a_qualify_driver as drv  # noqa: E402
from mlx_vlm.tests.test_p475a_qualify_driver import (  # noqa: E402,F401  (the fixture)
    rig,
)

FIXTURE = kv.FIXTURE
ALL_OFF = registry_states.apply(FIXTURE, frozenset(registry_states.ENTRY_OFF))
MTP_PICKS = list(registry_states.ENTRY_OFF)


def canon(obj):
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def sha(obj):
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()


def pinned(obj, want):
    got = sha(obj)
    assert got == want, f"golden drift: {got}\n{canon(obj)}"


REC = dict(
    versions=dict(mlx="0.32.4", **{"mlx-metal": "0.32.4"}),
    snapshots={"--model": "rev-target", "--draft-model": "rev-drafter"},
    weights={"--model": "w" * 64, "--draft-model": "d" * 64},
    harness="h" * 40,
    memory=dict(set_memory_limit=[100], set_cache_limit=[200]),
    generation_tokens=77,
    prompt=dict(sized=dict(sha256="p" * 64, tokens=1)),
)


# ------------------------------------------------------------------ qualify.fingerprint_of / expected_config
FINGERPRINTS = {
    "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed": "559ba288502b8ad8257ed43968cac55de428d208dc15020be2cb10493a58b501",
    "Qwen3.8-27B-mlx-uniform-4bit": "c3857f43e7276199e99204b439ae22ed9d2ef02f9c2ccc124cde34824f572670",
    "Ornith-1.0-35B-mlx-uniform-4bit": "c73a9f6a4f22bd18e9091b270c5b100b99db14700f5458c96fa2379fdddc9c04",
    "Qwen3.6-27B-Opus-Distill-OptiQ-4bit": "c3857f43e7276199e99204b439ae22ed9d2ef02f9c2ccc124cde34824f572670",
}


@pytest.mark.parametrize("name", MTP_PICKS)
def test_golden_fingerprint_of_every_pick_as_committed_and_mtp_off(name):
    entry = registry_states.settings(FIXTURE)[name]
    off = {k: v for k, v in entry.items() if k not in qualify.MTP_KEYS}
    got = [
        qualify.fingerprint_of(REC, qualify.expected_config(e)) for e in (entry, off)
    ]
    pinned(got, FINGERPRINTS[name])


ARM_FINGERPRINTS = "1a62c7f9a45e636e1ce4dc5c3f7ee77cdd44a2085efa6c1b7175b91e69c92014"


def test_golden_fingerprint_of_every_p475_arm_of_both_kv_picks():
    got = {
        f"{pick}/{arm}": qualify.fingerprint_of(
            REC, qualify.expected_config(kv.entry_for(arm, pick))
        )
        for pick, arm in kv.CELLS
    }
    pinned(got, ARM_FINGERPRINTS)


EXPECTED_CONFIGS = "950012a63342c49bc0f0efa6539d62d80f696ca6bb4c9d226bac103deaf9c686"


def test_golden_expected_config_of_every_fixture_entry():
    got = {
        name: qualify.expected_config(entry)
        for name, entry in registry_states.settings(FIXTURE).items()
    }
    pinned(got, EXPECTED_CONFIGS)


def test_golden_reference_comparison_of_the_committed_fingerprint():
    fp = qualify.fingerprint_of(
        REC, qualify.expected_config(registry_states.settings(FIXTURE)[kv.UNIFORM])
    )
    ref = dict(peak_gb=40.0, source="s", fingerprint=dict(fp))
    assert qualify.reference_comparison(ref, 41.5, fp) == dict(
        reference_gb=40.0, delta_gb=1.5, differs=[], comparable=True, source="s"
    )
    other = dict(fp, kv_codec="native16")
    got = qualify.reference_comparison(ref, 41.5, other)
    assert got["differs"] == ["kv_codec"] and got["comparable"] is False


# ------------------------------------------------------------------ registry_states.classify_kv_arm
KV_ARM_RESULTS = "6f5f5f86c5e0a4b6f70518feaf5ba41b754c7380e7599e752b61062824189efc"


def test_golden_classify_kv_arm_on_every_existing_row_pick_and_base():
    got = {}
    for pick, arm in kv.CELLS:
        for base_off in ((), (pick,), tuple(MTP_PICKS)):
            base = registry_states.apply(FIXTURE, set(base_off))
            res = registry_states.classify_kv_arm(
                kv.arm_text(arm, pick, base), FIXTURE, pick
            )
            key = f"{pick}/{arm}/{'+'.join(sorted(base_off)) or 'none'}"
            got[key] = None if res is None else [sorted(res[0]), res[1]]
    pinned(got, KV_ARM_RESULTS)


def test_golden_kv_arm_constants():
    assert registry_states.KV_ARM_ENTRIES == (kv.UNIFORM, kv.PICK_A)
    for row, want in kv.ARMS.items():
        assert registry_states.KV_ARM_ROWS[row] == want


# ------------------------------------------------------------------ qualify reports through the real driver
def _normalized(report, rig):
    text = canon(report)
    for path in (str(rig.tmp), str(rig.stack), str(rig.fork)):
        text = text.replace(path, "<tmp>")
    for label in (
        "P0",
        "P",
        "R",
    ):  # the temp fork's commits: their shas depend on the clock
        text = text.replace(getattr(rig, label), f"<{label}>")
    text = text.replace(rig.P[:8], "<p>")  # the short production ref
    text = re.sub(r'"(started|finished)":"[^"]*"', r'"\1":"<t>"', text)
    return json.loads(text)


REPORTS = {
    "uniform-off": "ac5dfc6745eae4abdd1f2861b7d1fd758d9b676ae81925a2a1b5728c7da889cb",
    "fable-off": "e82aad595d541aacc46de78f0e8e4d5bd571ff1de887da329269b628ad9c22dc",
    "uniform-kv-AF8": "aca8b1fae0e95a1d7b91475c17edb2c0f1760d43b86a8b029883c83e529b8453",
    "fable-kv-TQ4": "9f9aeb9552487c2ebced4f1aca17fb154d5d06a8260759bcbf019db425a0ba28",
    "fable-kv-on-N16": "f04931ec78d0d6984fe6a75a9916b38d53743ddeedc1991c7ae1a2a1f762f7cd",
}


@pytest.mark.parametrize("case", list(REPORTS))
def test_golden_qualify_report_bytes(rig, case):
    if case == "uniform-off":
        drv.serve(rig, kv.UNIFORM, "TQ4")
        got = rig.run("--predictor", "off", "--entry", kv.UNIFORM)
    elif case == "fable-off":
        drv.serve(rig, kv.PICK_A, "N16")
        got = rig.run("--predictor", "off", "--entry", kv.PICK_A)
    elif case == "uniform-kv-AF8":
        rig.registry(kv.arm_text("AF8", kv.UNIFORM))
        drv.serve(rig, kv.UNIFORM, "AF8")
        got = drv.kv_run(rig, kv.UNIFORM)
    elif case == "fable-kv-TQ4":
        rig.registry(kv.arm_text("TQ4", kv.PICK_A))
        drv.serve(rig, kv.PICK_A, "TQ4")
        got = drv.kv_run(rig, kv.PICK_A)
    else:
        argv_on = drv.mlx_serve_argv(
            registry_states.settings(FIXTURE)[kv.PICK_A], rig.stack
        )[1:]
        summary = rig.summary(kv.PICK_A, argv_on=argv_on)
        drv.serve(rig, kv.PICK_A, "N16", mtp_on=True)
        got = rig.run(
            "--kv-arm",
            "--predictor",
            "on",
            "--entry",
            kv.PICK_A,
            "--p369-summary",
            summary,
        )
    assert got.code == 0, got.text()
    pinned(_normalized(got.report, rig), REPORTS[case])


# ------------------------------------------------------------------ P369: plan-only output, prepare() manifest, jobs
def _plan_only(tmp_path, monkeypatch, *argv):
    monkeypatch.setenv("STACK_WORKDIR", str(tmp_path / "wd"))
    monkeypatch.setattr(
        orchestrate,
        "preflight",
        lambda args, out: dict(problems=[], power=dict(ac_ok=True), free_gb=99.0),
    )
    out = io.StringIO()
    with redirect_stdout(out):
        orchestrate.main(
            [
                "--stack-root",
                str(tmp_path / "stack"),
                "--fork-root",
                str(DEV.parent),
                "--out",
                str(tmp_path / "wd" / "p369" / "phase2" / "run"),
                "--plan-only",
                *argv,
            ]
        )
    return out.getvalue()


PLAN_ONLY_STDOUT = '{\n "preflight": {\n  "problems": [],\n  "power": {\n   "ac_ok": true\n  },\n  "free_gb": 99.0\n }\n}\n'


@pytest.mark.parametrize("pick", MTP_PICKS)
@pytest.mark.parametrize("phase", ("smoke", "full"))
def test_golden_plan_only_for_every_existing_pick(tmp_path, monkeypatch, pick, phase):
    extra = ["--smoke-summary", "x"] if phase == "full" else []
    if phase == "full":  # a full plan needs an authorizing smoke; the gate is today's
        monkeypatch.setattr(
            orchestrate, "smoke_authorization_problems", lambda s, **k: []
        )
        (tmp_path / "x").write_text("{}")
        extra = ["--smoke-summary", str(tmp_path / "x")]
        monkeypatch.setattr(
            orchestrate, "reviewed_head_problems", lambda phase, r, h: []
        )
    got = _plan_only(tmp_path, monkeypatch, "--phase", phase, "--pick", pick, *extra)
    assert got == PLAN_ONLY_STDOUT


def _campaign(tmp_path, monkeypatch, pick, registry=ALL_OFF):
    stack = tmp_path / "stack"
    stack.mkdir(exist_ok=True)
    (stack / "main_models.yaml").write_bytes(registry.encode("utf-8"))
    args = argparse.Namespace(
        stack_root=str(stack),
        fork_root=str(DEV.parent),
        out=str(tmp_path / "run"),
        pick=[pick],
        phase="smoke",
        wall_limit_s=None,
        reviewed_head=None,
        smoke_summary=None,
    )
    built = []

    def worker_argv(stack_root, path, name):
        text = Path(path).read_bytes().decode("utf-8")
        built.append((Path(stack_root).name, Path(path).name))
        return drv.mlx_serve_argv(registry_states.settings(text)[name], "/s")[1:]

    for name, value in dict(
        stack_bench_problems=lambda *a, **k: [],
        stack_bench_state=lambda *a, **k: dict(tree="t", bundles={}),
        fork_versions=lambda *a, **k: {"mlx": "0.32.4"},
        worker_argv=worker_argv,
        weights_record=lambda m: dict(hf_path=m, revision="r", files={"a": "b"}),
        pins_problems=lambda *a, **k: [],
    ).items():
        monkeypatch.setattr(orchestrate, name, value)
    return orchestrate.Campaign(args), built


MASKED = (
    "fork_head",
    "harness_tree",
    "env_keys",
    "env_dropped",
    "env_values",
    "env_digest",
)
MANIFESTS = {
    "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed": "de56226df15ad5d808d2af80e3079eebec1a153f0d48d0cf2b00867b75c4f68d",
    "Qwen3.8-27B-mlx-uniform-4bit": "5d9341957ad21864d88eedd9b3c2884fb76a8278ed223247bd38693e3c5fb8cc",
    "Ornith-1.0-35B-mlx-uniform-4bit": "a90e3f4975152f3b9ccc6d71c4983541820360b4197358a83469ddd1b592e5e2",
    "Qwen3.6-27B-Opus-Distill-OptiQ-4bit": "cbd565b872e1ccc34fe3ac78d16bdc264171dc4e83dd3a47284058bdfdbe826e",
}


def _manifest_view(camp, manifest, tmp_path):
    view = {k: v for k, v in manifest.items() if k not in MASKED}
    text = canon(view).replace(str(tmp_path), "<tmp>")
    return json.loads(text)


@pytest.mark.parametrize("pick", MTP_PICKS)
def test_golden_prepare_manifest_and_jobs_for_every_existing_pick(
    tmp_path, monkeypatch, pick
):
    camp, built = _campaign(tmp_path, monkeypatch, pick)
    manifest = camp.prepare()
    assert set(manifest) >= set(MASKED)
    assert json.loads((camp.out / "manifest.json").read_text()) == manifest
    jobs = {
        arm: camp.job(pick, f"j-{arm}", "e1c_pair", arm, [dict(name="r", timeout_s=10)])
        for arm in ("on", "off")
    }
    view = dict(
        manifest=_manifest_view(camp, manifest, tmp_path),
        built=built,
        jobs=json.loads(
            canon(jobs)
            .replace(str(tmp_path), "<tmp>")
            .replace(str(DEV.parent), "<fork>")
        ),
        binding_keys=sorted(orchestrate.binding(manifest)),
    )
    pinned(view, MANIFESTS[pick])


def test_golden_pick_tables():
    pinned(
        dict(
            picks=orchestrate.PICKS,
            production=orchestrate.PRODUCTION,
            thresholds=orchestrate.THRESHOLDS,
            smoke=orchestrate.SMOKE_THRESHOLDS,
            binding_keys=orchestrate.BINDING_KEYS,
            binding_pick_keys=orchestrate.BINDING_PICK_KEYS,
        ),
        "268b92f90cd7d6dd982364c1799cb9818ff2f632da66d4d015588a6321d57d16",
    )


@pytest.mark.parametrize("pick", MTP_PICKS)
def test_golden_argv_scan_problems_for_every_pick(pick):
    entry = registry_states.settings(FIXTURE)[pick]
    on = drv.mlx_serve_argv(entry, "/s")[1:]
    off = drv.mlx_serve_argv(
        {k: v for k, v in entry.items() if k not in qualify.MTP_KEYS}, "/s"
    )[1:]
    assert orchestrate.argv_scan_problems(pick, on) == []
    want = orchestrate.PICKS[pick]["verify_policy"]
    assert orchestrate.argv_scan_problems(pick, off) == (
        []
        if want is None
        else [f"{pick}: argv verify scan 'per_query', expected {want!r}"]
    )
