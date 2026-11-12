"""M76 B2 acceptance 2: ``qualify --kv-arm`` accepts the TQ4 ``tq4_attention`` arms X / V / XO (fork-only, dev; CPU only).

Written from SPEC v1 (``m76-b2-fork-dev-tools.md``) before the implementation, by a test writer who did not write it.  Each test
carries a ``# class:`` comment: (a) reproduces a failure the specification fixes, (b) preserves an invariant (passes on the
baseline and must keep passing), (c) new behaviour.

Real: ``qualify.main`` -> ``drive``, the gate, ``entry_problems``, ``mtp_off_registry``, ``expected_config``, ``command_problems``,
``request_problems``, ``fingerprint_of``, the report, and mlx-serve's ``_build_command`` (the sibling copy, in its own venv python:
``test_m76_b2_fe_common.real_command``).  Mocked: the host and the worker boundary (the P475a / P542 rig).  The worker record is
the P475a hand-built TQ4 manifest (15 TQ4 layers + the native last layer in the rig's 4-entry template) with the attention counter
the policy requires.

The ``--certified-by`` record is Errata v1.1's: a P369 tuple full-run ``summary.json`` with its ``manifest.json`` beside it;
``tuple`` equal in the summary, its binding and the manifest and in ``TUPLES`` naming this pick and mode; E1c and E2 PASS (and the
C199 gate: ``phase`` full, ``status`` PASS, the production binding); ``stack_head`` and ``fork_head`` (binding and manifest) equal
this run's (the rig's ``_tree_state`` reports the stack head); ``manifest["picks"][pick]["argv_on"]`` equals the served argv.
(Frozen version of these tests guessed the schema; amended in delivery 2 to the errata.)  All of it lives in ``stage2_record``.
"""

# isort: skip_file

import copy
import hashlib
import json
from pathlib import Path

import pytest

from mlx_vlm.tests import test_p475a_kv_arm as kv
from mlx_vlm.tests.test_m76_b2_fe_common import (  # also puts dev/ on sys.path
    ARM_TEXTS,
    FIXTURE,
    MTP_KEYS,
    PICK,
    UNIFORM,
    V_TEXT,
    X_TEXT,
    XO_TEXT,
    BuilderRefused,
    entry_of,
    kv_text,
    real_command,
)
from mlx_vlm.tests.test_p475a_qualify_driver import (  # noqa: F401  (rig is the pytest fixture)
    FP,
    flags_of,
    refused,
    rig,
)
from mlx_vlm.tests.test_p658_engagement_only import GOOD as GOOD_DRAFTER_MANIFEST
from p369_phase2 import kvmanifest, orchestrate, qualify, registry_states  # noqa: E402

KV_OFF = ("--kv-arm", "--predictor", "off")
KV_ON = ("--kv-arm", "--predictor", "on")
ENGAGE = KV_ON + ("--engagement-only",)
MODE_OF = {"X": "m72_exact", "V": "m72_v1", "XO": "m72_exact"}
SHA40 = "s" * 40  # the rig's stack HEAD


# ------------------------------------------------------------------ rig helpers
@pytest.fixture
def real_rig(rig, monkeypatch):
    """The P475a rig with ``qualify.full_command`` answering from mlx-serve's REAL ``_build_command``.  Fix round 1 / 3: the
    tuple's / M77's production is the rig fork's P0 (a real strict ancestor of the reviewed head, distinct from the rig's
    ``orchestrate.PRODUCTION`` = P), patched as the rig patches ``PRODUCTION``."""
    m77 = rig.P0[:8]
    monkeypatch.setattr(orchestrate, "M77_PRODUCTION", m77)
    monkeypatch.setitem(orchestrate.TUPLES["m76-x"], "production", m77)

    def full_command(stack, path, name):
        text = Path(path).read_bytes().decode("utf-8")
        rig.served.append(dict(name=name, path=Path(path), text=text))
        return real_command(stack, path, name), "m"

    monkeypatch.setattr(qualify, "full_command", full_command)
    return rig


def serve_arm(rig, text, *, mtp_on, drafter_manifest=GOOD_DRAFTER_MANIFEST):
    """The correct worker for the arm registry ``text`` (its MTP keys dropped when ``mtp_on`` is False: the derived OFF)."""
    entry = entry_of(text)
    if not mtp_on:
        for key in MTP_KEYS:
            entry.pop(key, None)
    expected = qualify.expected_config(entry)
    base = kv.valid_record(expected, kv.arm_manifest("TQ4"), sdpa_forced=640)

    def worker(job):
        rec = dict(base)
        if job.get("drafter_manifest") is True:
            rec["drafter_manifest"] = copy.deepcopy(drafter_manifest)
        return rec

    rig.worker = worker
    return expected


def off_run(rig, arm, *extra):
    rig.registry(ARM_TEXTS[arm])
    serve_arm(rig, ARM_TEXTS[arm], mtp_on=False)
    return rig.run(*KV_OFF, "--entry", PICK, *extra)


def argv_on_of(rig, text=X_TEXT):
    """The ON worker argv (``_build_command(...)[1:]``) the committed arm registry serves."""
    return real_command(rig.stack, rig.committed, PICK)[1:]


def stage2_record(rig, argv_on, tag, *, tuple_name="m76-x", production=None, **changes):
    """A stage-2 record as ``orchestrate`` writes it (Errata v1.1 + fix round 3): ``manifest.json`` with every binding key on
    the tuple's production, and a ``summary.json`` whose ``binding`` is ``orchestrate.binding(manifest)`` and whose
    ``manifest`` block agrees with it.  Mutations applied after the summary was bound: ``binding_tuple`` / ``manifest_tuple``
    (the tuple label of the summary's binding / of manifest.json), ``replace_manifest`` (keys of manifest.json replaced AFTER
    the summary was bound to it: a PASS summary paired with another manifest), ``summary_manifest`` (keys of the summary's
    ``manifest`` block), ``e1c`` / ``e2`` / ``picks`` and any other summary key; ``stack_head`` / ``fork_head`` are the heads
    bound (``None`` binds a null head).
    """
    run = rig.workdir / "p369" / "phase2" / f"m76-{tag}"
    run.mkdir(parents=True)
    prod = production or orchestrate.TUPLES["m76-x"]["production"]
    fork_head = changes.pop("fork_head", rig.R)
    heads = dict(stack_head=changes.pop("stack_head", SHA40), off_stack_head="o" * 40)
    if (
        tuple_name is not None
    ):  # Errata v1.1 (a): ``tuple`` equal in the summary, its binding and the manifest
        heads["tuple"] = tuple_name
    off = list(argv_on)
    for flag in ("--mtp-verify-scan", "--draft-kind", "--draft-model"):
        if flag in off:
            del off[off.index(flag) : off.index(flag) + 2]
    manifest = dict(
        production=prod,
        fork_head=fork_head,
        harness_tree="h" * 40,
        reviewed_head=fork_head,
        versions=dict(mlx="0.32.4"),
        env_digest="e" * 64,
        stack_bench=dict(tree="b" * 40),
        **heads,
        picks={
            PICK: dict(
                overlay_sha256=None,
                argv_on=argv_on,
                argv_off=off,
                params=dict(temperature=0.0),
                target=dict(revision="t", files={}),
                drafter=dict(revision="d", files={}),
                registry_on_sha256="1" * 64,
                registry_off_sha256="2" * 64,
            )
        },
    )
    pick = dict(outcome=dict(e1c="PASS"), e2=dict(verdict="PASS"))
    summary = dict(
        phase="full",
        status="PASS",
        binding=orchestrate.binding(manifest),
        manifest=dict(production=prod, fork_head=fork_head, **heads),
        picks={PICK: pick},
    )
    if tuple_name is not None:
        summary["tuple"] = tuple_name
    replaced = changes.pop("replace_manifest", {})
    for key, value in changes.items():
        if key == "binding_tuple":
            summary["binding"]["tuple"] = value
        elif key == "manifest_tuple":
            manifest["tuple"] = value
        elif key == "summary_manifest":
            summary["manifest"].update(value)
        elif key == "e1c":
            pick["outcome"]["e1c"] = value
        elif key == "e2":
            pick["e2"] = value if isinstance(value, dict) else dict(verdict=value)
        else:
            summary[key] = value
    manifest.update(replaced)
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run / "summary.json"


def certify(rig, path):
    return rig.run(*KV_ON, "--entry", PICK, "--certified-by", str(path))


def x_certified_rig(rig, text=X_TEXT):
    """A committed X (or ``text``) registry, its ON worker, and the matching record; returns ``(argv_on, record_path)``."""
    rig.registry(text)
    serve_arm(rig, text, mtp_on=True)
    argv_on = argv_on_of(rig)
    return argv_on, stage2_record(rig, argv_on, "good")


def replace_value(argv, flag, value):
    out = list(argv)
    out[out.index(flag) + 1] = value
    return out


def drop_flag(argv, flag):
    i = argv.index(flag)
    return argv[:i] + argv[i + 2 :]


def swap_pairs(argv, a, b):
    i, j = argv.index(a), argv.index(b)
    out = list(argv)
    out[i : i + 2], out[j : j + 2] = argv[j : j + 2], argv[i : i + 2]
    return out


# ================================================================== the registry layer: SERVING, expected_config, OFF derivation
# class: (a) (qualify refuses the key as unknown today)
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_expected_config_accepts_the_mode_key_and_spells_its_flag(arm):
    expected = qualify.expected_config(entry_of(ARM_TEXTS[arm]))
    assert expected["flags"]["--tq4-attention"] == MODE_OF[arm]
    assert expected["flags"]["--kv-bits"] == "4"
    assert expected["kv_bits"] == 4 and expected["kv_quant_scheme"] == "turboquant"


# class: (b) (a spelling outside the reviewed table still refuses as an unknown key)
@pytest.mark.parametrize(
    "key", ("tq4_attn", "tq4-attention", "tq4_attention_mode", "tq4")
)
def test_a_misspelt_mode_key_is_still_an_unknown_key(key):
    entry = entry_of(X_TEXT)
    entry[key] = entry.pop("tq4_attention", "m72_exact")
    with pytest.raises(qualify.UnknownKey):
        qualify.expected_config(entry)


# class: (b) (an entry without the key has no mode flag)
def test_an_entry_without_the_mode_has_no_mode_flag():
    flags = qualify.expected_config(entry_of(FIXTURE))["flags"]
    assert "--tq4-attention" not in flags


# class: (c) (the OFF derivation strips the three MTP keys and keeps the mode)
@pytest.mark.parametrize("arm, mode", (("X", "m72_exact"), ("V", "m72_v1")))
def test_the_off_derivation_of_an_arm_strips_the_mtp_keys_and_keeps_the_mode(arm, mode):
    derived = qualify.mtp_off_registry(ARM_TEXTS[arm], PICK)
    entry = registry_states.settings(derived)[PICK]
    assert not any(k in entry for k in MTP_KEYS)
    assert entry["tq4_attention"] == mode and entry["kv_bits"] == 4
    want = {k: v for k, v in entry_of(ARM_TEXTS[arm]).items() if k not in MTP_KEYS}
    assert entry == want
    for name in registry_states.settings(
        ARM_TEXTS[arm]
    ):  # every other entry byte-for-byte as committed
        if name != PICK:
            assert (
                registry_states.settings(derived)[name]
                == registry_states.settings(ARM_TEXTS[arm])[name]
            )


# class: (b) (the derived OFF registry of X / V is what mlx-serve at 3b02e4c serves: no refusal, no joint scan, the mode kept)
@pytest.mark.parametrize("arm", ("X", "V"))
def test_mlx_serve_serves_the_derived_off_registry(tmp_path, arm):
    derived = tmp_path / "derived.yaml"
    derived.write_text(qualify.mtp_off_registry(ARM_TEXTS[arm], PICK))
    cmd = real_command(tmp_path / "stack", derived, PICK)
    assert cmd[cmd.index("--tq4-attention") + 1] == MODE_OF[arm]
    assert not {"--draft-kind", "--draft-model", "--mtp-verify-scan"} & set(cmd)


# class: (a) (fails on the baseline as the unknown key ``tq4_attention``; the derived registry then satisfies the two-way check)
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_the_real_argv_of_each_arm_passes_the_two_way_command_check(tmp_path, arm):
    path = tmp_path / "r.yaml"
    path.write_text(
        ARM_TEXTS[arm]
        if arm == "XO"
        else qualify.mtp_off_registry(ARM_TEXTS[arm], PICK)
    )
    stack = tmp_path / "stack"
    cmd = real_command(stack, path, PICK)
    expected = qualify.expected_config(registry_states.settings(path.read_text())[PICK])
    assert qualify.command_problems(expected, cmd, stack / ".venv" / "bin") == []


# class: (c) (the two-way check refuses an argv whose mode flag is missing or changed, with the mode expected)
def test_the_command_check_binds_the_mode_flag(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text(XO_TEXT)
    stack = tmp_path / "stack"
    cmd = real_command(stack, path, PICK)
    expected = qualify.expected_config(entry_of(XO_TEXT))
    venv = stack / ".venv" / "bin"
    assert qualify.command_problems(expected, cmd, venv) == []
    assert qualify.command_problems(
        expected,
        cmd[: cmd.index("--tq4-attention")] + cmd[cmd.index("--tq4-attention") + 2 :],
        venv,
    )
    assert qualify.command_problems(
        expected, replace_value(cmd, "--tq4-attention", "m72_v1"), venv
    )


# ================================================================== the expected manifest and the attention rule
# class: (c) (the mode does not change cache classes: the TQ4 manifest, the native last layer, forced calls under fused_v1)
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_the_expected_manifest_is_todays_tq4_manifest(arm):
    expected = qualify.expected_config(entry_of(ARM_TEXTS[arm]))
    want = kvmanifest.expected_manifest(
        kv.t415._template(), qualify.expected_kv(expected)
    )
    assert want == kvmanifest.expected_manifest(
        kv.t415._template(),
        qualify.expected_kv(
            qualify.expected_config(entry_of(kv.arm_text("TQ4", PICK)))
        ),
    )
    assert [m["kind"] for m in want].count("native") >= 1 and any(
        m["kind"] == "tq" for m in want
    )
    assert (
        qualify.request_problems(
            kv.valid_record(expected, kv.arm_manifest("TQ4"), sdpa_forced=640), expected
        )
        == []
    )


# class: (c) (the sdpa_forced rule is unchanged: the native leaf requires forced calls under fused_v1)
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_zero_forced_calls_fail_a_mode_arm(arm):
    expected = qualify.expected_config(entry_of(ARM_TEXTS[arm]))
    problems = qualify.request_problems(
        kv.valid_record(expected, kv.arm_manifest("TQ4"), sdpa_forced=0), expected
    )
    assert problems and all("sdpa_forced" in p for p in problems)


# ================================================================== --predictor off: X, V, XO complete through the real driver
# class: (c) (a on the baseline: refused as an unknown key)
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_predictor_off_completes_on_each_arm_commit(real_rig, arm):
    got = off_run(real_rig, arm)
    assert got.completed(PICK), got.text()
    state = got.state(PICK)
    assert state["arm"] == ("served_off" if arm == "XO" else "mtp_off")
    assert got.report["kv_arm"]["entry"] == PICK
    assert {k: tuple(v) for k, v in got.report["kv_arm"]["delta"].items()} == {
        "kv_bits": (0, 4),
        "tq4_attention": (None, MODE_OF[arm]),
    }


# class: (c) (the served argv is mlx-serve's own spelling of the derived / committed registry and carries the mode)
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_the_off_argv_carries_the_mode_and_no_mtp_flags(real_rig, arm):
    got = off_run(real_rig, arm)
    assert got.completed(PICK), got.text()
    argv = real_rig.specs[0]["argv"]
    flags = flags_of(argv)
    assert flags["--tq4-attention"] == MODE_OF[arm]
    assert flags["--kv-bits"] == "4" and flags["--kv-quant-scheme"] == "turboquant"
    assert (
        flags["--attention-policy"] == "fused_v1"
        and flags["--quantized-kv-start"] == "0"
    )
    assert not {"--draft-kind", "--draft-model", "--mtp-verify-scan"} & set(flags)
    job = real_rig.specs[0]
    assert job["expect"]["drafter"] is False and job["expect"]["verify_policy"] is None
    assert job["expect"]["kv"] == kvmanifest.kv_from_argv(argv)
    assert job["expect"]["kv"]["kv_bits"] == 4.0
    (seen,) = real_rig.expecteds
    assert seen["flags"].get("--tq4-attention") == MODE_OF[arm]
    served = real_rig.served[0]
    assert (served["path"] == real_rig.committed) == (
        arm == "XO"
    )  # X / V: the derived registry; XO: as committed
    if arm != "XO":
        assert served["path"].name.startswith("mtp-off-")


# class: (c) (the arm commit is judged on its worker's manifest: a wrong manifest fails it)
@pytest.mark.parametrize("arm", ("X", "XO"))
def test_a_native16_manifest_fails_a_mode_arm(real_rig, arm):
    real_rig.registry(ARM_TEXTS[arm])
    expected = serve_arm(real_rig, ARM_TEXTS[arm], mtp_on=False)
    wrong = kv.valid_record(expected, kv.arm_manifest("N16"), sdpa_forced=640)
    real_rig.worker = lambda job: wrong
    got = real_rig.run(*KV_OFF, "--entry", PICK)
    assert got.code == 1 and got.state(PICK)["status"] == "FAILED", got.text()
    assert any("layer" in p for p in got.state(PICK)["problems"])


# class: (c) (zero forced SDPA calls fail the arm through the real driver: the rule is the native leaf's)
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_zero_forced_calls_fail_the_arm_through_the_driver(real_rig, arm):
    real_rig.registry(ARM_TEXTS[arm])
    expected = serve_arm(real_rig, ARM_TEXTS[arm], mtp_on=False)
    real_rig.worker = lambda job: kv.valid_record(
        expected, kv.arm_manifest("TQ4"), sdpa_forced=0
    )
    got = real_rig.run(*KV_OFF, "--entry", PICK)
    assert got.code == 1 and got.state(PICK)["status"] == "FAILED"
    assert any("sdpa_forced" in p for p in got.state(PICK)["problems"])


# class: (c) (every existing check still applies around a mode: a non-KV key beside the mode refuses before any job; the
# control needs the arm, so the baseline fails on the control)
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
@pytest.mark.parametrize(
    "key, value",
    (
        ("prefill_step_size", 1024),
        ("quantized_kv_start", 5000),
        ("max_kv_cache_size", 131072),
    ),
)
def test_a_non_kv_change_beside_the_mode_refuses(real_rig, arm, key, value):
    got = off_run(real_rig, arm)
    assert got.completed(PICK), got.text()  # control
    real_rig.registry(kv.set_kv(ARM_TEXTS[arm], PICK, **{key: value}))
    refused(real_rig, real_rig.run(*KV_OFF, "--entry", PICK))


# class: (c) (a mode with another row's width / scheme / value refuses through the driver; control = X completes)
@pytest.mark.parametrize(
    "text, entry",
    (
        (kv_text(PICK, 8, "turboquant", mode="m72_exact"), PICK),
        (kv_text(PICK, 4, "uniform", mode="m72_exact"), PICK),
        (kv_text(PICK, 4, mode="m72_v2"), PICK),
        (kv_text(PICK, 4, mode="false"), PICK),
        (kv_text(PICK, 4, mode="null"), PICK),
        (kv_text(UNIFORM, 4, "turboquant", mode="m72_exact"), UNIFORM),
    ),
    ids=("tq8", "af4", "unknown-mode", "false", "null", "uniform-pick"),
)
def test_a_bad_mode_arm_refuses_before_any_job(real_rig, text, entry):
    assert off_run(real_rig, "X").completed(PICK)  # control
    real_rig.registry(text)
    refused(real_rig, real_rig.run(*KV_OFF, "--entry", entry))


# ================================================================== the fingerprint
REC = dict(
    versions={"mlx": "0.32.4"},
    snapshots={"--model": "rev"},
    weights={"--model": "digest"},
    harness="h" * 40,
    generation_tokens=2048,
    prompt={"sized": {"sha256": "p" * 64}},
    memory={"set_memory_limit": [1], "set_cache_limit": [2]},
)


def fingerprint(entry):
    return qualify.fingerprint_of(REC, qualify.expected_config(entry))


# class: (c)
def test_the_fingerprint_tells_x_from_v():
    x, v = fingerprint(entry_of(X_TEXT)), fingerprint(entry_of(V_TEXT))
    assert x != v
    assert x["tq4_attention"] == "m72_exact" and v["tq4_attention"] == "m72_v1"
    assert {k: w for k, w in x.items() if k != "tq4_attention"} == {
        k: w for k, w in v.items() if k != "tq4_attention"
    }


# class: (c) (the mode is recorded for the MTP-OFF arms too: XO is X with MTP off, nothing else added)
def test_the_fingerprint_of_xo_is_x_with_mtp_off():
    x, xo = fingerprint(entry_of(X_TEXT)), fingerprint(entry_of(XO_TEXT))
    assert (
        xo["tq4_attention"] == "m72_exact" and xo["mtp"] == "off" and x["mtp"] == "on"
    )
    assert {k: v for k, v in xo.items() if k != "mtp"} == {
        k: v for k, v in x.items() if k != "mtp"
    }


# class: (c) (the mode key is added ONLY when there is a mode)
def test_the_mode_key_is_absent_from_every_fingerprint_without_a_mode():
    for pick, arm in kv.CELLS:
        for on in (True, False):
            assert "tq4_attention" not in fingerprint(
                kv.entry_for(arm, pick, mtp_on=on)
            ), (pick, arm, on)
    assert "tq4_attention" not in fingerprint(entry_of(FIXTURE))


# class: (b) GOLDEN: the baseline's fingerprints and expected configurations (12 hex of sha256 of the canonical json) of every
# existing arm cell, MTP-ON and MTP-OFF, and of the four committed picks
GOLDEN_FINGERPRINTS = {
    "A-N16-on": "8e8ee71aa90d",
    "A-N16-off": "fac83d60aae3",
    "A-AF8-on": "9114976191d3",
    "A-AF8-off": "292c283e15f4",
    "A-TQ8-on": "ada49db00e8c",
    "A-TQ8-off": "7ed910e575f1",
    "A-AF4-on": "a1f199e14ec4",
    "A-AF4-off": "851e75a6246d",
    "A-TQ4-on": "72a03b54c3d7",
    "A-TQ4-off": "292afecc729e",
    "U-N16-on": "ed3d8c86217c",
    "U-N16-off": "a878ef6fff7c",
    "U-AF8-on": "7bea1d197304",
    "U-AF8-off": "0d5845ab2141",
    "U-TQ8-on": "128694adbcbd",
    "U-TQ8-off": "6986ed6a0c2f",
    "U-AF4-on": "d988246b0151",
    "U-AF4-off": "59b9f4ecc4c9",
    "U-TQ4-on": "ff62fa531316",
    "U-TQ4-off": "310510df2fee",
    "fixture-Ornith-1.0-35B-mlx-uniform-4bit": "ed3d8c86217c",
    "fixture-Qwen3.6-27B-Opus-Distill-OptiQ-4bit": "ff62fa531316",
    "fixture-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit": "a878ef6fff7c",
    "fixture-Qwen3.8-27B-mlx-uniform-4bit": "ff62fa531316",
    "fixture-Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed": "8e8ee71aa90d",
}
GOLDEN_EXPECTED = {  # of expected_config minus a possible new ``tq4_attention`` key (the spec does not pin that dict's shape)
    "A-N16-on": "ff2688bcc31d",
    "A-N16-off": "1e6ab796d986",
    "A-AF8-on": "4245efc6b0d7",
    "A-AF8-off": "cc977236376f",
    "A-TQ8-on": "ca7b3e49e2c4",
    "A-TQ8-off": "ac20e8543173",
    "A-AF4-on": "f9351ec8111e",
    "A-AF4-off": "2b9088e5711a",
    "A-TQ4-on": "6f912361871c",
    "A-TQ4-off": "d76c2adb7ee5",
    "U-N16-on": "4904d3bf3958",
    "U-N16-off": "3332d8688dc6",
    "U-AF8-on": "9630c3b82845",
    "U-AF8-off": "528eb3b03da5",
    "U-TQ8-on": "e3aa95bdcce0",
    "U-TQ8-off": "029a31d2e0c4",
    "U-AF4-on": "5ca23343c26e",
    "U-AF4-off": "402c64306e2b",
    "U-TQ4-on": "2098a0e828ed",
    "U-TQ4-off": "85f15c8e0b76",
    "fixture-Ornith-1.0-35B-mlx-uniform-4bit": "56944583ec53",
    "fixture-Qwen3.6-27B-Opus-Distill-OptiQ-4bit": "d1a3ff2404d0",
    "fixture-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit": "4a1c806a9f95",
    "fixture-Qwen3.8-27B-mlx-uniform-4bit": "2098a0e828ed",
    "fixture-Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed": "ff2688bcc31d",
}


def _h(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _existing_entries():
    for pick, arm in kv.CELLS:
        for on in (True, False):
            yield f"{'A' if pick == PICK else 'U'}-{arm}-{'on' if on else 'off'}", kv.entry_for(
                arm, pick, mtp_on=on
            )
    for name, entry in registry_states.settings(FIXTURE).items():
        yield f"fixture-{name}", dict(entry)


# class: (b)
def test_golden_fingerprints_of_every_existing_entry_are_unchanged():
    got = {label: _h(fingerprint(e)) for label, e in _existing_entries()}
    assert got == GOLDEN_FINGERPRINTS, json.dumps(got, indent=1)


# class: (b)
def test_golden_expected_configs_of_every_existing_entry_are_unchanged():
    got = {
        label: _h(
            {
                k: v
                for k, v in qualify.expected_config(e).items()
                if k != "tq4_attention"
            }
        )
        for label, e in _existing_entries()
    }
    assert got == GOLDEN_EXPECTED, json.dumps(got, indent=1)


# ================================================================== --predictor on with a delta: the certified-by binding
# class: (c) (the refusal stays as today without a record)
@pytest.mark.parametrize("text", (X_TEXT, V_TEXT), ids=("X", "V"))
def test_predictor_on_without_certified_by_refuses(real_rig, text):
    real_rig.registry(text)
    serve_arm(real_rig, text, mtp_on=True)
    got = real_rig.run(*KV_ON, "--entry", PICK)
    refused(real_rig, got)
    assert "uncertified" in (got.refused or ""), got.text()


# class: (c) (invariant of the certified-argv binding, only exercisable once X is accepted: a --p369-summary PASS record,
# argv equal to X's, never certifies a KV delta)
def test_a_p369_summary_does_not_certify_an_x_run(real_rig):
    real_rig.registry(X_TEXT)
    serve_arm(real_rig, X_TEXT, mtp_on=True)
    summary = real_rig.summary(PICK, argv_on=argv_on_of(real_rig))
    refused(real_rig, real_rig.run(*KV_ON, "--entry", PICK, "--p369-summary", summary))


# class: (c) (the certifying run on X, bound to its record: completes, ON as committed, certified_by = path + sha256)
def test_x_certifying_run_with_a_matching_record_completes_and_reports_certified_by(
    real_rig,
):
    argv_on, record = x_certified_rig(real_rig)
    got = certify(real_rig, record)
    assert got.completed(PICK), got.text()
    state = got.state(PICK)
    assert state["arm"] == "served_on" and "overlay_sha256" not in state
    assert [s["path"] for s in real_rig.served] == [real_rig.committed]
    assert real_rig.specs[0]["argv"] == argv_on  # the served argv IS the certified one
    assert not list(got.out.glob("overlay-*")) and not list(got.out.glob("mtp-off-*"))
    flags = flags_of(real_rig.specs[0]["argv"])
    assert (
        flags["--tq4-attention"] == "m72_exact"
        and flags["--mtp-verify-scan"] == "joint_v1"
    )
    assert flags["--draft-kind"] == "mtp" and "--draft-model" in flags
    assert real_rig.specs[0]["expect"]["drafter"] is True
    assert real_rig.specs[0]["expect"]["verify_policy"] == "joint_v1"
    cert = got.report["certified_by"]
    assert Path(cert["path"]).resolve() == record.resolve()
    assert cert["sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    assert got.report["kv_arm"]["entry"] == PICK and not got.report.get(
        "engagement_only"
    )
    assert got.report.get("certifies") is not False


# class: (c) (a certified run's reference comparison stays labelled non-comparable)
def test_the_certified_run_is_labelled_non_comparable(real_rig, monkeypatch):
    ref = dict(peak_gb=40.0, source="synthetic", source_path="p", fingerprint=dict(FP))
    monkeypatch.setattr(qualify, "load_references", lambda path, **k: {PICK: ref})
    argv_on, record = x_certified_rig(real_rig)
    got = certify(real_rig, record)
    assert got.completed(PICK), got.text()
    assert got.state(PICK)["reference"]["label"] == "non-comparable: KV arm"


# class: (c) (a record that fails check (a): not a PASS m76-x report)
RECORD_A = [
    ("tuple-absent", dict(tuple_name=None)),
    ("tuple-p369", dict(tuple_name="p369")),
    ("tuple-other", dict(tuple_name="m76-v")),
    (
        "tuple-binding-differs",
        dict(binding_tuple="p369"),
    ),  # Errata v1.1 (a): equal in summary, binding AND manifest
    ("tuple-manifest-differs", dict(manifest_tuple="p369")),
    ("tuple-binding-missing", dict(binding_tuple=None)),
    ("tuple-manifest-missing", dict(manifest_tuple=None)),
    ("status-fail", dict(status="FAIL")),
    ("status-incomplete", dict(status="INCOMPLETE")),
    ("status-missing", dict(status=None)),
    ("phase-smoke", dict(phase="smoke")),
    ("e1c-fail", dict(e1c="FAIL")),
    ("e1c-abort", dict(e1c="ABORT")),
    ("e1c-lowercase", dict(e1c="pass")),
    ("e2-fail", dict(e2="FAIL")),
    ("e2-incomplete", dict(e2="INCOMPLETE")),
    ("e2-missing", dict(e2={})),
    ("e2-bool", dict(e2=True)),
    ("no-picks", dict(picks={})),
    (
        "other-pick",
        dict(picks={UNIFORM: dict(outcome=dict(e1c="PASS"), e2=dict(verdict="PASS"))}),
    ),
]


@pytest.mark.parametrize("label, changes", RECORD_A, ids=[a for a, _ in RECORD_A])
def test_a_record_failing_a_is_refused(real_rig, label, changes):
    argv_on, record = x_certified_rig(real_rig)
    assert certify(real_rig, record).completed(
        PICK
    )  # control: the same rig, the matching record
    refused(
        real_rig, certify(real_rig, stage2_record(real_rig, argv_on, label, **changes))
    )


# class: (c) (a path that is no record at all)
@pytest.mark.parametrize(
    "kind", ("missing", "not-json", "list", "empty-file", "directory")
)
def test_an_unreadable_or_malformed_record_is_refused(real_rig, kind):
    argv_on, record = x_certified_rig(real_rig)
    assert certify(real_rig, record).completed(PICK)  # control
    base = real_rig.workdir / "p369" / "phase2" / "m76-junk"
    base.mkdir(parents=True)
    path = base / "summary.json"
    if kind == "not-json":
        path.write_text("{not json")
    elif kind == "list":
        path.write_text("[1, 2]")
    elif kind == "empty-file":
        path.write_text("")
    elif kind == "directory":
        path = base
    refused(real_rig, certify(real_rig, path))


# class: (c) check (b): the stack root commit and the fork head are this run's
@pytest.mark.parametrize(
    "label, changes",
    (
        ("stack-head", dict(stack_head="t" * 40)),
        ("stack-head-missing", dict(stack_head=None)),
        ("fork-head", dict(fork_head="e" * 40)),
        ("fork-head-missing", dict(fork_head=None)),
        ("stack-head-short", dict(stack_head="s" * 8)),
    ),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_record_of_another_commit_is_refused(real_rig, label, changes):
    argv_on, record = x_certified_rig(real_rig)
    assert certify(real_rig, record).completed(PICK)  # control
    refused(
        real_rig, certify(real_rig, stage2_record(real_rig, argv_on, label, **changes))
    )


# class: (c) check (c): the ON worker argv equals this run's served argv EXACTLY
def _argv_mutations(argv):
    return {
        "mode-value": replace_value(argv, "--tq4-attention", "m72_v1"),
        "step-size": replace_value(argv, "--prefill-step-size", "1024"),
        "kv-bits": replace_value(argv, "--kv-bits", "8"),
        "mode-flag-missing": drop_flag(argv, "--tq4-attention"),
        "scan-missing": drop_flag(argv, "--mtp-verify-scan"),
        "drafter-missing": drop_flag(argv, "--draft-model"),
        "extra-flag": argv + ["--extra", "1"],
        "order": swap_pairs(argv, "--kv-bits", "--prefill-step-size"),
        "truncated": argv[:-2],
        "empty": [],
        "with-executable": ["/stack/.venv/bin/mlx_vlm.server"] + argv,
        "off-argv": drop_flag(
            drop_flag(drop_flag(argv, "--mtp-verify-scan"), "--draft-kind"),
            "--draft-model",
        ),
    }


@pytest.mark.parametrize(
    "label",
    sorted(
        _argv_mutations(
            ["--a", "1"]
            + [
                "--tq4-attention",
                "x",
                "--prefill-step-size",
                "2",
                "--kv-bits",
                "4",
                "--mtp-verify-scan",
                "j",
                "--draft-kind",
                "m",
                "--draft-model",
                "d",
            ]
        )
    ),
)
def test_a_record_with_another_on_argv_is_refused(real_rig, label):
    argv_on, record = x_certified_rig(real_rig)
    assert certify(real_rig, record).completed(PICK)  # control
    bad = stage2_record(real_rig, _argv_mutations(argv_on)[label], label)
    got = certify(real_rig, bad)
    refused(real_rig, got)


# class: (c) (a record without the manifest beside it carries no argv: refused)
def test_a_record_without_a_manifest_is_refused(real_rig):
    argv_on, record = x_certified_rig(real_rig)
    assert certify(real_rig, record).completed(PICK)  # control
    lonely = stage2_record(real_rig, argv_on, "lonely")
    (lonely.parent / "manifest.json").unlink()
    refused(real_rig, certify(real_rig, lonely))


# class: (c) (the record binds to X only: V's committed MTP-ON arm with X's record, whose ON argv differs, is refused)
def test_the_x_record_does_not_certify_v(real_rig):
    argv_on_x, record = x_certified_rig(real_rig)
    real_rig.registry(V_TEXT)
    serve_arm(real_rig, V_TEXT, mtp_on=True)
    assert argv_on_of(real_rig) != argv_on_x
    refused(real_rig, certify(real_rig, record))


# class: (b) (no other KV arm gains a way to certify: `Qwen3.8-27B-mlx-uniform-4bit`'s AF8 MTP-ON, even with a record carrying ITS argv)
def test_certified_by_does_not_open_the_other_arms(real_rig):
    text = kv.arm_text("AF8", UNIFORM)
    real_rig.registry(text)
    expected = qualify.expected_config(entry_of(text, UNIFORM))
    real_rig.worker = lambda job: kv.valid_record(
        expected, kv.arm_manifest("AF8"), sdpa_forced=0
    )
    argv = real_command(real_rig.stack, real_rig.committed, UNIFORM)[1:]
    record = stage2_record(real_rig, argv, "af8")
    refused(
        real_rig,
        real_rig.run(*KV_ON, "--entry", UNIFORM, "--certified-by", str(record)),
    )


# class: (c) (the MTP-OFF commit has nothing to certify: --predictor on stays refused on XO, record or not; needs XO accepted)
def test_predictor_on_on_the_xo_commit_stays_refused(real_rig):
    real_rig.registry(XO_TEXT)
    serve_arm(real_rig, XO_TEXT, mtp_on=False)
    argv = real_command(real_rig.stack, real_rig.committed, PICK)[1:]
    refused(real_rig, certify(real_rig, stage2_record(real_rig, argv, "xo")))
    refused(real_rig, real_rig.run(*KV_ON, "--entry", PICK))


# class: (c) (an OFF run is no certification: its report carries no certified_by)
def test_a_predictor_off_report_has_no_certified_by(real_rig):
    got = off_run(real_rig, "X")
    assert got.completed(PICK) and "certified_by" not in got.report


# ================================================================== --engagement-only: unchanged, now on X / V
# class: (c)
@pytest.mark.parametrize("text, arm", ((X_TEXT, "X"), (V_TEXT, "V")), ids=("X", "V"))
def test_engagement_only_on_x_and_v_completes_without_a_record(real_rig, text, arm):
    real_rig.registry(text)
    serve_arm(real_rig, text, mtp_on=True)
    got = real_rig.run(*ENGAGE, "--entry", PICK)
    assert got.completed(PICK), got.text()
    assert got.report["engagement_only"] is True and got.report["certifies"] is False
    assert "certified_by" not in got.report
    state = got.state(PICK)
    assert (
        state["arm"] == "served_on"
        and state["drafter_manifest"] == GOOD_DRAFTER_MANIFEST
    )
    flags = flags_of(real_rig.specs[0]["argv"])
    assert flags["--tq4-attention"] == MODE_OF[arm] and flags["--draft-kind"] == "mtp"
    assert {k: tuple(v) for k, v in got.report["kv_arm"]["delta"].items()} == {
        "kv_bits": (0, 4),
        "tq4_attention": (None, MODE_OF[arm]),
    }


# class: (c) (XO has no drafter: engagement-only on the OFF commit stays refused, as on every OFF arm; control = X)
def test_engagement_only_on_the_xo_commit_stays_refused(real_rig):
    real_rig.registry(X_TEXT)
    serve_arm(real_rig, X_TEXT, mtp_on=True)
    assert real_rig.run(*ENGAGE, "--entry", PICK).completed(PICK)  # control
    real_rig.registry(XO_TEXT)
    refused(real_rig, real_rig.run(*ENGAGE, "--entry", PICK))


# class: (c) (engagement-only keeps its kv-arm gate under a mode: an unknown mode refuses; control = X)
def test_engagement_only_refuses_an_unknown_mode(real_rig):
    real_rig.registry(X_TEXT)
    serve_arm(real_rig, X_TEXT, mtp_on=True)
    assert real_rig.run(*ENGAGE, "--entry", PICK).completed(PICK)  # control
    real_rig.registry(kv_text(PICK, 4, mode="m72_v9"))
    refused(real_rig, real_rig.run(*ENGAGE, "--entry", PICK))


# ================================================================== the no-flag path with a mode is no accepted state
# class: (c) (without --kv-arm the 16-state rule stands: an X registry is refused with --predictor on, no job runs; the worker
# record needs X accepted by qualify.expected_config, so the baseline errors in setup)
def test_without_kv_arm_a_mode_registry_is_not_an_accepted_state(real_rig):
    real_rig.registry(X_TEXT)
    serve_arm(real_rig, X_TEXT, mtp_on=False)
    got = real_rig.run(
        "--predictor", "on", "--entry", PICK
    )  # restore needs the 16 states
    assert got.code not in (0, None) and real_rig.jobs == []


# ================================================================== delivery 3: the fix round (production pin, record binding, message)
@pytest.fixture
def diffed(real_rig, monkeypatch):
    """The rig with the preflight's production-diff check REAL (``orchestrate.production_diff_problem`` on the rig's fork,
    against the ref the run resolved and handed to the preflight); the rest of the host preflight stays mocked.  Returns
    ``(rig, refs)``: the production refs the preflight saw."""
    refs = []

    def preflight(args, out):
        refs.append(args.production_ref)
        diff = orchestrate.production_diff_problem(args.fork_root, args.production_ref)
        return dict(problems=[diff] if diff else [])

    monkeypatch.setattr(orchestrate, "preflight", preflight)
    return real_rig, refs


# class: (a) (fix 1: a tq4_attention arm attested the PRODUCTION pin, not M77's)
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_a_mode_arm_run_defaults_to_m77s_production(diffed, arm):
    rig, refs = diffed
    got = off_run(rig, arm)
    assert got.completed(PICK), got.text()
    assert (
        refs == [rig.P0] and got.report["production"] == rig.P0
    )  # M77's, resolved: not PRODUCTION's (rig.P)
    assert rig.P0 != rig.P


# class: (a)
def test_an_engagement_run_defaults_to_m77s_production(diffed):
    rig, refs = diffed
    rig.registry(X_TEXT)
    serve_arm(rig, X_TEXT, mtp_on=True)
    got = rig.run(*ENGAGE, "--entry", PICK)
    assert got.completed(PICK), got.text()
    assert refs == [rig.P0] and got.report["production"] == rig.P0


# class: (a)
def test_a_certifying_run_defaults_to_m77s_production(diffed):
    rig, refs = diffed
    argv_on, record = x_certified_rig(rig)
    got = certify(rig, record)
    assert got.completed(PICK), got.text()
    assert refs == [rig.P0] and got.report["production"] == rig.P0


# class: (b) (an explicit --production-ref still wins on a mode arm)
@pytest.mark.parametrize("arm", ("X", "XO"))
def test_an_explicit_production_ref_wins_on_a_mode_arm(diffed, arm):
    rig, refs = diffed
    got = off_run(rig, arm, "--production-ref", rig.P[:8])
    assert got.completed(PICK), got.text()
    assert refs == [rig.P] and got.report["production"] == rig.P


# class: (b) (an explicit ref equal to the PRODUCTION default string is still explicit: P wins over M77 on a mode arm)
def test_an_explicit_ref_equal_to_the_default_string_is_still_explicit(diffed):
    rig, refs = diffed
    got = off_run(rig, "X", "--production-ref", orchestrate.PRODUCTION)
    assert got.completed(PICK), got.text()
    assert refs == [rig.P]


# class: (b) (existing arms: `Qwen3.8-27B-mlx-uniform-4bit`'s KV arm still attests PRODUCTION; no mode, no change)
def test_an_existing_arm_still_defaults_to_production(diffed):
    rig, refs = diffed
    text = kv.arm_text("AF8", UNIFORM)
    rig.registry(text)
    entry = {
        k: v for k, v in entry_of(text, UNIFORM).items() if k not in MTP_KEYS
    }  # the derived MTP-OFF entry
    expected = qualify.expected_config(entry)
    rig.worker = lambda job: kv.valid_record(
        expected, kv.arm_manifest("AF8"), sdpa_forced=0
    )
    got = rig.run(*KV_OFF, "--entry", UNIFORM)
    assert got.completed(UNIFORM), got.text()
    assert refs == [rig.P] and got.report["production"] == rig.P


# class: (a) (the diff is the REAL one, against M77's production: production code changed since it refuses, naming it)
def test_the_real_production_diff_runs_against_m77s_production(diffed):
    rig, refs = diffed
    rig.registry(X_TEXT)
    serve_arm(rig, X_TEXT, mtp_on=False)
    (rig.fork / "mlx_vlm").mkdir()
    (rig.fork / "mlx_vlm" / "changed.py").write_text("x = 1\n")
    from mlx_vlm.tests.test_p542_qualify_any import _git

    _git(rig.fork, "add", "mlx_vlm/changed.py")
    got = rig.run(*KV_OFF, "--entry", PICK)
    refused(rig, got)
    assert refs == [rig.P0] and f"production code differs from {rig.P0}" in got.text()


# class: (a) (fix 3: the summary is bound to THIS manifest; a PASS summary beside a replaced manifest.json refuses)
@pytest.mark.parametrize(
    "label, replaced",
    (
        ("harness-tree", dict(harness_tree="x" * 40)),
        ("env-digest", dict(env_digest="y" * 64)),
        ("versions", dict(versions=dict(mlx="0.0.0"))),
        ("stack-bench", dict(stack_bench=dict(tree="z" * 40))),
        (
            "registry-sha",
            dict(picks=None),
        ),  # filled below: the pick's registry hashes replaced
    ),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_pass_summary_beside_a_replaced_manifest_is_refused(
    real_rig, label, replaced
):
    argv_on, record = x_certified_rig(real_rig)
    assert certify(real_rig, record).completed(PICK)  # control
    if replaced.get("picks", 0) is None:
        probe = json.loads((record.parent / "manifest.json").read_text())["picks"]
        probe[PICK]["registry_on_sha256"] = "9" * 64
        replaced = dict(picks=probe)
    bad = stage2_record(real_rig, argv_on, label, replace_manifest=replaced)
    refused(real_rig, certify(real_rig, bad))


# class: (a) (the replaced manifest keeps the ON argv, so the argv binding alone would accept it: the argv_off differs)
def test_a_replaced_manifest_with_the_same_on_argv_is_refused(real_rig):
    argv_on, record = x_certified_rig(real_rig)
    assert certify(real_rig, record).completed(PICK)  # control
    picks = json.loads((record.parent / "manifest.json").read_text())["picks"]
    picks[PICK]["argv_off"] = picks[PICK]["argv_off"] + ["--extra", "1"]
    refused(
        real_rig,
        certify(
            real_rig,
            stage2_record(
                real_rig, argv_on, "offargv", replace_manifest=dict(picks=picks)
            ),
        ),
    )


# class: (a) (the summary's own manifest block must agree with manifest.json)
@pytest.mark.parametrize(
    "block",
    (
        dict(fork_head="e" * 40),
        dict(stack_head="t" * 40),
        dict(off_stack_head="p" * 40),
        dict(production="b26fa7c9"),
        dict(tuple="p369"),
    ),
    ids=lambda b: next(iter(b)),
)
def test_a_summary_manifest_block_that_differs_from_the_manifest_is_refused(
    real_rig, block
):
    argv_on, record = x_certified_rig(real_rig)
    assert certify(real_rig, record).completed(PICK)  # control
    refused(
        real_rig,
        certify(
            real_rig, stage2_record(real_rig, argv_on, "block", summary_manifest=block)
        ),
    )


# class: (a) (the record's production must be the tuple's: a record on another production, bound consistently, refuses)
@pytest.mark.parametrize("which", ("rig-production", "unrelated"))
def test_a_record_on_another_production_is_refused(real_rig, which):
    argv_on, record = x_certified_rig(real_rig)
    assert certify(real_rig, record).completed(PICK)  # control
    other = real_rig.P[:8] if which == "rig-production" else "b26fa7c9"
    refused(
        real_rig,
        certify(real_rig, stage2_record(real_rig, argv_on, "prod", production=other)),
    )


# class: (a) (an explicit --production-ref other than the record's production refuses the certifying run: the record is bound)
def test_an_explicit_production_ref_other_than_the_records_refuses(real_rig):
    argv_on, record = x_certified_rig(real_rig)
    assert certify(real_rig, record).completed(PICK)  # control
    refused(
        real_rig,
        real_rig.run(
            *KV_ON,
            "--entry",
            PICK,
            "--certified-by",
            str(record),
            "--production-ref",
            real_rig.P[:8],
        ),
    )


# class: (c) (fix 3: the report records the sha256 of BOTH files of the record)
def test_the_report_records_both_files_of_the_record(real_rig):
    argv_on, record = x_certified_rig(real_rig)
    got = certify(real_rig, record)
    assert got.completed(PICK), got.text()
    cert = got.report["certified_by"]
    beside = record.parent / "manifest.json"
    assert cert["sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    assert cert["manifest_sha256"] == hashlib.sha256(beside.read_bytes()).hexdigest()
    assert Path(cert["manifest_path"]).resolve() == beside.resolve()


# class: (a) (fix 8: the gate's refusal names the three delta keys, the mode among them)
def test_the_kv_arm_refusal_names_the_delta_keys(real_rig):
    real_rig.registry(kv_text(PICK, 4, mode="m72_v9"))
    got = real_rig.run(*KV_OFF, "--entry", PICK)
    refused(real_rig, got)
    assert "tq4_attention" in (got.refused or "") and "kv_quant_scheme" in (
        got.refused or ""
    )
