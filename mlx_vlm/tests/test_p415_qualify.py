"""P415 (plan v5, C196 / C198 / C199): the R7 256K operational qualification — pure parts (fork-only, dev, CPU).

The tool runs each entry exactly as it ships (the four non-MTP entries as served; the MTP-certified ones MTP-ON), at the
largest prompt the server accepts, with the registry's own request settings; these tests pin the independent expectation
(registry → argv → live), the measured request's validity, the three peak phases, labelled reference deltas and the
per-entry outcome.
"""

import json
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import qualify  # noqa: E402

FABLE = "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed"
UNIFORM = "Qwen3.8-27B-mlx-uniform-4bit"
QWEN36 = "Qwen3.6-27B-Opus-Distill-OptiQ-4bit"
STATIC = "Qwen3.8-27B-static-mixed-4bit"
VENV_BIN = Path("/stack/.venv/bin")

GEN = dict(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    max_tokens=102400,
    enable_thinking=True,
    thinking_budget=81920,
)
FABLE_ENTRY = dict(
    name=FABLE,
    type="vision",
    on_demand=True,
    hf_path=f"caslca/{FABLE}",
    max_kv_cache_size=262144,
    kv_prealloc_tokens=262144,
    cache_session_shrink=True,
    kv_quant_scheme="turboquant",
    kv_bits=0,
    quantized_kv_start=0,
    prefill_step_size=512,
    attention_policy="fused_v1",
    lazy_prompt_embeddings=True,
    draft_kind="mtp",
    draft_model="/drafters/fable",
    mtp_verify_scan="joint_v1",
    generation_defaults=GEN,
    presentation=dict(role="main"),
)
STATIC_ENTRY = dict(
    name=STATIC,
    type="vision",
    on_demand=True,
    hf_path=f"caslca/{STATIC}",
    max_kv_cache_size=262144,
    kv_prealloc_tokens=262144,
    kv_quant_scheme="turboquant",
    kv_bits=4,
    quantized_kv_start=0,
    prefill_step_size=512,
    generation_defaults=GEN,
)


def _cmd(entry):
    """What mlx-serve's ``_build_command`` emits for ``entry`` (hand-written per fixture, never generated)."""
    if entry is FABLE_ENTRY:
        return [
            str(VENV_BIN / "mlx_vlm.server"),
            "--model",
            f"caslca/{FABLE}",
            "--host",
            "127.0.0.1",
            "--port",
            "8091",
            "--max-kv-size",
            "262144",
            "--kv-prealloc-tokens",
            "262144",
            "--cache-session-shrink",
            "on",
            "--attention-policy",
            "fused_v1",
            "--lazy-prompt-embeddings",
            "--mtp-verify-scan",
            "joint_v1",
            "--kv-quant-scheme",
            "turboquant",
            "--prefill-step-size",
            "512",
            "--generation-defaults",
            json.dumps(GEN),
            "--draft-kind",
            "mtp",
            "--draft-model",
            "/drafters/fable",
            "--quantized-kv-start",
            "0",
            "--memory-limit-frac",
            "0.85",
        ]
    return [
        str(VENV_BIN / "mlx_vlm.server"),
        "--model",
        f"caslca/{STATIC}",
        "--host",
        "127.0.0.1",
        "--port",
        "8091",
        "--max-kv-size",
        "262144",
        "--kv-prealloc-tokens",
        "262144",
        "--kv-bits",
        "4",
        "--kv-quant-scheme",
        "turboquant",
        "--prefill-step-size",
        "512",
        "--generation-defaults",
        json.dumps(GEN),
        "--quantized-kv-start",
        "0",
        "--memory-limit-frac",
        "0.85",
    ]


# ------------------------------------------------------------------ T1: the entries and their arms (C199)
def test_t1_the_seven_entries_and_their_arms():
    assert set(qualify.ENTRIES) == {
        FABLE,
        UNIFORM,
        QWEN36,
        STATIC,
        "Qwen3.8-27B-OptiQ-4.5bpw-mixed",
        "Qwen3.8-27B-Fable-Distill-mlx-uniform-4bit",
        "Qwen3.8-27B-Opus-Distill-v2-mlx-uniform-4bit",
    }
    assert qualify.ENTRIES[FABLE] == qualify.ENTRIES[UNIFORM] == "mtp_on"
    assert qualify.ENTRIES[QWEN36] == "mtp_on_gated"
    assert qualify.ENTRIES[STATIC] == "served"
    assert (
        QWEN36 not in qualify.default_entries() and len(qualify.default_entries()) == 6
    )


def test_t1_an_unknown_entry_or_an_ungated_qwen36_refuses(tmp_path):
    assert qualify.entry_problems(["nope"], p369_summary=None)
    assert qualify.entry_problems(
        [QWEN36], p369_summary=None
    )  # its campaign must have passed
    bad = tmp_path / "s.json"
    bad.write_text(json.dumps(dict(status="SMOKE_PASS", picks={QWEN36: {}})))
    assert qualify.entry_problems(
        [QWEN36], p369_summary=bad
    )  # a smoke is not the campaign
    good = tmp_path / "f.json"
    good.write_text(
        json.dumps(
            dict(
                phase="full",
                status="PASS",
                picks={QWEN36: dict(outcome=dict(e1c="PASS"), e2=dict(verdict="PASS"))},
            )
        )
    )
    assert qualify.entry_problems([QWEN36], p369_summary=good) == []
    assert qualify.entry_problems([FABLE, STATIC], p369_summary=None) == []


# ------------------------------------------------------------------ T8 / T14 / T18: registry → full command (Q1 / Q9)
def test_t8_the_expectation_comes_from_the_registry_entry():
    want = qualify.expected_config(FABLE_ENTRY)
    assert want["flags"]["--prefill-step-size"] == "512"
    assert (
        want["flags"]["--memory-limit-frac"] == "0.85"
    )  # mlx-serve's default, absent from the registry
    assert "--kv-bits" not in want["flags"]  # kv_bits 0 = native: no flag
    assert want["drafter"] is True and want["verify_policy"] == "joint_v1"
    assert want["generation_defaults"] == GEN
    assert want["prompt_tokens"] == 262144 - 2048
    assert qualify.expected_config(STATIC_ENTRY)["drafter"] is False


def test_t8_an_unknown_registry_key_refuses():
    with pytest.raises(qualify.UnknownKey, match="new_knob"):
        qualify.expected_config(dict(STATIC_ENTRY, new_knob=1))


@pytest.mark.parametrize("entry", [FABLE_ENTRY, STATIC_ENTRY], ids=["fable", "static"])
def test_t8_the_shipping_command_matches(entry):
    assert (
        qualify.command_problems(qualify.expected_config(entry), _cmd(entry), VENV_BIN)
        == []
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: [
            x for x in c if x != "--lazy-prompt-embeddings"
        ],  # a dropped boolean flag
        lambda c: c[: c.index("--kv-quant-scheme")]
        + c[c.index("--kv-quant-scheme") + 2 :],
        lambda c: [("1024" if x == "512" else x) for x in c],  # prefill step changed
        lambda c: c + ["--surprise", "1"],  # a flag the registry does not ask for
        lambda c: [
            (json.dumps(dict(GEN, temperature=1.0)) if x == json.dumps(GEN) else x)
            for x in c
        ],  # T14: a changed generation default
    ],
    ids=["lazy", "scheme", "step", "extra", "gen-defaults"],
)
def test_t8_t14_any_command_drift_fails(mutate):
    assert qualify.command_problems(
        qualify.expected_config(FABLE_ENTRY), mutate(_cmd(FABLE_ENTRY)), VENV_BIN
    )


def test_t18_a_wrong_backend_prefix_fails():
    cmd = _cmd(STATIC_ENTRY)
    cmd[0] = str(VENV_BIN / "mlx_lm.server")  # valid trailing arguments, wrong backend
    assert any(
        "prefix" in p
        for p in qualify.command_problems(
            qualify.expected_config(STATIC_ENTRY), cmd, VENV_BIN
        )
    )


# ------------------------------------------------------------------ T9 / T15 / T16: the measured request (Q2 Q3 Q10 Q11)
def test_t15_the_prefill_chunk_sequence_is_derived_from_the_step():
    chunks = qualify.chunk_sequence(260096, 512)
    assert chunks[:-1] == [512] * 507 and chunks[-1] == 511 and sum(chunks) == 260095
    assert qualify.chunk_sequence(1030, 512) == [512, 512, 5]


def _record(**changes):
    sha = "ab" * 32
    rec = dict(
        prompt=dict(
            sized=dict(tokens=260096, sha256=sha),
            preprocessed=dict(tokens=260096, sha256=sha),
            prepared=dict(tokens=260096, sha256=sha),
        ),
        cached_tokens=0,
        backend="cached_session",
        resolved=dict(GEN, max_tokens=2048, thinking_budget=1638, seed=7),
        generate_seed=7,
        finish_reason="stop",
        generation_tokens=1500,
        chunks=qualify.chunk_sequence(260096, 512),
        prompt_step_tokens=1,
        boundary_seen=True,
        peaks=dict(prefill=36.5e9, decode=36.9e9, retire=37.1e9),
        rates=dict(prefill_tps=700.0, decode_tps=12.0),
        capacities=[
            dict(leaf="layers.3", cls="PreallocKVCache", capacity=262144),
            dict(leaf="layers.0", cls="ArraysCache", capacity=None),
        ],
        sdpa_forced=640,
        lazy_calls=1,
        shrink_calls=1,
        memory=dict(
            memory_limit_frac=0.85,
            phys=128e9,
            rec_wss=110e9,
            set_memory_limit=[108.8e9],
            set_cache_limit=[20e9],
        ),
        server_defaults=GEN,
        drafter_live_checks=[[]],
    )
    rec.update(changes)
    return rec


def test_t9_a_valid_fable_record_passes():
    assert (
        qualify.request_problems(_record(), qualify.expected_config(FABLE_ENTRY)) == []
    )


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (dict(cached_tokens=5), "cached"),
        (dict(backend="plain"), "backend"),
        (dict(generation_tokens=4096), "tokens"),
        (dict(finish_reason="error"), "finish"),
        (dict(boundary_seen=False), "boundary"),
        (dict(prompt_step_tokens=2), "prompt step"),
        (dict(chunks=[512] * 508), "chunks"),
        (dict(sdpa_forced=0), "sdpa_forced"),
        (dict(lazy_calls=0), "lazy"),
        (dict(shrink_calls=0), "shrink"),
        (dict(drafter_live_checks=[]), "drafter"),
        (dict(drafter_live_checks=[["bad head"]]), "drafter"),
        (dict(generate_seed=None), "seed"),
        (dict(server_defaults=dict(GEN, top_k=40)), "defaults"),
        (dict(peaks=dict(prefill=float("nan"), decode=1.0, retire=1.0)), "peak"),
    ],
)
def test_t9_t15_each_invalid_fable_record_fails(change, why):
    problems = qualify.request_problems(
        _record(**change), qualify.expected_config(FABLE_ENTRY)
    )
    assert any(why in p for p in problems), (why, problems)


@pytest.mark.parametrize(
    ("field", "value"),
    [("sized", "cd"), ("preprocessed", "cd"), ("prepared", "cd")],
)
def test_t16_every_prompt_digest_must_agree(field, value):
    rec = _record()
    rec["prompt"][field] = dict(tokens=260096, sha256=value * 32)
    assert any(
        "prompt" in p
        for p in qualify.request_problems(rec, qualify.expected_config(FABLE_ENTRY))
    )


def test_c198_the_request_resolves_the_registry_defaults_with_the_servers_own_clamp():
    """C198 (1): no override except the seed; the server clamps max_tokens to the remaining 2,048 and thinking to 0.8."""
    want = qualify.expected_config(FABLE_ENTRY)
    for key, bad in (
        ("max_tokens", 32),
        ("thinking_budget", 81920),
        ("temperature", 1.0),
        ("seed", 8),
    ):
        rec = _record()
        rec["resolved"] = dict(rec["resolved"], **{key: bad})
        assert any(key in p for p in qualify.request_problems(rec, want)), key


def test_t15_a_served_auto_entry_must_not_take_the_fused_or_lazy_paths():
    want = qualify.expected_config(STATIC_ENTRY)
    ok = _record(sdpa_forced=0, lazy_calls=0, shrink_calls=0, drafter_live_checks=None)
    ok["peaks"] = dict(prefill=30e9, decode=30.2e9, retire=None)
    assert qualify.request_problems(ok, want) == []
    for change in (dict(sdpa_forced=3), dict(lazy_calls=1), dict(shrink_calls=1)):
        bad = dict(ok, **change)
        assert qualify.request_problems(bad, want), change


def test_t15_preallocated_kv_leaves_must_hold_the_registry_floor():
    rec = _record(
        capacities=[dict(leaf="layers.3", cls="PreallocKVCache", capacity=4096)]
    )
    assert any(
        "capacity" in p
        for p in qualify.request_problems(rec, qualify.expected_config(FABLE_ENTRY))
    )
    rec = _record(capacities=[])
    assert any(
        "capacity" in p
        for p in qualify.request_problems(rec, qualify.expected_config(FABLE_ENTRY))
    )


def test_t15_the_memory_limit_is_mlx_serves_fraction_of_physical_memory():
    rec = _record()
    rec["memory"] = dict(rec["memory"], set_memory_limit=[64e9])
    assert any(
        "memory limit" in p
        for p in qualify.request_problems(rec, qualify.expected_config(FABLE_ENTRY))
    )
    rec["memory"] = dict(rec["memory"], set_memory_limit=[], set_cache_limit=[])
    assert qualify.request_problems(rec, qualify.expected_config(FABLE_ENTRY))


def test_q2_capacity_is_the_max_of_the_three_phases():
    assert qualify.capacity_gb(dict(prefill=36e9, decode=37e9, retire=38e9)) == 38.0
    assert qualify.capacity_gb(dict(prefill=36e9, decode=37e9, retire=None)) == 37.0


# ------------------------------------------------------------------ T6 / T13: references (Q7, C199 (a))
FP = dict(
    prompt_tokens=260096,
    kv_codec="native16",
    p405_window=False,
    mtp="on",
    attention_policy="fused_v1",
    prealloc=262144,
    lazy_embeddings=True,
    prefill_step=512,
    mlx="0.32.3",
    method="mx.get_peak_memory",
    snapshot="r1",
    harness="p415",
    context_cap=262144,
    memory_limits="0.85",
    shrink="on",
    output_tokens=2048,
    backend="mlx_vlm.server",
    prompt_identity="sha",
)


def test_t13_a_delta_is_always_labelled_by_every_differing_field():
    ref = dict(peak_gb=37.97, fingerprint=dict(FP, mlx="0.32.2", prompt_tokens=261449))
    got = qualify.reference_comparison(ref, 38.5, FP)
    assert got["delta_gb"] == pytest.approx(0.53)
    assert sorted(got["differs"]) == ["mlx", "prompt_tokens"]
    assert got["comparable"] is False
    same = qualify.reference_comparison(
        dict(peak_gb=38.0, fingerprint=dict(FP)), 38.5, FP
    )
    assert same["comparable"] is True and same["differs"] == []


def test_t13_an_absent_reference_field_is_a_difference():
    fp = dict(FP)
    del fp["harness"]
    got = qualify.reference_comparison(dict(peak_gb=37.0, fingerprint=fp), 38.0, FP)
    assert "harness" in got["differs"] and got["comparable"] is False


def test_t6_the_reference_table_loads_and_refuses_bad_rows(tmp_path):
    good = {
        FABLE: dict(
            peak_gb=37.97, source="docs/campaign-results.md:294", fingerprint=FP
        ),
        "Qwen3.8-27B-Opus-Distill-v2-mlx-uniform-4bit": None,  # none recorded
    }
    p = tmp_path / "refs.json"
    p.write_text(json.dumps(good))
    assert qualify.load_references(p)[FABLE]["peak_gb"] == 37.97
    for bad in (
        {"not-an-entry": None},
        {FABLE: dict(peak_gb=37.97, fingerprint=FP)},  # no source
    ):
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            qualify.load_references(p)


# ------------------------------------------------------------------ T5 / T10 / T7 / T11: outcomes and the schedule
def test_t5_idle_before_every_entry_after_the_first():
    assert qualify.idle_before(["a", "b", "c"]) == [False, True, True]


@pytest.mark.parametrize(
    ("pre", "post", "error", "problems", "status"),
    [
        (dict(ok=True), dict(ok=True), None, [], "COMPLETED"),
        (dict(ok=True), dict(ok=True), None, ["chunks"], "FAILED"),
        (dict(ok=True), dict(ok=True), "timeout", [], "FAILED"),
        (dict(ok=True), dict(ok=False, why="power"), None, [], "INVALID"),
        (dict(ok=False, why="process"), None, None, [], "INVALID"),
    ],
)
def test_t10_t11_the_entry_outcome(pre, post, error, problems, status):
    assert (
        qualify.entry_status(pre=pre, post=post, error=error, problems=problems)[
            "status"
        ]
        == status
    )


def test_t7_the_run_exits_nonzero_unless_every_entry_completed():
    assert qualify.exit_code({"a": dict(status="COMPLETED")}) == 0
    assert (
        qualify.exit_code({"a": dict(status="COMPLETED"), "b": dict(status="NOT_RUN")})
        == 1
    )
    assert qualify.exit_code({"a": dict(status="INVALID")}) == 1


# ------------------------------------------------------------------ T7 / T11: the driver's lifecycle (orchestrator mocked)
def _drive(tmp_path, monkeypatch, run_job, names=(FABLE, STATIC)):
    import argparse

    from p369_phase2 import orchestrate

    registry = "models: []\n"
    blocks = {FABLE: FABLE_ENTRY, STATIC: STATIC_ENTRY}
    for name, value in dict(
        acquire_lock=lambda lock: 3,
        release_lock=lambda fd: None,
        lock_path=lambda env: tmp_path / ".lock",
        preflight=lambda args, out: dict(problems=[]),
        read_registry=lambda p: registry,
        sha256_file=lambda p: "s",
        fork_versions=lambda f: {"mlx": "0.32.3"},
        overlay_registry=lambda text, name, fixture: (text, True),
        weights_record=lambda m: dict(revision="r", files={}),
        run_job=run_job,
        worker_env=lambda env: env,
    ).items():
        monkeypatch.setattr(orchestrate, name, value)
    monkeypatch.setattr(orchestrate, "Monitor", _NullMonitor)
    from p369_phase2 import registry_states

    monkeypatch.setattr(registry_states, "fixture_text", lambda d: "fx")
    monkeypatch.setattr(registry_states, "classify", lambda text, fx: set())
    monkeypatch.setattr(qualify, "load_references", lambda p: {})
    monkeypatch.setattr(qualify, "interference", lambda o: dict(ok=True, why=None))
    monkeypatch.setattr(qualify, "binding_problems", lambda *a: [])
    monkeypatch.setattr(qualify, "_git_head", lambda p: "h")
    monkeypatch.setattr(
        qualify, "full_command", lambda stack, path, name: (_cmd(blocks[name]), "m")
    )
    monkeypatch.setattr(
        "yaml.safe_load", lambda text: dict(models=list(blocks.values()))
    )
    monkeypatch.setattr("time.sleep", lambda s: None)
    args = argparse.Namespace(
        stack_root=str(tmp_path / "stack"),
        fork_root=str(tmp_path / "fork"),
        out=str(tmp_path / "run"),
        entry=list(names),
        p369_summary=None,
        idle_s=600.0,
    )
    (tmp_path / "stack" / ".venv" / "bin").mkdir(parents=True)
    code = qualify.drive(args)
    return code, json.loads((tmp_path / "run" / "report.json").read_text())


class _NullMonitor:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _good(job, *a, **k):
    rec = (
        _record()
        if job["name"].startswith(FABLE)
        else _record(
            sdpa_forced=0,
            lazy_calls=0,
            shrink_calls=0,
            drafter_live_checks=None,
            peaks=dict(prefill=30e9, decode=30.2e9, retire=None),
        )
    )
    return dict(requests=[dict(qualify=rec)])


def test_t7_every_entry_completed_exits_zero(tmp_path, monkeypatch):
    code, report = _drive(tmp_path, monkeypatch, _good)
    assert code == 0, report
    assert {n: e["status"] for n, e in report["entries"].items()} == {
        FABLE: "COMPLETED",
        STATIC: "COMPLETED",
    }
    assert report["entries"][FABLE]["capacity_gb"] == pytest.approx(37.1)


def test_t11_a_failing_entry_is_recorded_and_the_run_continues(tmp_path, monkeypatch):
    def run_job(job, *a, **k):
        if job["name"].startswith(FABLE):
            raise RuntimeError("job failed (rc 1)")
        return _good(job)

    code, report = _drive(tmp_path, monkeypatch, run_job)
    assert code == 1
    assert report["entries"][FABLE]["status"] == "FAILED"
    assert report["entries"][STATIC]["status"] == "COMPLETED"


def test_t11_an_interruption_marks_the_rest_not_run(tmp_path, monkeypatch):
    def run_job(job, *a, **k):
        raise SystemExit(143)  # what the SIGTERM handler raises

    with pytest.raises(SystemExit):
        _drive(tmp_path, monkeypatch, run_job)
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert {e["status"] for e in report["entries"].values()} == {"NOT_RUN"}
