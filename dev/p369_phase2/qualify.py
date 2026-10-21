"""P415 (plan v5, C196 / C198 / C199): the R7 256K operational qualification — pure parts (fork-only, dev).

Per entry, the configuration it SHIPS (C199: the MTP-certified entries MTP-ON, the others as served), the largest prompt the
server accepts (context cap − ``MIN_OUTPUT_TOKENS``) and the registry's own request settings (C198: no override but the
seed). The expectation is derived from the registry entry independently of mlx-serve's argv builder (plan v2 Q1): a
reviewed key → flag table with mlx-serve's own defaults; an unknown registry key refuses.
"""

import json
import math
import os
from pathlib import Path

ENTRIES = {  # C199 (c): the arm each entry is qualified in
    "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed": "mtp_on",
    "Qwen3.8-27B-mlx-uniform-4bit": "mtp_on",
    "Qwen3.6-27B-Opus-Distill-OptiQ-4bit": "mtp_on_gated",  # only after its P369 campaign passed
    "Qwen3.8-27B-static-mixed-4bit": "served",
    "Qwen3.8-27B-OptiQ-4.5bpw-mixed": "served",
    "Qwen3.8-27B-Fable-Distill-mlx-uniform-4bit": "served",
    "Qwen3.8-27B-Opus-Distill-v2-mlx-uniform-4bit": "served",
}
MIN_OUTPUT_TOKENS = (
    2048  # mlx-vlm server: a request leaving less is refused (fork 02e8046d)
)
THINKING_CLAMP = 0.8  # mlx-vlm server: THINKING_BUDGET_CLAMP_RATIO
SEED = 7
VISION_SERVER = "mlx_vlm.server"  # mlx-serve _MLX_VLM_SERVER: <stack venv>/bin/mlx_vlm.server for type: vision
NON_SERVING = ("name", "on_demand", "presentation", "display_name", "description")
# registry key -> (mlx-serve default, flag, how the value is spelled); mlx-serve config.ModelConfig / _build_command
SERVING = {
    "type": ("text", None, None),
    "hf_path": (None, "--model", "str"),
    "context_length": (0, "--max-tokens", "positive"),
    "max_kv_cache_size": (0, "--max-kv-size", "positive"),
    "kv_prealloc_tokens": (0, "--kv-prealloc-tokens", "positive"),
    "cache_session_shrink": (None, "--cache-session-shrink", "onoff"),
    "attention_policy": ("", "--attention-policy", "not_auto"),
    "lazy_prompt_embeddings": (None, "--lazy-prompt-embeddings", "flag"),
    "mtp_verify_scan": ("", "--mtp-verify-scan", "not_per_query"),
    "kv_bits": (0, "--kv-bits", "positive"),
    "kv_quant_scheme": ("", "--kv-quant-scheme", "nonempty"),
    "kv_quant_mode": ("", "--kv-quant-mode", "nonempty"),
    "prefill_step_size": (2048, "--prefill-step-size", "positive"),
    "generation_defaults": ({}, "--generation-defaults", "json"),
    "draft_kind": ("", "--draft-kind", "nonempty"),
    "draft_model": ("", "--draft-model", "nonempty"),
    "quantized_kv_start": (0, "--quantized-kv-start", "always"),
    "cache_limit_gb": (0.0, "--cache-limit-gb", "positive"),
    "memory_limit_frac": (0.85, "--memory-limit-frac", "positive"),
}
FIXED = {"--host": "127.0.0.1", "--port": None}  # present, value not the registry's
VALUELESS = ("--lazy-prompt-embeddings",)
KV_CLASSES = ("PreallocKVCache", "PreallocQuantizedKVCache", "TurboQuantKVCache")


class UnknownKey(ValueError):
    pass


def default_entries():
    return [e for e, arm in ENTRIES.items() if arm != "mtp_on_gated"]


def entry_problems(names, *, p369_summary):
    """Every name is one of the seven; ``Qwen3.6-27B-Opus-Distill-OptiQ-4bit`` only with a P369 FULL-run PASS for it."""
    problems = [f"unknown entry {n!r}" for n in names if n not in ENTRIES]
    for name in names:
        if ENTRIES.get(name) != "mtp_on_gated":
            continue
        summary = json.loads(Path(p369_summary).read_text()) if p369_summary else None
        pick = ((summary or {}).get("picks") or {}).get(name) or {}
        passed = (
            summary is not None
            and summary.get("phase") == "full"
            and summary.get("status") == "PASS"
            and (pick.get("outcome") or {}).get("e1c") == "PASS"
            and (pick.get("e2") or {}).get("verdict") == "PASS"
        )
        if not passed:
            problems.append(f"{name}: MTP-ON only after its P369 full-run PASS (C199)")
    return problems


def _spelled(kind, value):
    """The flag's value as ``_build_command`` writes it; None = the flag is absent."""
    if kind == "str":
        return str(value)
    if kind == "positive":
        return str(value) if value and value > 0 else None
    if kind == "onoff":
        return None if value is None else ("on" if value else "off")
    if kind == "not_auto":
        return None if value in ("", "auto") else str(value)
    if kind == "flag":
        return "" if value is True else None
    if kind == "not_per_query":
        return None if value in ("", "per_query") else str(value)
    if kind == "nonempty":
        return str(value) if value else None
    if kind == "json":
        return json.dumps(value) if value else None
    if kind == "always":
        return str(value)
    raise ValueError(kind)


def expected_config(entry):
    """The served configuration the registry entry asks for (plan v2 Q1): flags, drafter, verify policy, KV, request
    defaults and the prompt size (C198: cap − MIN_OUTPUT_TOKENS)."""
    unknown = sorted(k for k in entry if k not in SERVING and k not in NON_SERVING)
    if unknown:
        raise UnknownKey(
            f"{entry.get('name')}: registry keys outside the reviewed table: {unknown}"
        )
    values = {k: entry.get(k, default) for k, (default, _, _) in SERVING.items()}
    flags = {}
    for key, (_, flag, kind) in SERVING.items():
        if flag is None:
            continue
        if kind != "str" and values[key] is None and kind not in ("onoff", "flag"):
            continue
        spelled = _spelled(kind, values[key])
        if spelled is not None:
            flags[flag] = spelled
    drafter = bool(values["draft_kind"] == "mtp" and values["draft_model"])
    cap = int(values["max_kv_cache_size"])
    return dict(
        name=entry.get("name"),
        type=values["type"],
        flags=flags,
        drafter=drafter,
        verify_policy=(values["mtp_verify_scan"] or "per_query") if drafter else None,
        kv_bits=int(values["kv_bits"]) or None,
        kv_quant_scheme=values["kv_quant_scheme"] or None,
        prefill_step=int(values["prefill_step_size"]),
        prealloc=int(values["kv_prealloc_tokens"]),
        cap=cap,
        attention_policy=values["attention_policy"] or "auto",
        lazy_embeddings=values["lazy_prompt_embeddings"] is True,
        shrink=values["cache_session_shrink"] is True,
        memory_limit_frac=float(values["memory_limit_frac"]),
        cache_limit_gb=float(values["cache_limit_gb"]),
        generation_defaults=dict(values["generation_defaults"]),
        prompt_tokens=cap - MIN_OUTPUT_TOKENS,
    )


def _parse_flags(args):
    flags, i = {}, 0
    while i < len(args):
        flag = args[i]
        if not flag.startswith("--"):
            raise ValueError(f"positional argument {flag!r}")
        if flag in flags:
            raise ValueError(f"duplicate flag {flag}")
        if flag in VALUELESS:
            flags[flag] = ""
            i += 1
        else:
            if i + 1 >= len(args):
                raise ValueError(f"flag {flag} without a value")
            flags[flag] = args[i + 1]
            i += 2
    return flags


def command_problems(expected, cmd, venv_bin):
    """The FULL ``_build_command`` command (plan v3 Q9): the vision server prefix of the stack venv, then exactly the
    expected flags (two-way: missing, changed and unexpected flags all fail; ``--host`` / ``--port`` fixed).
    """
    problems = []
    if expected["type"] != "vision":
        problems.append(f"type {expected['type']!r}: only vision entries are qualified")
    want_prefix = str(Path(venv_bin) / VISION_SERVER)
    if not cmd or cmd[0] != want_prefix:
        problems.append(f"backend prefix {cmd[:1]} != {want_prefix}")
    try:
        got = _parse_flags(cmd[1:])
    except ValueError as exc:
        return problems + [f"argv: {exc}"]
    for flag, want in expected["flags"].items():
        if flag not in got:
            problems.append(f"argv lacks {flag} (registry wants {want!r})")
        elif flag == "--generation-defaults":
            if json.loads(got[flag]) != json.loads(want):
                problems.append(f"argv {flag} {got[flag]} != registry {want}")
        elif got[flag] != want:
            problems.append(f"argv {flag} {got[flag]!r} != registry {want!r}")
    for flag in got:
        if flag not in expected["flags"] and flag not in FIXED:
            problems.append(f"argv carries {flag} the registry does not ask for")
    for flag, value in FIXED.items():
        if flag not in got or (value is not None and got[flag] != value):
            problems.append(f"argv {flag} {got.get(flag)!r}")
    return problems


def chunk_sequence(prompt_tokens, step):
    """The chunked prefill's forwards for a fresh prompt (``generate/ar.py``: chunks of ``min(step, left − 1)`` while
    more than one token is left; the last token goes through the prompt step)."""
    out, left = [], prompt_tokens
    while left > 1:
        n = min(step, left - 1)
        out.append(n)
        left -= n
    return out


def _finite_positive(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
        and x > 0
    )


def request_problems(rec, expected):
    """The measured request is the one claimed and ran the configuration claimed (plan v2–v4 Q2 Q3 Q10 Q11, C198)."""
    problems = []
    target = expected["prompt_tokens"]
    prompts = rec.get("prompt") or {}
    shas = {
        k: (prompts.get(k) or {}).get("sha256")
        for k in ("sized", "preprocessed", "prepared")
    }
    for k in ("sized", "preprocessed", "prepared"):
        if (prompts.get(k) or {}).get("tokens") != target:
            problems.append(
                f"prompt {k} tokens {(prompts.get(k) or {}).get('tokens')} != {target}"
            )
    if None in shas.values() or len(set(shas.values())) != 1:
        problems.append(f"prompt digests differ {shas}")
    if rec.get("cached_tokens") != 0:
        problems.append(f"cached tokens {rec.get('cached_tokens')}")
    if rec.get("backend") != "cached_session":
        problems.append(f"backend {rec.get('backend')!r}")
    # C198 (1): the registry's own settings, clamped by the server to the remaining context
    budget = min(
        int(expected["generation_defaults"].get("max_tokens", 0) or 0)
        or MIN_OUTPUT_TOKENS,
        expected["cap"] - target,
    )
    resolved = rec.get("resolved") or {}
    want = dict(expected["generation_defaults"], max_tokens=budget, seed=SEED)
    if "thinking_budget" in want:
        want["thinking_budget"] = min(
            want["thinking_budget"], int(budget * THINKING_CLAMP)
        )
    for key, value in want.items():
        if resolved.get(key) != value:
            problems.append(f"resolved {key} {resolved.get(key)!r} != {value!r}")
    if rec.get("generate_seed") != SEED:
        problems.append(f"generate_step seed {rec.get('generate_seed')!r} != {SEED}")
    if rec.get("server_defaults") != expected["generation_defaults"]:
        problems.append(f"server defaults {rec.get('server_defaults')} != registry")
    if rec.get("finish_reason") not in ("stop", "length"):
        problems.append(f"finish {rec.get('finish_reason')!r}")
    tokens = rec.get("generation_tokens")
    if not (isinstance(tokens, int) and 0 < tokens <= budget):
        problems.append(f"generated tokens {tokens!r} not in 1..{budget}")
    if rec.get("boundary_seen") is not True:
        problems.append("no prefill / decode boundary observed (Q11)")
    if rec.get("prompt_step_tokens") != 1:
        problems.append(f"prompt step {rec.get('prompt_step_tokens')} tokens, not 1")
    if rec.get("chunks") != chunk_sequence(target, expected["prefill_step"]):
        got = rec.get("chunks") or []
        problems.append(
            f"prefill chunks {len(got)} / {sum(got)} tokens != the step's sequence"
        )
    peaks = rec.get("peaks") or {}
    for phase in ("prefill", "decode"):
        if not _finite_positive(peaks.get(phase)):
            problems.append(f"{phase} peak {peaks.get(phase)!r}")
    if expected["shrink"] != _finite_positive(peaks.get("retire")):
        problems.append(
            f"retire peak {peaks.get('retire')!r} (shrink {expected['shrink']})"
        )
    for name in ("prefill_tps", "decode_tps"):
        if not _finite_positive((rec.get("rates") or {}).get(name)):
            problems.append(f"{name} {(rec.get('rates') or {}).get(name)!r}")
    kv = [c for c in rec.get("capacities") or [] if c.get("cls") in KV_CLASSES]
    if not kv:
        problems.append("no preallocated KV leaf capacity observed")
    for c in kv:
        if not (
            isinstance(c.get("capacity"), int) and c["capacity"] >= expected["prealloc"]
        ):
            problems.append(
                f"{c.get('leaf')} capacity {c.get('capacity')} < {expected['prealloc']}"
            )
    fused = expected["attention_policy"] == "fused_v1"
    forced = rec.get("sdpa_forced") or 0
    if fused != (forced > 0):
        problems.append(
            f"sdpa_forced {forced} (attention {expected['attention_policy']})"
        )
    if expected["lazy_embeddings"] != ((rec.get("lazy_calls") or 0) > 0):
        problems.append(
            f"lazy embedding calls {rec.get('lazy_calls')} ({expected['lazy_embeddings']})"
        )
    if expected["shrink"] != ((rec.get("shrink_calls") or 0) > 0):
        problems.append(
            f"shrink calls {rec.get('shrink_calls')} ({expected['shrink']})"
        )
    memory = rec.get("memory") or {}
    limits = [
        v
        for v in (
            expected["memory_limit_frac"] * (memory.get("phys") or 0),
            memory.get("rec_wss") or 0,
        )
        if v > 0
    ]
    if memory.get("memory_limit_frac") != expected["memory_limit_frac"]:
        problems.append(f"memory limit fraction {memory.get('memory_limit_frac')}")
    want_limit = int(min(limits)) if limits else None
    if want_limit is None or memory.get("set_memory_limit") != [want_limit]:
        problems.append(
            f"memory limit {memory.get('set_memory_limit')} != {want_limit}"
        )
    caches = memory.get("set_cache_limit") or []
    if expected["cache_limit_gb"] > 0:
        if caches != [int(expected["cache_limit_gb"] * 2**30)]:
            problems.append(f"cache limit {caches}")
    elif len(caches) != 1 or not _finite_positive(caches[0]):
        problems.append(f"cache limit {caches} (auto-derived, one positive value)")
    checks = rec.get("drafter_live_checks")
    if expected["drafter"]:
        if not checks or any(checks):
            problems.append(f"drafter live checks {checks}")
    elif checks:
        problems.append(f"drafter checks without a drafter {checks}")
    return problems


def capacity_gb(peaks):
    """Plan v4 Q2: capacity = the max of the prefill, decode and retire (shrink) phases, GB (1e9)."""
    return max(v for v in peaks.values() if v is not None) / 1e9


FINGERPRINT = (
    "prompt_tokens",
    "kv_codec",
    "p405_window",
    "mtp",
    "attention_policy",
    "prealloc",
    "lazy_embeddings",
    "prefill_step",
    "mlx",
    "method",
    "snapshot",
    "harness",
    "context_cap",
    "memory_limits",
    "shrink",
    "output_tokens",
    "backend",
    "prompt_identity",
)


def reference_comparison(ref, peak_gb, fingerprint):
    """C199 (a): the delta vs a historical peak, labelled by EVERY fingerprint field that differs or is absent."""
    theirs = ref.get("fingerprint") or {}
    differs = [
        f for f in FINGERPRINT if f not in theirs or theirs[f] != fingerprint.get(f)
    ]
    return dict(
        reference_gb=ref["peak_gb"],
        delta_gb=round(peak_gb - ref["peak_gb"], 4),
        differs=differs,
        comparable=not differs,
        source=ref.get("source"),
    )


def load_references(path):
    table = json.loads(Path(path).read_text())
    for name, ref in table.items():
        if name not in ENTRIES:
            raise ValueError(f"reference for an unknown entry {name!r}")
        if ref is not None and not (
            ref.get("source") and _finite_positive(ref.get("peak_gb"))
        ):
            raise ValueError(
                f"{name}: a reference needs a source and a positive peak_gb"
            )
    return table


def idle_before(names):
    return [i > 0 for i in range(len(names))]


def entry_status(*, pre, post, error, problems):
    """C198 (3): interference / power before or after an entry ⇒ INVALID; an error or a problem ⇒ FAILED."""
    if not pre.get("ok"):
        return dict(status="INVALID", reason=f"before: {pre.get('why')}")
    if post is not None and not post.get("ok"):
        return dict(status="INVALID", reason=f"after: {post.get('why')}")
    if error:
        return dict(status="FAILED", reason=str(error))
    if problems:
        return dict(status="FAILED", reason="; ".join(problems[:10]))
    return dict(status="COMPLETED", reason=None)


def exit_code(entries):
    return (
        0
        if entries and all(e.get("status") == "COMPLETED" for e in entries.values())
        else 1
    )


# ------------------------------------------------------------------ the driver (plan v5 §2, Q5 / Q6; C198 (3))
IDLE_S = 600.0  # AGENTS.md: >= 10 min idle between arms
BOOT_S, REQUEST_S, JOB_S = 600, 2400, 3600  # plan v2 Q8
REFERENCES = Path(__file__).resolve().parents[1] / "p415_reference_peaks.json"


def interference(orchestrate):
    """C198 (3): power and the process scan, once before and once after each entry."""
    import psutil

    power = orchestrate.power_state()
    if not (power["ac_ok"] and power["battery_ok"]):
        return dict(ok=False, why=f"power {power}", power=power)
    me = {os.getpid(), *(c.pid for c in psutil.Process().children(recursive=True))}
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] in me:
            continue
        why = orchestrate.refused_process(p.info["cmdline"] or [])
        if why:
            return dict(ok=False, why=f"process {p.info['pid']} ({why})", power=power)
    return dict(ok=True, why=None, power=power)


def full_command(stack, registry_path, name):
    """mlx-serve's own ``_build_command`` for ``name`` on ``registry_path`` — WITH its executable prefix (plan v3 Q9)."""
    import subprocess

    code = (
        "import json, os\n"
        f"os.environ['MLX_SERVE_CONFIG'] = {str(registry_path)!r}\n"
        "from mlx_serve import config, process_manager\n"
        f"print(json.dumps(process_manager._build_command(config._load()[0][{name!r}])))\n"
        "print(process_manager.__file__)\n"
    )
    out = subprocess.run(
        [str(Path(stack) / ".venv" / "bin" / "python"), "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=stack,
    )
    lines = out.stdout.strip().splitlines()
    return json.loads(lines[-2]), lines[-1]


def binding_problems(stack, module_file, registry_sha, sha256_file):
    """Plan v2 Q6: the stack HEAD is clean for the registry, the registry unchanged, and the argv builder is the
    ``src/mlx-serve`` submodule at the stack's gitlink, clean."""
    import subprocess

    def git(*a, cwd=stack):
        return subprocess.run(
            ["git", "-C", str(cwd), *a], capture_output=True, text=True, check=True
        ).stdout.strip()

    problems = []
    sub = Path(stack) / "src" / "mlx-serve"
    gitlink = git("ls-tree", "HEAD", "src/mlx-serve").split()[2]
    if git("rev-parse", "HEAD", cwd=sub) != gitlink:
        problems.append(f"mlx-serve submodule not at the stack gitlink {gitlink}")
    if git("status", "--porcelain", cwd=sub):
        problems.append("mlx-serve submodule has local changes")
    if not Path(module_file).resolve().is_relative_to(sub.resolve()):
        problems.append(f"mlx_serve imported from {module_file}, not {sub}")
    if git("status", "--porcelain", "--", "main_models.yaml"):
        problems.append("main_models.yaml has uncommitted changes")
    if sha256_file(Path(stack) / "main_models.yaml") != registry_sha:
        problems.append("main_models.yaml changed during the run")
    return problems


def _git_head(path):
    import subprocess

    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def drive(args):
    import signal
    import sys
    import time

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from p369_phase2 import kvmanifest, orchestrate, registry_states

    stack, fork, out = (
        Path(p).resolve() for p in (args.stack_root, args.fork_root, args.out)
    )
    names = list(args.entry or default_entries())
    problems = entry_problems(names, p369_summary=args.p369_summary)
    if problems:
        raise SystemExit(f"refused: {problems}")
    out.mkdir(parents=True, exist_ok=False)
    lock = orchestrate.acquire_lock(orchestrate.lock_path(dict(os.environ)))
    pre = orchestrate.preflight(args, out)
    if pre["problems"]:
        raise SystemExit(f"refused: {pre['problems']}")
    registry = orchestrate.read_registry(stack / "main_models.yaml")
    registry_sha = orchestrate.sha256_file(stack / "main_models.yaml")
    fixture = registry_states.fixture_text(fork / "mlx_vlm" / "tests" / "fixtures")
    if registry_states.classify(registry, fixture) is None:
        raise SystemExit(
            "refused: the live registry is not one of the 16 accepted byte states"
        )
    references = load_references(REFERENCES)
    report = dict(
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
        fork_head=_git_head(fork),
        stack_head=_git_head(stack),
        production=orchestrate.PRODUCTION,
        registry_sha256=registry_sha,
        versions=orchestrate.fork_versions(fork),
        entries={n: dict(status="PENDING", arm=ENTRIES[n]) for n in names},
    )
    _atomic(out / "report.json", report)
    monitor = orchestrate.Monitor(out / "monitor.log", estimate_s=len(names) * 1200)
    stop = dict(signal=None)

    def on_signal(signum, _frame):
        stop["signal"] = signum
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    monitor.__enter__()
    try:
        for i, name in enumerate(names):
            if i:
                time.sleep(args.idle_s)
            state = report["entries"][name]
            state.update(status="RUNNING", started=time.strftime("%Y-%m-%dT%H:%M:%S"))
            _atomic(out / "report.json", report)
            before = interference(orchestrate)
            error, problems, after, rec = None, [], None, None
            if before["ok"]:
                try:
                    if ENTRIES[name] == "served":
                        path, text = stack / "main_models.yaml", registry
                    else:  # C199: the shipped MTP-ON form, the 16-state generator's exact block
                        text, _ = orchestrate.overlay_registry(registry, name, fixture)
                        path = out / f"overlay-{name}.yaml"
                        path.write_text(text)
                    entry = {e["name"]: e for e in yaml.safe_load(text)["models"]}[name]
                    expected = expected_config(entry)
                    cmd, module_file = full_command(stack, path, name)
                    problems = binding_problems(
                        stack, module_file, registry_sha, orchestrate.sha256_file
                    ) + command_problems(expected, cmd, stack / ".venv" / "bin")
                    if not problems:
                        argv = cmd[1:]
                        model = argv[argv.index("--model") + 1]
                        revisions = {"--model": orchestrate.weights_record(model)}
                        if expected["drafter"]:
                            revisions["--draft-model"] = orchestrate.weights_record(
                                argv[argv.index("--draft-model") + 1]
                            )
                        job = dict(
                            name=f"{name}/qualify",
                            mode="qualify",
                            argv=argv,
                            expect=dict(
                                drafter=expected["drafter"],
                                verify_policy=expected["verify_policy"],
                                revisions={
                                    k: dict(revision=v["revision"], files=v["files"])
                                    for k, v in revisions.items()
                                },
                                kv=kvmanifest.kv_from_argv(argv),
                                drafter_shape=(
                                    orchestrate.PICKS[name]["drafter"]
                                    if expected["drafter"]
                                    else None
                                ),
                            ),
                            source_root=str(fork),
                            dev_dir=str(fork / "dev"),
                            out_dir=str(out / name),
                            requests=[
                                dict(
                                    name="max-context",
                                    model=name,
                                    target=expected["prompt_tokens"],
                                    prefix=orchestrate.SIZE_PREFIX,
                                    suffix=orchestrate.SIZE_SUFFIX,
                                    seed=SEED,
                                    generation_defaults=expected["generation_defaults"],
                                    timeout_s=REQUEST_S,
                                )
                            ],
                            timeout_s=JOB_S,
                        )
                        result = orchestrate.run_job(
                            job,
                            fork,
                            orchestrate.worker_env(dict(os.environ)),
                            monitor,
                            deadline=math.inf,
                        )
                        rec = result["requests"][0]["qualify"]
                        problems = request_problems(rec, expected)
                except SystemExit:
                    raise
                except (
                    Exception
                ) as exc:  # recorded per entry; the run continues (plan v2 Q5)
                    error = f"{type(exc).__name__}: {exc}"
                after = interference(orchestrate)
            state.update(
                entry_status(pre=before, post=after, error=error, problems=problems),
                problems=problems[:50],
                before=before,
                after=after,
                record=rec,
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            if rec and state["status"] == "COMPLETED":
                peak = capacity_gb(rec["peaks"])
                state["capacity_gb"] = peak
                state["vs_guidance_gb"] = round(peak - 48.0, 3)
                ref = references.get(name)
                state["reference"] = (
                    None
                    if ref is None
                    else reference_comparison(ref, peak, fingerprint_of(rec, expected))
                )
            _atomic(out / "report.json", report)
    finally:
        for state in report["entries"].values():
            if state["status"] in ("PENDING", "RUNNING"):
                state["status"] = "NOT_RUN"
                state["reason"] = (
                    f"interrupted (signal {stop['signal']})"
                    if stop["signal"]
                    else "not reached"
                )
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _atomic(out / "report.json", report)
        monitor.__exit__(None, None, None)
        orchestrate.release_lock(lock)
    return exit_code(report["entries"])


def fingerprint_of(rec, expected):
    """This run's comparability fingerprint (plan v4 Q7)."""
    return dict(
        prompt_tokens=expected["prompt_tokens"],
        kv_codec=(
            "native16"
            if not expected["kv_bits"]
            else f"{expected['kv_quant_scheme']}{expected['kv_bits']}"
        ),
        p405_window=False,
        mtp="on" if expected["drafter"] else "off",
        attention_policy=expected["attention_policy"],
        prealloc=expected["prealloc"],
        lazy_embeddings=expected["lazy_embeddings"],
        prefill_step=expected["prefill_step"],
        mlx=(rec.get("versions") or {}).get("mlx"),
        method="mx.get_peak_memory phases (P415)",
        snapshot=rec.get("snapshot"),
        harness="p415",
        context_cap=expected["cap"],
        memory_limits=expected["memory_limit_frac"],
        shrink="on" if expected["shrink"] else "off",
        output_tokens=rec.get("generation_tokens"),
        backend=VISION_SERVER,
        prompt_identity=((rec.get("prompt") or {}).get("sized") or {}).get("sha256"),
    )


def main(argv=None):
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack-root", required=True)
    ap.add_argument("--fork-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--entry", action="append")
    ap.add_argument("--p369-summary")
    ap.add_argument("--idle-s", type=float, default=IDLE_S)
    args = ap.parse_args(argv)
    if args.idle_s < IDLE_S:
        ap.error(f"--idle-s must be at least {IDLE_S:.0f} (AGENTS.md)")
    sys.exit(drive(args))


if __name__ == "__main__":
    main()
