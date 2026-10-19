"""P369 phase 2 orchestrator — run with the STACK's venv python (it imports mlx-serve and the stack bench, never
``mlx_vlm``). Plan: the stack's ``docs/specs/p369-phase2-run-plan.md`` (v4).

  <stack>/.venv/bin/python <fork>/dev/p369_phase2/orchestrate.py --stack-root <stack> --fork-root <fork> \\
      --out <STACK_WORKDIR>/p369/phase2/<run-id> --phase smoke|full [--pick NAME ...] [--plan-only]

Launch from a shell that sourced the stack's ``.env`` (as the lean router start does): the worker env is that
env minus every variable ``cli.main`` sets only conditionally and ``APC_ENABLED``, plus
``MLX_VLM_CACHE_SESSION_MAX=1``; only the variable NAMES are recorded. Each (pick, arm) job is one worker
subprocess (``python -I worker.py``) under a watchdog; a monitor appends to ``monitor.log`` every 5 min.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

PRODUCTION = "4e6d0e03"
PICKS = {
    "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed": dict(
        kv="native16", verify_policy="joint_v1"
    ),
    "Qwen3.8-27B-mlx-uniform-4bit": dict(kv="tq4", verify_policy=None),
}
THRESHOLDS = (1024, 8192, 32768, 65536)
C84_AXES = ("math500", "humanevalplus", "mbppplus", "cjudge")
# Code round 1 H3: the worker env is an ALLOWLIST (process / locale / Hugging Face cache plumbing); every
# generation knob (MLX_*, KV_*, TQ_*, APC_*, MIN_OUTPUT_TOKENS, future ones) is dropped, then exactly
# MLX_VLM_CACHE_SESSION_MAX=1 is set. mlx-serve spawns the worker with the router's inherited env, and the
# deployed tuple lives in the registry argv, not in env knobs.
ENV_ALLOW = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "LANG",
    "TERM",
    "TZ",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_CACHE",
    "XDG_CACHE_HOME",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "__CF_USER_TEXT_ENCODING",
)
ENV_ALLOW_PREFIXES = ("LC_",)
ENV_RECORDED_VALUES = ("MLX_VLM_CACHE_SESSION_MAX", "HF_HUB_OFFLINE")
# Code round 1 H11: refused by argv STRUCTURE (an element, a ``-m`` module root, a script path), never by a
# substring of a shell command string.
REFUSED_EXECUTABLES = ("mlx-serve", "mlx_serve", "pytest", "py.test", "opencode")
REFUSED_MODULES = ("mlx_vlm", "mlx_lm", "mlx_serve", "pytest", "bench")
REFUSED_SCRIPTS = (
    "worker.py",
    "run_tg1_probe.py",
    "run_opencode_probe.py",
    "run_opencode_probe_v2.py",
    "session_cache_probe.py",
)
REFUSED_SCRIPT_DIRS = ("benchmark", "mlx_vlm")
PORTS = (8000, 8091)
MIN_FREE_GB = 20
MONITOR_S = 300
C177_MARK = "# P369 INTERIM (C177"


# ------------------------------------------------------------------ pure parts (tested)
def overlay_registry(text, pick):
    """Restore exactly the C177 lines of ``pick``'s block: its commented ``draft_kind: mtp`` (marked), the
    commented ``draft_model`` on the next line, and (Fable) the marked ``mtp_verify_scan: joint_v1``.
    Returns ``(new_text, restored_lines)``; refuses anything else."""
    lines = text.split("\n")
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if re.match(rf"^\s*- name: {re.escape(pick)}\s*$", line)
        ),
        None,
    )
    if start is None:
        raise ValueError(f"{pick}: no registry block")
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if re.match(r"^\s*- name: ", lines[i])
        ),
        len(lines),
    )
    restored = []
    for i in range(start, end):
        m = re.match(r"^(\s*)# (draft_kind: mtp)\s+" + re.escape(C177_MARK), lines[i])
        if m:
            nxt = re.match(r"^(\s*)# (draft_model: \S+)", lines[i + 1])
            if nxt is None:
                raise ValueError(
                    f"{pick}: marked draft_kind without an adjacent draft_model"
                )
            lines[i] = m.group(1) + m.group(2)
            lines[i + 1] = nxt.group(1) + nxt.group(2)
            restored += [lines[i].strip(), lines[i + 1].strip()]
            continue
        m = re.match(
            r"^(\s*)# (mtp_verify_scan: joint_v1)\s+" + re.escape(C177_MARK), lines[i]
        )
        if m:
            lines[i] = m.group(1) + m.group(2)
            restored.append(lines[i].strip())
    want = 3 if PICKS[pick]["verify_policy"] == "joint_v1" else 2
    if len(restored) != want:
        raise ValueError(f"{pick}: restored {restored}, expected {want} C177 lines")
    return "\n".join(lines), restored


def worker_env(base):
    env = {
        k: v
        for k, v in base.items()
        if k in ENV_ALLOW or k.startswith(ENV_ALLOW_PREFIXES)
    }
    env["MLX_VLM_CACHE_SESSION_MAX"] = "1"
    return env


def dropped_env(base):
    """The NAMES of the launch-env variables the worker does not get (recorded; values never)."""
    kept = worker_env(base)
    return sorted(k for k in base if k not in kept)


def refused_process(argv):
    """Why a process with ``argv`` must not run beside the campaign (None = allowed)."""
    if not argv:
        return None
    names = [os.path.basename(a) for a in argv]
    for name in names:
        if name in REFUSED_EXECUTABLES:
            return f"executable {name}"
    if not names[0].lower().startswith("python"):
        return None
    args = argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-m" and i + 1 < len(args):
            root = args[i + 1].split(".")[0]
            return f"module {args[i + 1]}" if root in REFUSED_MODULES else None
        if arg == "-c":
            return None
        if arg in ("-X", "-W"):  # options that take a value
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        script = Path(arg)
        if script.name in REFUSED_SCRIPTS:
            return f"script {script.name}"
        if any(part in REFUSED_SCRIPT_DIRS for part in script.parts[:-1]):
            return f"script under {script.parent}"
        return None
    return None


def tree_status_problems(porcelain):
    """Code round 1 H9: the reviewed fork tree must be clean — no tracked change, no untracked file."""
    lines = [line for line in porcelain.splitlines() if line.strip()]
    return [f"fork worktree not clean: {lines[:10]}"] if lines else []


def parse_power(ac, batt):
    """``pmset -g ac`` / ``-g batt`` → watts, volts, battery % and the AGENTS.md rule (140 W / 28 V, > 20 %)."""
    watts = re.search(r"Wattage\s*=\s*(\d+)W", ac)
    millivolts = re.search(r"Voltage\s*=\s*(\d+)mV", ac)
    pct = re.search(r"(\d+)%", batt)
    watts = int(watts.group(1)) if watts else None
    volts = int(millivolts.group(1)) / 1000 if millivolts else None
    battery = int(pct.group(1)) if pct else None
    return dict(
        watts=watts,
        volts=volts,
        battery=battery,
        ac_ok=watts == 140 and volts == 28.0,
        battery_ok=battery is not None and battery > 20,
    )


BINDING_PICK_KEYS = (
    "overlay_sha256",
    "argv_on",
    "argv_off",
    "params",
    "target",
    "drafter",
)


BINDING_KEYS = (
    "production",
    "fork_head",
    "harness_tree",
    "reviewed_head",
    "versions",
    "env_digest",
    "stack_bench",
)


def binding(manifest):
    """Code rounds 1 H9, 2 H9: what a full run must share with the smoke that authorizes it — the
    production pin, the fork head and harness tree, the reviewed head, the dependency versions, the worker
    env digest, the stack bench tree + C84 bundle shas, and per pick the overlay / argv / params / weights.
    """
    return dict(
        {key: manifest[key] for key in BINDING_KEYS},
        picks={
            pick: {k: entry[k] for k in BINDING_PICK_KEYS}
            for pick, entry in sorted(manifest["picks"].items())
        },
    )


def binding_problems(smoke_summary, manifest):
    want = smoke_summary.get("binding")
    if not want:
        return ["the smoke summary carries no binding"]
    have = binding(manifest)
    problems = []
    if not want.get("reviewed_head"):
        problems.append(
            "the smoke ran on an unreviewed head: only a smoke on the reviewed head authorizes a full run"
        )
    problems += [
        f"{key}: smoke {want.get(key)!r} != now {have[key]!r}"
        for key in BINDING_KEYS
        if want.get(key) != have[key]
    ]
    if sorted(want.get("picks", {})) != sorted(have["picks"]):
        problems.append(
            f"picks: smoke {sorted(want.get('picks', {}))} != now {sorted(have['picks'])}"
        )
    for pick, entry in have["picks"].items():
        for key in BINDING_PICK_KEYS:
            if want.get("picks", {}).get(pick, {}).get(key) != entry[key]:
                problems.append(f"{pick} {key} differs from the smoke")
    return problems


def reviewed_head_problems(phase, reviewed, head):
    """P394 (refined): smoke iterations may run on any committed clean head; ``--reviewed-head`` (a commit a
    Cursor round cleared), when given, must equal HEAD; the full run requires it."""
    if reviewed is None:
        return (
            ["the full run needs --reviewed-head (a Cursor-cleared commit)"]
            if phase == "full"
            else []
        )
    if reviewed != head and not head.startswith(reviewed):
        return [f"--reviewed-head {reviewed} is not HEAD {head}"]
    return []


def env_digest(env):
    """The worker env's variable names and ONE sha256 over its sorted items — values are never recorded."""
    items = json.dumps(sorted(env.items()), separators=(",", ":"))
    return dict(names=sorted(env), sha256=hashlib.sha256(items.encode()).hexdigest())


def hf_hub_cache(env, home):
    """The Hugging Face hub cache as ``huggingface_hub`` resolves it: HF_HUB_CACHE, else
    HUGGINGFACE_HUB_CACHE, else ``$HF_HOME/hub``, else ``~/.cache/huggingface/hub``."""
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(key):
            return Path(env[key])
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "hub"
    return Path(home) / ".cache" / "huggingface" / "hub"


def tree_guard(fork_root, head):
    """Code round 2 H9: before every worker launch (and at exit) the fork is still at ``head`` with no
    tracked change and no untracked file — a mid-campaign edit or commit aborts the run.
    """

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(fork_root), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    now = git("rev-parse", "HEAD").strip()
    if now != head:
        raise RuntimeError(f"fork HEAD moved during the campaign: {head} -> {now}")
    problems = tree_status_problems(
        git("status", "--porcelain", "--untracked-files=all")
    )
    if problems:
        raise RuntimeError(problems[0])


def clopper_pearson(k, n, alpha=0.05):
    """Two-sided exact interval for k events in n independent trials (bisection on the binomial tail)."""
    if n == 0:
        return (0.0, 1.0)

    def cdf(x, p):
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, x + 1))

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - cdf(k - 1, p) < alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: cdf(k, p) > alpha / 2)
    return (lower, upper)


def straddle_target(threshold):
    return threshold - 64


def pick_outcome(verdicts):
    """Q37 precedence per pick (code round 2 H15): any C1 failure / incompleteness ABORTs (no E1c verdict,
    no bound); else an ESTABLISHED same-instance row / state difference FAILs (that pick's E1c finished for
    the bound, E2 / E7 skipped); else invalid evidence (kernels, engagement, prompt_end, straddle) is
    INCOMPLETE — never reported as non-neutrality; else PASS. The cross-load / configuration comparison (C2)
    never decides E1c: ``identical`` or ``unestablished``.
    """
    values = list(verdicts.values())
    if any(not v["c1"]["identical"] for v in values):
        e1c = "ABORT"
    elif any(not v["neutral"] for v in values):
        e1c = "FAIL"
    elif any(not v["evidence_ok"] or not v["pass_"] for v in values):
        e1c = "INCOMPLETE"
    else:
        e1c = "PASS"
    cross = (
        "identical" if all(v["c2"]["identical"] for v in values) else "unestablished"
    )
    return dict(e1c=e1c, run_e2_e7=e1c == "PASS", cross_load=cross)


def e2_verdict(*, all_identical, control_identical, engaged):
    """Code round 1 H10 (plan §3 E2): INCOMPLETE without engagement on every ON request (verifier rounds,
    argmax readout, kernel verdict); INFORMATIVE_ONLY when the OFF reload control differs; else PASS iff
    every request is identical. ``control_identical`` None = no control (the smoke)."""
    if not engaged:
        return "INCOMPLETE"
    if control_identical is False:
        return "INFORMATIVE_ONLY"
    return "PASS" if all_identical else "FAIL"


E7_CLASSES = ("M4a", "M4b", "M4b_first", "M3")


def _first_terminal(turn):
    """A turn whose prefill's first token was terminal: no MTP round can run (an ``M4b_first`` event)."""
    kinds = {ev["cls"] for ev in turn["events"]}
    return turn.get("mtp_rounds", 0) <= 0 and "M4b_first" in kinds


def e7a_stats(done, censored, *, digests=None):
    """E7a's inferential unit is an ENGAGED, VALID, uncensored request (code rounds 1 H5, 2 H6 / H10): MTP
    rounds ran — or the prefill's first token was terminal — and (unless first-terminal) the kernel verdict
    passed, and the templated prompt equals the reconstruction's digest (``digests``). Unengaged and
    invalid requests are reported, never counted event-free. Per class: requests with an event / n, with a
    Clopper–Pearson 95 % interval; ``status`` OBSERVED iff n >= 1, else INCOMPLETE."""
    engaged, not_engaged, invalid = {}, [], []
    for name, entry in done.items():
        turn = entry["turns"][0]
        first = _first_terminal(turn)
        if turn.get("mtp_rounds", 0) <= 0 and not first:
            not_engaged.append(name)
            continue
        digest_bad = digests is not None and (turn.get("prompt") or {}).get(
            "sha256"
        ) != digests.get(name)
        kernel_bad = not first and not (entry.get("verdict") or {}).get("pass")
        if digest_bad or kernel_bad:
            invalid.append(name)
            continue
        engaged[name] = {ev["cls"] for ev in turn["events"]}
    n = len(engaged)
    classes = {}
    for cls in E7_CLASSES:
        k = sum(1 for kinds in engaged.values() if cls in kinds)
        classes[cls] = dict(
            requests_with_event=k, requests=n, ci95=clopper_pearson(k, n)
        )
    return dict(
        requests=n,
        classes=classes,
        not_engaged=sorted(not_engaged),
        invalid=sorted(invalid),
        censored=list(censored),
        status="OBSERVED" if n else "INCOMPLETE",
    )


def e7b_session(name, entry, *, digest):
    """A session is OBSERVED (uncensored) iff its first prompt equals the reconstruction's ``digest``, the
    entry's kernel verdict passed, every turn ran MTP rounds (or was first-token terminal; code round 2 H18)
    and every follow-up turn is continuous (``worker.turn_continuity``, code round 1 H7); else censored with
    the first reason. Events are reported per session, never pooled with E7a."""
    turns = entry["turns"]
    reasons = []
    if (turns[0].get("prompt") or {}).get("sha256") != digest:
        reasons.append("the first prompt's digest differs from the reconstruction")
    if not (entry.get("verdict") or {}).get("pass"):
        reasons.append("the kernel verdict failed")
    reasons += [
        f"turn {t} ran no MTP rounds"
        for t, turn in enumerate(turns)
        if turn.get("mtp_rounds", 0) <= 0 and not _first_terminal(turn)
    ]
    reasons += [
        (t["continuity"] or {}).get("reason") or "no continuity record"
        for t in turns[1:]
        if not (t.get("continuity") or {}).get("continuity")
    ]
    return dict(
        name=name,
        continuity=all(
            (t.get("continuity") or {}).get("continuity") for t in turns[1:]
        ),
        censored=bool(reasons),
        reason=reasons[0] if reasons else None,
        events=[ev["cls"] for t in turns for ev in t["events"]],
        mtp_rounds=[t.get("mtp_rounds") for t in turns],
    )


def e7b_status(sessions):
    return "OBSERVED" if any(not s["censored"] for s in sessions) else "INCOMPLETE"


def prompt_digest_problems(want, entries):
    """Code round 1 H6: every C84 request's templated prompt digest equals the reconstruction's."""
    problems = []
    for name, entry in entries.items():
        if name not in want:
            continue
        got = (entry["turns"][0].get("prompt") or {}).get("sha256")
        if got != want[name]:
            problems.append(
                f"{name}: prompt digest {got} != reconstruction {want[name]}"
            )
    return problems


def overall_status(summary):
    """The phase status and the process exit code (code rounds 1 H10, 2 H10): 0 = PASS (full: every pick's
    E1c PASS, E2 PASS, E7a and E7b OBSERVED), 2 = completed without PASS (a verdict for the operator);
    harness errors record ``HARNESS_ERROR`` and exit 1."""
    if summary.get("phase") == "smoke":
        gate = summary.get("smoke_gate") or {}
        ok = bool(gate.get("pass_") and gate.get("fits"))
        return ("SMOKE_PASS", 0) if ok else ("SMOKE_NOT_PASS", 2)
    picks = summary.get("picks") or {}
    ok = bool(picks) and all(
        (r.get("outcome") or {}).get("e1c") == "PASS"
        and (r.get("e2") or {}).get("verdict") == "PASS"
        and (r.get("e7a") or {}).get("status") == "OBSERVED"
        and (r.get("e7b") or {}).get("status") == "OBSERVED"
        for r in picks.values()
    )
    return ("PASS", 0) if ok else ("NOT_PASS", 2)


def e1c_verdict(on_entry, off_entry, *, threshold, verify_policy):
    """Per-request E1c rule (plan §2.2; code round 2 H15): NEUTRALITY = rows and states equal; EVIDENCE =
    kernels, engagement, the exact prompt and the straddle. ``pass_`` needs both; ``pick_outcome`` reports
    a neutrality failure as FAIL and an evidence-only failure as INCOMPLETE."""
    neutrality, evidence = [], []
    if not off_entry.get("rows", {}).get("pass"):
        neutrality.append("rows")
    if not off_entry.get("states", {}).get("pass"):
        neutrality.append("states")
    if not on_entry.get("verdict", {}).get("pass"):
        evidence.append("kernels")
    if not on_entry.get("draft_rounds"):
        evidence.append("no MTP rounds")
    if threshold is not None:
        if on_entry.get("prompt_end") != straddle_target(threshold):
            evidence.append(
                f"prompt_end {on_entry.get('prompt_end')} != {straddle_target(threshold)}"
            )
        if verify_policy == "joint_v1":
            timings = (on_entry.get("response") or {}).get("timings") or {}
            if not timings.get("verify_blocks_straddle"):
                evidence.append("no straddle counter")
            if timings.get("verify_fallback_reasons"):
                evidence.append(
                    f"fallback reasons {timings['verify_fallback_reasons']}"
                )
        elif not on_entry.get("straddle"):
            evidence.append("no observed per-query straddle")
    return dict(
        pass_=not (neutrality or evidence),
        neutral=not neutrality,
        evidence_ok=not evidence,
        problems=neutrality + evidence,
        neutrality_problems=neutrality,
        evidence_problems=evidence,
    )


# ------------------------------------------------------------------ runtime parts
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(args, out):
    import psutil

    problems = []
    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] == me:
            continue
        why = refused_process(p.info["cmdline"] or [])
        if why:
            cmd = " ".join(p.info["cmdline"] or [])
            problems.append(f"process {p.info['pid']} ({why}): {cmd[:120]}")
    for (
        port
    ) in (
        PORTS
    ):  # system-wide psutil.net_connections needs root on macOS: lsof, as the stack does
        listening = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if listening:
            problems.append(
                f"port {port} is listening: {listening.splitlines()[-1][:120]}"
            )
    free_gb = shutil.disk_usage(out).free / 1e9
    if free_gb < MIN_FREE_GB:
        problems.append(f"free disk {free_gb:.1f} GB < {MIN_FREE_GB}")
    power = power_state()
    if not power["ac_ok"] or not power["battery_ok"]:
        problems.append(f"power {power}")
    diff = subprocess.run(
        [
            "git",
            "-C",
            args.fork_root,
            "diff",
            "--stat",
            PRODUCTION,
            "--",
            "mlx_vlm",
            ":!mlx_vlm/tests",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if diff:
        problems.append(f"production code differs from {PRODUCTION}:\n{diff}")
    porcelain = subprocess.run(
        ["git", "-C", args.fork_root, "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    problems += tree_status_problems(porcelain)
    return dict(problems=problems, power=power, free_gb=free_gb)


def power_state():
    ac = subprocess.run(["pmset", "-g", "ac"], capture_output=True, text=True).stdout
    batt = subprocess.run(
        ["pmset", "-g", "batt"], capture_output=True, text=True
    ).stdout
    therm = subprocess.run(
        ["pmset", "-g", "therm"], capture_output=True, text=True
    ).stdout
    return dict(parse_power(ac, batt), thermal=" ".join(therm.split())[:200])


def worker_argv(stack_root, overlay, pick):
    code = (
        "import json, os, sys\n"
        f"os.environ['MLX_SERVE_CONFIG'] = {str(overlay)!r}\n"
        "from mlx_serve import config, process_manager\n"
        f"models = config._load()[0]\n"
        f"print(json.dumps(process_manager._build_command(models[{pick!r}])[1:]))\n"
    )
    out = subprocess.run(
        [str(Path(stack_root) / ".venv" / "bin" / "python"), "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=stack_root,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


class Monitor:
    def __init__(self, path, estimate_s):
        self.path, self.estimate_s = Path(path), estimate_s
        self.started = time.monotonic()
        self.current = "-"
        self.current_dir = None
        self.errors = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(MONITOR_S):
            self.tick()

    def tick(self):
        elapsed = time.monotonic() - self.started
        with self.path.open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        elapsed_s=round(elapsed),
                        estimate_s=self.estimate_s,
                        eta_s=max(0, round(self.estimate_s - elapsed)),
                        job=self.current,
                        request=(
                            _read_json(self.current_dir / "progress.json")
                            if self.current_dir
                            else None
                        ),
                        errors=self.errors[-3:],
                        power=power_state(),
                    )
                )
                + "\n"
            )

    def __enter__(self):
        self.tick()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.tick()
        return False


BOOT_TIMEOUT_S = 900  # model load + served-state checks until the first heartbeat


class JobTimeout(RuntimeError):
    """A worker was stopped by the supervisor (code round 1 H8). ``info``: the outcome (``request_timeout`` /
    ``hang_after`` with the request index, ``boot_timeout``, ``job_timeout``); ``completed``: ``{index:
    entry}`` of the requests that finished before it."""

    def __init__(self, info, completed):
        super().__init__(f"worker stopped: {info}")
        self.info, self.completed = info, completed


class WallLimit(RuntimeError):
    """The campaign's total-wall limit was reached (the current worker was stopped)."""


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def launch_worker(cmd, *, stdout, env):
    """Code round 2 H17: every worker starts in its own session / process group, so a stop reaches every
    descendant it spawned."""
    return subprocess.Popen(
        cmd, stdout=stdout, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )


def _signal(proc, sig):
    """Signal ``proc``'s whole process group when it leads one (``launch_worker``), else ``proc`` alone."""
    try:
        if os.getpgid(proc.pid) == proc.pid:
            os.killpg(proc.pid, sig)
            return
    except ProcessLookupError:
        pass
    try:
        os.kill(proc.pid, sig)
    except ProcessLookupError:
        pass


def _stop(proc, grace_s):
    """SIGTERM, ``grace_s`` to exit, then SIGKILL to the group (descendants that outlived the worker too)."""
    import signal

    _signal(proc, signal.SIGTERM)
    try:
        proc.wait(grace_s)
    except subprocess.TimeoutExpired:
        pass
    _kill_group(proc)
    proc.wait()


def _kill_group(proc):
    import signal

    try:
        os.killpg(
            proc.pid, signal.SIGKILL
        )  # a group survives its leader while a member lives
    except (ProcessLookupError, PermissionError):
        if proc.poll() is None:
            proc.kill()


def supervise(
    proc,
    job_dir,
    *,
    boot_timeout_s,
    deadline,
    job_deadline=None,
    poll_s=2.0,
    grace_s=30.0,
):
    """Watch one worker until it exits; stop it (SIGTERM, then SIGKILL after ``grace_s``) when its current
    request (``progress.json``) overruns its deadline, when no heartbeat came within ``boot_timeout_s``, at
    the job deadline or at the campaign ``deadline`` (monotonic). Every stop writes ``timeout.json``.
    """
    job_dir = Path(job_dir)
    started = time.monotonic()
    while True:
        returncode = proc.poll()
        if returncode is not None:
            return dict(outcome="exited", returncode=returncode)
        now = time.monotonic()
        progress = _read_json(job_dir / "progress.json")
        info = None
        if now >= deadline:
            info = dict(outcome="wall_limit")
        elif job_deadline is not None and now >= job_deadline:
            info = dict(outcome="job_timeout")
        elif progress is None:
            if now - started > boot_timeout_s:
                info = dict(outcome="boot_timeout", waited_s=round(now - started, 1))
        elif time.time() > progress["started"] + progress["deadline_s"]:
            phase = progress.get("phase", "request")
            info = dict(
                outcome="request_timeout" if phase == "request" else "hang_after",
                index=progress["index"],
                name=progress["name"],
                deadline_s=progress["deadline_s"],
                elapsed_s=round(time.time() - progress["started"], 1),
            )
        if (
            info is not None
        ):  # code round 2 H8: the decision is durable BEFORE the signal
            tmp = job_dir / ".timeout.json.tmp"
            tmp.write_text(json.dumps(info, indent=1))
            os.replace(tmp, job_dir / "timeout.json")
            _stop(proc, grace_s)
            return info
        time.sleep(poll_s)


def run_with_censoring(run, requests):
    """E7 under the per-request watchdog: ``run(batch) -> {index: entry}`` raises ``JobTimeout``; a
    ``request_timeout`` censors that request, a ``hang_after`` censors none, and the requests after it are
    relaunched in a fresh worker. Any other stop re-raises. Returns ``({name: entry}, [censored names])``.
    """
    done, censored = {}, []
    pending = list(requests)
    while pending:
        try:
            entries = run(pending)
        except JobTimeout as exc:
            if exc.info.get("outcome") not in ("request_timeout", "hang_after"):
                raise
            index = exc.info["index"]
            stop = index if exc.info["outcome"] == "request_timeout" else index + 1
            for k, entry in exc.completed.items():
                if k < stop:
                    done[pending[k]["name"]] = entry
            if exc.info["outcome"] == "request_timeout":
                censored.append(pending[index]["name"])
            pending = pending[index + 1 :]
            continue
        for k, entry in entries.items():
            done[pending[k]["name"]] = entry
        pending = []
    return done, censored


def run_job(job, fork_root, env, monitor, *, deadline):
    job_dir = Path(job["out_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps(job, indent=1))
    monitor.current, monitor.current_dir = job["name"], job_dir
    python = str(Path(fork_root) / ".venv" / "bin" / "python")
    with (job_dir / "worker.log").open("w") as log:
        proc = launch_worker(
            [
                python,
                "-I",
                str(Path(job["dev_dir"]) / "p369_phase2" / "worker.py"),
                str(job_dir / "job.json"),
            ],
            stdout=log,
            env=env,
        )
        try:
            outcome = supervise(
                proc,
                job_dir,
                boot_timeout_s=BOOT_TIMEOUT_S,
                deadline=deadline,
                job_deadline=time.monotonic() + job["timeout_s"],
            )
        except BaseException:
            _stop(proc, 5)
            raise
        if outcome["outcome"] == "exited":
            _kill_group(proc)  # H17: nothing the worker spawned outlives it
    if outcome["outcome"] != "exited":
        monitor.errors.append(f"{job['name']}: {outcome}")
        if outcome["outcome"] == "wall_limit":
            raise WallLimit(f"campaign wall limit reached in {job['name']}")
        completed = {}
        for path in job_dir.glob("req-*.json"):
            entry = _read_json(path)
            if entry is not None:
                completed[int(path.stem.split("-")[1])] = entry
        raise JobTimeout(outcome, completed)
    result = _read_json(job_dir / "result.json")
    returncode = outcome["returncode"]
    if returncode != 0 or result is None or result.get("status") != "ok":
        monitor.errors.append(f"{job['name']}: rc {returncode}")
        raise RuntimeError(
            f"job {job['name']} failed (rc {returncode}); see {job_dir}/worker.log"
        )
    return result


# ------------------------------------------------------------------ the campaign
SEEDS = (
    101,
    202,
    303,
    404,
    505,
)  # pre-registered E1c seed list (an invalid straddle cell takes the next)
PROBE_TOKEN = 13
E1C_TOKENS = 160
E2_TOKENS = 512
FOLLOW_UPS = (
    "Summarize your previous answer in two sentences.",
    "Now point out one possible weakness in your previous answer.",
)
SIZE_PREFIX = (
    "Below is a long list of words. Read it, then write a short story that uses as many of them as "
    "you can."
)
SIZE_SUFFIX = "Now write the story."
SHORT_PROMPT = "Explain, step by step, why the sum of two odd integers is even."
ESTIMATE_LIMIT_S = 3.5 * 3600


def campaign_deadline(wall_limit_s, *, now):
    """C184: no campaign hard stop unless the operator sets ``--wall-limit-s``; the per-request watchdogs
    (non-convergent / wedged requests) are the only automatic stops."""
    return math.inf if wall_limit_s is None else now + wall_limit_s


def e1c_timeout(threshold):
    tokens = threshold or 1024
    return int(tokens / 100 + E1C_TOKENS * 6 + 900)


def e2_timeout():
    return int(4096 / 100 + E2_TOKENS / 2 + 600)


def e7_timeout():
    return int(102400 / 8 + 600)


def weights_record(hf_path):
    root = hf_hub_cache(
        os.environ, Path.home()
    )  # code round 2 H9: the worker's own resolution
    repo = root / ("models--" + hf_path.replace("/", "--"))
    ref = (repo / "refs" / "main").read_text().strip()
    snap = repo / "snapshots" / ref
    files = sorted(p for p in snap.iterdir() if p.suffix in (".safetensors", ".json"))
    return dict(
        hf_path=hf_path,
        revision=ref,
        files={p.name: sha256_file(p.resolve()) for p in files},
    )


def fork_versions(fork_root):
    code = (
        "import importlib.metadata as m, json\n"
        "print(json.dumps({p: m.version(p) for p in ('mlx', 'mlx-metal')}))\n"
    )
    out = subprocess.run(
        [str(Path(fork_root) / ".venv" / "bin" / "python"), "-I", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


C84_SCRIPT = """
import json, sys
from pathlib import Path
stack, pick, kv, axes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
sys.path.insert(0, str(Path(stack) / "benchmark"))
from bench import benchmarks
items = []
for axis in axes:
    path = Path(stack) / "benchmark" / "results" / pick / f"{axis}.c84bundle-after-{kv}-20260914.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    corpus = {str(item["id"]): item for item in benchmarks.load(axis)}
    for row in rows:
        item = corpus.get(str(row["id"]))
        if item is None:
            raise SystemExit(f"{axis}/{row['id']}: not in the corpus")
        items.append(dict(name=f"{axis}/{row['id']}", messages=benchmarks.build_messages(axis, item),
                          seed=row.get("sampler_seed"), prompt_tokens=row.get("prompt_tokens")))
print(json.dumps(items))
"""


def c84_items(stack_root, pick):
    """The C84-item reconstruction (plan §3 E2): the stored ids / seeds / prompt counts of the pick's C84
    rows, prompts rebuilt by the stack bench's own builders (in the bench venv, which has the datasets).
    """
    stack_root = Path(stack_root).resolve()
    out = subprocess.run(
        [
            str(Path(stack_root) / ".venv-bench" / "bin" / "python"),
            "-c",
            C84_SCRIPT,
            str(stack_root),
            pick,
            PICKS[pick]["kv"],
            ",".join(C84_AXES),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(stack_root) / "benchmark"),
    )
    if out.returncode != 0:
        raise RuntimeError(f"{pick}: C84 reconstruction failed: {out.stderr[-2000:]}")
    items = json.loads(out.stdout.strip().splitlines()[-1])
    if len(items) != 5 * len(C84_AXES):
        raise RuntimeError(f"{pick}: {len(items)} C84 items, expected 20")
    return items


class Campaign:
    def __init__(self, args):
        self.args = args
        self.stack, self.fork = (
            Path(args.stack_root).resolve(),
            Path(args.fork_root).resolve(),
        )
        self.out = Path(args.out).resolve()
        self.dev = self.fork / "dev"
        self.picks = args.pick or list(PICKS)
        self.registry = (self.stack / "main_models.yaml").read_text()
        import yaml

        entries = {e["name"]: e for e in yaml.safe_load(self.registry)["models"]}
        self.params = {
            p: {
                k: entries[p]["generation_defaults"][k]
                for k in ("temperature", "top_p", "top_k", "min_p")
            }
            for p in self.picks
        }
        self.env = worker_env(dict(os.environ))
        self.digests = {}  # pick -> {C84 item: templated prompt digest} (H6)
        self.attested = (
            {}
        )  # pick -> {target, drafter}: the snapshot revisions hashed (H9)
        self.wall_limit_s = args.wall_limit_s
        self.deadline = campaign_deadline(self.wall_limit_s, now=time.monotonic())
        self.summary = dict(
            picks={},
            started=time.strftime("%Y-%m-%dT%H:%M:%S"),
            phase=args.phase,
            wall_limit_s=self.wall_limit_s,
        )

    # -- setup
    def prepare(self):
        self.out.mkdir(parents=True, exist_ok=False)
        self.arms = {}

        def rev(spec):
            return subprocess.run(
                ["git", "-C", str(self.fork), "rev-parse", spec],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

        self.head = rev("HEAD")
        bundles = {
            str(p.relative_to(self.stack)): sha256_file(p)
            for pick in self.picks
            for p in sorted(
                (self.stack / "benchmark" / "results" / pick).glob(
                    f"*.c84bundle-after-{PICKS[pick]['kv']}-20260914.jsonl"
                )
            )
        }
        manifest = dict(
            production=PRODUCTION,
            fork_head=self.head,
            harness_tree=rev("HEAD:dev/p369_phase2"),
            reviewed_head=self.args.reviewed_head,
            versions=fork_versions(self.fork),
            env_keys=sorted(self.env),
            env_dropped=dropped_env(dict(os.environ)),
            env_values={k: self.env[k] for k in ENV_RECORDED_VALUES if k in self.env},
            env_digest=env_digest(self.env),
            stack_bench=dict(
                tree=subprocess.run(
                    ["git", "-C", str(self.stack), "rev-parse", "HEAD:benchmark/bench"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip(),
                bundles=bundles,
            ),
            picks={},
        )
        for pick in self.picks:
            text, restored = overlay_registry(self.registry, pick)
            overlay = self.out / f"overlay-{pick}.yaml"
            overlay.write_text(text)
            on = worker_argv(self.stack, overlay, pick)
            off = worker_argv(self.stack, self.stack / "main_models.yaml", pick)
            self.arms[pick] = dict(on=on, off=off)
            model = on[on.index("--model") + 1]
            drafter = on[on.index("--draft-model") + 1]
            entry = manifest["picks"][pick] = dict(
                overlay_sha256=sha256_file(overlay),
                restored=restored,
                argv_on=on,
                argv_off=off,
                params=self.params[pick],
                target=weights_record(model),
                drafter=weights_record(drafter),
            )
            self.attested[pick] = dict(
                target=entry["target"]["revision"], drafter=entry["drafter"]["revision"]
            )
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        return manifest

    def job(self, pick, name, mode, arm, requests, **extra):
        attested = self.attested[pick]
        revisions = {"--model": attested["target"]}
        if arm == "on":
            revisions["--draft-model"] = attested["drafter"]
        expect = dict(
            drafter=arm == "on",
            verify_policy=PICKS[pick]["verify_policy"] if arm == "on" else None,
            revisions=revisions,  # code round 2 H9: what the worker must actually load
        )
        timeout = sum(int(r["timeout_s"]) for r in requests) + 900
        return dict(
            name=f"{pick}/{name}",
            mode=mode,
            argv=self.arms[pick][arm],
            expect=expect,
            source_root=str(self.fork),
            dev_dir=str(self.dev),
            out_dir=str(self.out / pick / name),
            params=self.params[pick],
            probe_token=PROBE_TOKEN,
            requests=requests,
            timeout_s=timeout,
            **extra,
        )

    def run(self, job):
        tree_guard(self.fork, self.head)  # code round 2 H9: before every worker launch
        return run_job(job, self.fork, self.env, self.monitor, deadline=self.deadline)

    # -- steps
    def size(self, pick, thresholds):
        base = dict(model=pick, max_tokens=1, stream=False)
        requests = [
            dict(
                name=f"T{t}",
                target=straddle_target(t),
                prefix=SIZE_PREFIX,
                suffix=SIZE_SUFFIX,
                payload=base,
                timeout_s=600,
            )
            for t in thresholds
        ]
        result = self.run(self.job(pick, "size", "size", "off", requests))
        prompts = {}
        for r, t in zip(result["requests"], thresholds):
            if r["text"] is None:
                raise RuntimeError(
                    f"{pick}: no exact prompt for T={t} (closest {r['tokens']})"
                )
            prompts[t] = r["text"]
        return prompts

    def e1c(self, pick, prompts, *, smoke):
        cells = [(t, text) for t, text in prompts.items()] + (
            [] if smoke else [(None, SHORT_PROMPT)]
        )
        accepted, attempts = {}, {}
        pending = list(cells)
        for attempt, seed in enumerate(SEEDS):
            if not pending:
                break
            requests = [
                dict(
                    name=f"T{t}" if t else "short",
                    threshold=t,
                    smoke_positives=smoke and t == max(prompts),
                    payload=dict(
                        model=pick,
                        messages=[dict(role="user", content=text)],
                        max_tokens=E1C_TOKENS,
                        seed=seed,
                        stream=False,
                    ),
                    timeout_s=3 * e1c_timeout(t),  # ON + OFF1 + OFF2 (Q40)
                )
                for t, text in pending
            ]
            name = f"e1c-pair-a{attempt}"
            self.run(self.job(pick, name, "e1c_pair", "on", requests))
            still = []
            for i, ((t, text), req) in enumerate(zip(pending, requests)):
                path = self.out / pick / name / f"req-{i}.json"
                entry = json.loads(path.read_text())
                attempts.setdefault(req["name"], []).append(
                    dict(seed=seed, path=str(path))
                )
                n = len(entry["final"]["token_ids"])
                valid = t is None or (entry.get("straddle") and n >= 65)
                if valid:
                    accepted[req["name"]] = dict(
                        threshold=t, req=req, dir=path.parent, index=i
                    )
                else:
                    still.append((t, text))
            pending = still
        if pending:
            raise RuntimeError(
                f"{pick}: E1c cells invalid after {len(SEEDS)} seeds: {pending}"
            )
        names = list(accepted)
        requests = []
        for k, name in enumerate(names):
            cell = accepted[name]
            requests.append(
                dict(
                    cell["req"],
                    # control C2: the cross-load OFF arm against the same-instance OFF (P391)
                    on_req=str(cell["dir"] / f"req-{cell['index']}.json"),
                    on_rows=str(cell["dir"] / f"rows-off-{cell['index']}.npz"),
                    on_snaps=str(cell["dir"] / f"snaps-off-{cell['index']}.npz"),
                    smoke_positives=False,
                )
            )
        self.run(self.job(pick, "e1c-c2", "e1c_off", "off", requests))
        verdicts = {}
        for i, name in enumerate(names):
            cell = accepted[name]
            on = json.loads((cell["dir"] / f"req-{cell['index']}.json").read_text())
            c2 = json.loads((self.out / pick / "e1c-c2" / f"req-{i}.json").read_text())
            v = e1c_verdict(
                on,
                on["pair"],  # the verdict is the SAME-INSTANCE pair (P391)
                threshold=cell["threshold"],
                verify_policy=PICKS[pick]["verify_policy"],
            )
            c1_ok = bool(on["c1"]["acceptance"]["pass"])
            if not c1_ok:
                v["pass_"] = False
                v["problems"].append(f"C1: {on['c1']['acceptance']['problems']}")
            v.update(
                pair=on["pair"],
                c1=dict(identical=c1_ok, **on["c1"]),
                c2=dict(
                    identical=bool(c2["rows"]["pass"] and c2["states"]["pass"]),
                    rows=c2["rows"],
                    states=c2["states"],
                ),
                kernels=on["verdict"],
                smoke_positives=on.get("smoke_positives"),
                attempts=attempts[name],
            )
            verdicts[name] = v
        return verdicts

    def c84_payloads(self, pick, items, *, greedy, limit=None):
        out = []
        for item in items[:limit]:
            payload = dict(model=pick, messages=item["messages"], stream=False)
            if greedy:
                payload.update(temperature=0, max_tokens=E2_TOKENS)
            elif item["seed"] is not None:
                payload["seed"] = int(item["seed"])
            out.append(
                dict(
                    name=item["name"],
                    payload=payload,
                    timeout_s=e2_timeout() if greedy else e7_timeout(),
                )
            )
        return out

    def check_reconstruction(self, pick, items):
        requests = [
            dict(
                name=i["name"],
                payload=dict(model=pick, messages=i["messages"], max_tokens=1),
                timeout_s=120,
            )
            for i in items
        ]
        result = self.run(self.job(pick, "c84-count", "count", "off", requests))
        bad = [
            (i["name"], r["tokens"], i["prompt_tokens"])
            for i, r in zip(items, result["requests"])
            if r["tokens"] != i["prompt_tokens"]
        ]
        if bad:
            raise RuntimeError(
                f"{pick}: C84-item reconstruction differs from the recorded counts: {bad}"
            )
        # H6: the C84 bundles store counts only, so the link to the original runs is the count; every later
        # E2 / E7 request is bound to THIS reconstruction by its templated token digest.
        self.digests[pick] = {r["name"]: r["sha256"] for r in result["requests"]}
        return {
            r["name"]: dict(tokens=r["tokens"], sha256=r["sha256"])
            for r in result["requests"]
        }

    def e2(self, pick, items, prompts, *, smoke):
        reqs = self.c84_payloads(pick, items, greedy=True, limit=2 if smoke else None)
        straddles = (
            []
            if smoke
            else [
                dict(
                    name=f"T{t}",
                    payload=dict(
                        model=pick,
                        messages=[dict(role="user", content=text)],
                        temperature=0,
                        max_tokens=E1C_TOKENS,
                        stream=False,
                    ),
                    timeout_s=e1c_timeout(t),
                )
                for t, text in prompts.items()
            ]
        )
        arms = [("off", reqs + straddles), ("on", reqs + straddles)] + (
            [] if smoke else [("off", reqs)]
        )
        results = []
        for k, (arm, requests) in enumerate(arms):
            name = f"e2-{arm}-{k}"
            self.run(self.job(pick, name, "e2", arm, requests))
            results.append(
                {
                    r["name"]: json.loads(
                        (self.out / pick / name / f"req-{i}.json").read_text()
                    )
                    for i, r in enumerate(requests)
                }
            )
        off, on = results[0], results[1]
        control = results[2] if len(results) > 2 else None
        tokens = lambda e: e["turns"][0]["final"]["token_ids"]  # noqa: E731
        identity = {n: tokens(on[n]) == tokens(off[n]) for n in off}
        control_ok = (
            None
            if control is None
            else all(tokens(control[n]) == tokens(off[n]) for n in control)
        )
        argmax_ok = all(
            on[n]["telemetry"]["speculative_argmax_from_hidden"] > 0
            and on[n]["verdict"]["pass"]
            and on[n]["turns"][0]["mtp_rounds"] > 0
            for n in on
        )
        digest_problems = [
            p
            for arm in results
            for p in prompt_digest_problems(self.digests.get(pick, {}), arm)
        ]
        all_identical = all(identity.values())
        return dict(
            identity=identity,
            all_identical=all_identical,
            control_identical=control_ok,
            argmax_and_kernels=argmax_ok,
            prompt_digest_problems=digest_problems,
            verdict=e2_verdict(
                all_identical=all_identical,
                control_identical=control_ok,
                engaged=argmax_ok and not digest_problems,
            ),
        )

    def batch_runner(self, pick, base, mode, arm):
        """``run(batch) -> {index: entry}`` for ``run_with_censoring``: each (re)launch is a fresh worker job
        ``<base>-w<k>``."""
        launches = []

        def run(batch):
            name = f"{base}-w{len(launches)}"
            launches.append(name)
            self.run(self.job(pick, name, mode, arm, batch))
            return {
                i: json.loads((self.out / pick / name / f"req-{i}.json").read_text())
                for i in range(len(batch))
            }

        return run

    def e7a(self, pick, items):
        reqs = self.c84_payloads(pick, items, greedy=False)
        done, censored = run_with_censoring(
            self.batch_runner(pick, "e7a", "e7a", "on"), reqs
        )
        digests = self.digests.get(pick, {})
        return dict(
            e7a_stats(done, censored, digests=digests),
            prompt_digest_problems=prompt_digest_problems(digests, done),
        )

    def e7b(self, pick, items):
        sessions = []
        for item in items[:5]:
            turns = [
                dict(
                    model=pick,
                    messages=item["messages"],
                    stream=False,
                    **({"seed": int(item["seed"])} if item["seed"] is not None else {}),
                )
            ]
            turns += [
                dict(model=pick, messages=[dict(role="user", content=f)], stream=False)
                for f in FOLLOW_UPS
            ]
            sessions.append(
                dict(
                    name=item["name"], turns=turns, timeout_s=e7_timeout() * len(turns)
                )
            )
        done_b, censored_b = run_with_censoring(
            self.batch_runner(pick, "e7b", "e7b", "on"), sessions
        )
        digests = self.digests.get(pick, {})
        out = [dict(name=name, censored=True, reason="timeout") for name in censored_b]
        out += [
            e7b_session(s["name"], done_b[s["name"]], digest=digests.get(s["name"]))
            for s in sessions
            if s["name"] in done_b
        ]
        return dict(
            sessions=out,
            status=e7b_status(out),
            prompt_digest_problems=prompt_digest_problems(digests, done_b),
        )

    # -- the whole run
    def run_pick(self, pick, *, smoke):
        """One pick under the Q37 precedence. A supervisor stop (``JobTimeout``) can never pass: in E1c the
        pick ABORTs (no verdict, no bound), in E2 the identity is INCOMPLETE; E7 censors per request.
        """
        started = time.monotonic()
        record = self.summary["picks"].setdefault(pick, {})

        def save():  # C184: every finished step reaches summary.json at once
            record["wall_s"] = round(time.monotonic() - started)
            self.write_summary()

        thresholds = (1024, 65536) if smoke else THRESHOLDS
        prompts = self.size(pick, thresholds)
        items = c84_items(self.stack, pick)
        record["reconstruction"] = self.check_reconstruction(
            pick, items[:2] if smoke else items
        )
        save()
        try:
            record["e1c"] = self.e1c(pick, prompts, smoke=smoke)
            record["outcome"] = pick_outcome(record["e1c"])
        except JobTimeout as exc:
            record["outcome"] = dict(
                e1c="ABORT", run_e2_e7=False, reason=f"E1c worker stopped: {exc.info}"
            )
        save()
        if record["outcome"]["run_e2_e7"]:  # Q37 precedence
            try:
                record["e2"] = self.e2(pick, items, prompts, smoke=smoke)
            except JobTimeout as exc:
                record["e2"] = dict(
                    verdict="INCOMPLETE", reason=f"E2 worker stopped: {exc.info}"
                )
            save()
            if (
                not smoke
            ):  # code round 2 H16: E7a reaches summary.json before E7b starts
                record["e7a"] = self.e7a(pick, items)
                save()
                record["e7b"] = self.e7b(pick, items)
                save()
        return record

    def execute(self):
        manifest = self.prepare()
        self.summary["binding"] = binding(manifest)
        if self.args.smoke_summary:  # code round 1 H9: before any GPU job
            smoke_summary = json.loads(Path(self.args.smoke_summary).read_text())
            problems = binding_problems(smoke_summary, manifest)
            if problems:
                raise SystemExit(f"full: the smoke does not bind this run: {problems}")
        estimate = 20 * 60 if self.args.phase == "smoke" else int(ESTIMATE_LIMIT_S)
        smoke = self.args.phase == "smoke"
        with Monitor(self.out / "monitor.log", estimate) as self.monitor:
            try:
                for pick in self.picks:
                    self.run_pick(pick, smoke=smoke)
            except (
                WallLimit
            ) as exc:  # code round 1 H8: the worker was stopped; nothing more runs
                self.summary["aborted"] = f"wall_limit: {exc}"
                self.summary["status"], self.summary["exit_code"] = "ABORTED", 1
                self.write_summary()
                raise
            except Exception as exc:  # code round 2 H16: recorded before it propagates
                self.summary["status"], self.summary["exit_code"] = "HARNESS_ERROR", 1
                self.summary["error"] = f"{type(exc).__name__}: {exc}"
                self.write_summary()
                raise
        tree_guard(self.fork, self.head)  # code round 2 H9: unchanged through the end
        self.summary["manifest"] = dict(
            fork_head=manifest["fork_head"], production=PRODUCTION
        )
        if smoke:
            self.summary["smoke_gate"] = self.smoke_gate()
        self.summary["status"], self.summary["exit_code"] = overall_status(self.summary)
        self.write_summary()
        return self.summary

    def smoke_gate(self):
        """Every live positive fired (TurboQuant's required where TurboQuant KV is served), every E1c cell
        passed, and the projected full run fits ESTIMATE_LIMIT_S."""
        problems = []
        projected = 0
        for pick, record in self.summary["picks"].items():
            if record["outcome"]["e1c"] != "PASS":
                problems.append(f"{pick}: E1c {record['outcome']}")
            for name, v in record.get("e1c", {}).items():
                if not v["pass_"]:
                    problems.append(f"{pick} {name}: {v['problems']}")
                sp = v.get("smoke_positives")
                if sp is not None:
                    fired = sp["fired"]
                    need = {"raw_ulp", "in_support", "recurrent", "probe_logit"}
                    need.add(
                        "tq_norm" if PICKS[pick]["kv"] == "tq4" else "old_prefix_k"
                    )
                    if not sp["baseline_pass"] or not need <= {
                        k for k, ok in fired.items() if ok
                    }:
                        problems.append(f"{pick}: smoke positives {sp}")
            e2 = record.get("e2")
            if e2 is None or e2.get("verdict") != "PASS":
                problems.append(f"{pick}: E2 smoke {e2}")
            projected += (
                record["wall_s"] * 6
            )  # the full run is ≈ 6× the smoke per pick (plan §4)
        return dict(
            problems=problems,
            projected_full_s=projected,
            fits=projected <= ESTIMATE_LIMIT_S,
            pass_=not problems,
        )

    def write_summary(self):
        tmp = self.out / ".summary.json.tmp"
        tmp.write_text(json.dumps(self.summary, indent=1, default=str))
        os.replace(tmp, self.out / "summary.json")


def acquire_lock(lock):
    """Code round 2 H14: the exclusive run lock is ``fcntl.flock`` on a stable descriptor — the kernel
    releases it when the holder exits or dies, so there is no stale-lock reclaim and the file is never
    unlinked (no pathname race). The holder's ``{pid, started}`` is written for diagnostics only.
    """
    import fcntl

    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        try:
            holder = lock.read_text().strip() or "unreadable"
        except OSError:
            holder = "unreadable"
        raise SystemExit(f"run lock {lock} is held: {holder}")
    os.ftruncate(fd, 0)
    os.write(fd, json.dumps(dict(pid=os.getpid(), started=time.time())).encode())
    os.fsync(fd)
    return fd


def release_lock(fd):
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-root", required=True)
    ap.add_argument("--fork-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=("smoke", "full"), required=True)
    ap.add_argument("--pick", action="append")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument(
        "--smoke-summary", help="full runs refuse unless this passed smoke summary fits"
    )
    ap.add_argument(
        "--wall-limit-s",
        type=float,
        help="an operator-set hard total-wall stop (default: none, C184)",
    )
    ap.add_argument(
        "--reviewed-head",
        help="the Cursor-cleared fork commit; must equal HEAD; required for --phase full (P394)",
    )
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(
        ["git", "-C", args.fork_root, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    problems = reviewed_head_problems(args.phase, args.reviewed_head, head)
    if problems:
        raise SystemExit(f"refused: {problems}")
    if args.reviewed_head:
        args.reviewed_head = head  # recorded as the full sha
    if args.phase == "full":
        if not args.smoke_summary:
            raise SystemExit("full: --smoke-summary of a passed smoke is required")
        gate = json.loads(Path(args.smoke_summary).read_text()).get("smoke_gate") or {}
        if not (gate.get("pass_") and gate.get("fits")):
            raise SystemExit(f"full: the smoke gate did not pass / fit: {gate}")
    checks = preflight(args, out.parent)
    print(json.dumps(dict(preflight=checks), indent=1))
    if args.plan_only:
        return
    if checks["problems"]:
        raise SystemExit(f"preflight refused: {checks['problems']}")
    lock = out.parent / ".lock"
    fd = acquire_lock(lock)
    import signal

    def on_term(
        signum, _frame
    ):  # H17: a TERM stops the current worker group via run_job
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)  # HUP stays as nohup set it (ignored)
    try:
        summary = Campaign(args).execute()
    finally:
        release_lock(fd)
    print(
        json.dumps(
            dict(status=summary["status"], smoke_gate=summary.get("smoke_gate")),
            indent=1,
        )
    )
    if summary["exit_code"]:
        raise SystemExit(summary["exit_code"])


if __name__ == "__main__":
    main()
