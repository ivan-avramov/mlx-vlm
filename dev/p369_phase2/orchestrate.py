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
DROP_ENV = (
    "APC_ENABLED",
    "MLX_VLM_DRAFT_MODEL",
    "MLX_VLM_DRAFT_KIND",
    "MLX_VLM_DRAFT_BLOCK_SIZE",
    "MLX_VLM_DRAFT_COOLDOWN",
    "MLX_VLM_SUFFIX_MIN_MATCH",
    "MLX_VLM_MTP_VERIFY_SCAN",
    "MLX_VLM_MTP_VERIFY_AB",
)
DROP_PREFIXES = ("MLX_VLM_PRELOAD_",)
REFUSED_PROCESSES = (
    "mlx_vlm.server",
    "mlx-serve",
    "mlx_serve",
    "pytest",
    "opencode",
    "run_tg1",
    "run_m64",
    "worker.py",
)
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
        if k not in DROP_ENV and not k.startswith(DROP_PREFIXES)
    }
    env["MLX_VLM_CACHE_SESSION_MAX"] = "1"
    return env


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


def e1c_verdict(on_entry, off_entry, *, threshold, verify_policy):
    """Per-request E1c pass rule (plan §2.2): rows, state, kernels, engagement, exact prompt, straddle."""
    problems = []
    if not off_entry.get("rows", {}).get("pass"):
        problems.append("rows")
    if not off_entry.get("states", {}).get("pass"):
        problems.append("states")
    if not on_entry.get("verdict", {}).get("pass"):
        problems.append("kernels")
    if not on_entry.get("draft_rounds"):
        problems.append("no MTP rounds")
    if threshold is not None:
        if on_entry.get("prompt_end") != straddle_target(threshold):
            problems.append(
                f"prompt_end {on_entry.get('prompt_end')} != {straddle_target(threshold)}"
            )
        if verify_policy == "joint_v1":
            timings = (on_entry.get("response") or {}).get("timings") or {}
            if not timings.get("verify_blocks_straddle"):
                problems.append("no straddle counter")
            if timings.get("verify_fallback_reasons"):
                problems.append(
                    f"fallback reasons {timings['verify_fallback_reasons']}"
                )
        elif not on_entry.get("straddle"):
            problems.append("no observed per-query straddle")
    return dict(pass_=not problems, problems=problems)


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
        cmd = " ".join(p.info["cmdline"] or [])
        if any(token in cmd for token in REFUSED_PROCESSES):
            problems.append(f"process {p.info['pid']}: {cmd[:120]}")
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
    return dict(problems=problems, power=power, free_gb=free_gb)


def power_state():
    ac = subprocess.run(["pmset", "-g", "ac"], capture_output=True, text=True).stdout
    batt = subprocess.run(
        ["pmset", "-g", "batt"], capture_output=True, text=True
    ).stdout
    therm = subprocess.run(
        ["pmset", "-g", "therm"], capture_output=True, text=True
    ).stdout
    watts = re.search(r"Wattage\s*=\s*(\d+)W", ac)
    volts = re.search(r"(\d+(?:\.\d+)?)\s*V", ac) or re.search(
        r"Voltage\s*=\s*(\d+)", ac
    )
    pct = re.search(r"(\d+)%", batt)
    return dict(
        watts=int(watts.group(1)) if watts else None,
        volts=float(volts.group(1)) if volts else None,
        battery=int(pct.group(1)) if pct else None,
        ac_ok=bool(watts and int(watts.group(1)) == 140),
        battery_ok=bool(pct and int(pct.group(1)) > 20),
        thermal=" ".join(therm.split())[:200],
    )


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


def run_job(job, fork_root, env, monitor):
    job_dir = Path(job["out_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps(job, indent=1))
    monitor.current = job["name"]
    python = str(Path(fork_root) / ".venv" / "bin" / "python")
    with (job_dir / "worker.log").open("w") as log:
        proc = subprocess.run(
            [
                python,
                "-I",
                str(Path(job["dev_dir"]) / "p369_phase2" / "worker.py"),
                str(job_dir / "job.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=job["timeout_s"],
        )
    result = (
        json.loads((job_dir / "result.json").read_text())
        if (job_dir / "result.json").exists()
        else None
    )
    if proc.returncode != 0 or result is None or result.get("status") != "ok":
        monitor.errors.append(f"{job['name']}: rc {proc.returncode}")
        raise RuntimeError(
            f"job {job['name']} failed (rc {proc.returncode}); see {job_dir}/worker.log"
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


def e1c_timeout(threshold):
    tokens = threshold or 1024
    return int(tokens / 100 + E1C_TOKENS * 6 + 900)


def e2_timeout():
    return int(4096 / 100 + E2_TOKENS / 2 + 600)


def e7_timeout():
    return int(102400 / 8 + 600)


def weights_record(hf_path):
    root = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    )
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
        self.summary = dict(
            picks={}, started=time.strftime("%Y-%m-%dT%H:%M:%S"), phase=args.phase
        )

    # -- setup
    def prepare(self):
        self.out.mkdir(parents=True, exist_ok=False)
        self.arms = {}
        manifest = dict(
            production=PRODUCTION,
            fork_head=subprocess.run(
                ["git", "-C", str(self.fork), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            versions=fork_versions(self.fork),
            env_keys=sorted(self.env),
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
            manifest["picks"][pick] = dict(
                overlay_sha256=sha256_file(overlay),
                restored=restored,
                argv_on=on,
                argv_off=off,
                params=self.params[pick],
                target=weights_record(model),
                drafter=weights_record(drafter),
            )
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        return manifest

    def job(self, pick, name, mode, arm, requests, **extra):
        expect = dict(
            drafter=arm == "on",
            verify_policy=PICKS[pick]["verify_policy"] if arm == "on" else None,
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
        return run_job(job, self.fork, self.env, self.monitor)

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
                    payload=dict(
                        model=pick,
                        messages=[dict(role="user", content=text)],
                        max_tokens=E1C_TOKENS,
                        seed=seed,
                        stream=False,
                    ),
                    timeout_s=e1c_timeout(t),
                )
                for t, text in pending
            ]
            name = f"e1c-on-a{attempt}"
            self.run(self.job(pick, name, "e1c_on", "on", requests))
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
                    on_req=str(cell["dir"] / f"req-{cell['index']}.json"),
                    on_rows=str(cell["dir"] / f"rows-{cell['index']}.npz"),
                    on_snaps=str(cell["dir"] / f"snaps-{cell['index']}.npz"),
                    smoke_positives=smoke and k == len(names) - 1,
                )
            )
        self.run(self.job(pick, "e1c-off", "e1c_off", "off", requests))
        verdicts = {}
        for i, name in enumerate(names):
            cell = accepted[name]
            on = json.loads((cell["dir"] / f"req-{cell['index']}.json").read_text())
            off = json.loads(
                (self.out / pick / "e1c-off" / f"req-{i}.json").read_text()
            )
            v = e1c_verdict(
                on,
                off,
                threshold=cell["threshold"],
                verify_policy=PICKS[pick]["verify_policy"],
            )
            v.update(
                rows=off["rows"],
                states=off["states"],
                kernels=on["verdict"],
                smoke_positives=off.get("smoke_positives"),
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
            for n in on
        )
        return dict(
            identity=identity,
            all_identical=all(identity.values()),
            control_identical=control_ok,
            argmax_and_kernels=argmax_ok,
        )

    def e7(self, pick, items):
        reqs = self.c84_payloads(pick, items, greedy=False)
        self.run(self.job(pick, "e7a", "e7a", "on", reqs))
        per = [
            json.loads((self.out / pick / "e7a" / f"req-{i}.json").read_text())
            for i in range(len(reqs))
        ]
        n = len(per)
        classes = {}
        for cls in ("M4a", "M4b", "M4b_first", "M3"):
            k = sum(
                1
                for e in per
                if any(ev["cls"] == cls for ev in e["turns"][0]["events"])
            )
            classes[cls] = dict(
                requests_with_event=k, requests=n, ci95=clopper_pearson(k, n)
            )
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
        self.run(self.job(pick, "e7b", "e7b", "on", sessions))
        e7b = []
        for i, s in enumerate(sessions):
            entry = json.loads((self.out / pick / "e7b" / f"req-{i}.json").read_text())
            turns = entry["turns"]
            continuity = all(
                (turns[t]["final"]["cached_tokens"] or 0)
                > (turns[t - 1]["final"]["prompt_tokens"] or 0)
                for t in range(1, len(turns))
            )
            e7b.append(
                dict(
                    name=s["name"],
                    continuity=continuity,
                    censored=not continuity,
                    events=[ev["cls"] for t in turns for ev in t["events"]],
                )
            )
        return dict(e7a=classes, e7b=e7b)

    # -- the whole run
    def execute(self):
        manifest = self.prepare()
        estimate = 20 * 60 if self.args.phase == "smoke" else int(ESTIMATE_LIMIT_S)
        smoke = self.args.phase == "smoke"
        with Monitor(self.out / "monitor.log", estimate) as self.monitor:
            for pick in self.picks:
                started = time.monotonic()
                thresholds = (1024, 65536) if smoke else THRESHOLDS
                prompts = self.size(pick, thresholds)
                items = c84_items(self.stack, pick)
                self.check_reconstruction(pick, items[:2] if smoke else items)
                record = dict(e1c=self.e1c(pick, prompts, smoke=smoke))
                record["e2"] = self.e2(pick, items, prompts, smoke=smoke)
                if not smoke:
                    record.update(self.e7(pick, items))
                record["wall_s"] = round(time.monotonic() - started)
                self.summary["picks"][pick] = record
                self.write_summary()
        self.summary["manifest"] = dict(
            fork_head=manifest["fork_head"], production=PRODUCTION
        )
        if smoke:
            self.summary["smoke_gate"] = self.smoke_gate()
        self.write_summary()
        return self.summary

    def smoke_gate(self):
        """Every live positive fired (TurboQuant's required where TurboQuant KV is served), every E1c cell
        passed, and the projected full run fits ESTIMATE_LIMIT_S."""
        problems = []
        projected = 0
        for pick, record in self.summary["picks"].items():
            for name, v in record["e1c"].items():
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
            if (
                not record["e2"]["all_identical"]
                or not record["e2"]["argmax_and_kernels"]
            ):
                problems.append(f"{pick}: E2 smoke {record['e2']}")
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
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
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
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        summary = Campaign(args).execute()
    finally:
        os.close(fd)
        lock.unlink()
    print(json.dumps(summary.get("smoke_gate", {}), indent=1))


if __name__ == "__main__":
    main()
