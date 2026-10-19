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
import sys
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
    raise NotImplementedError


def worker_env(base):
    raise NotImplementedError


def clopper_pearson(k, n, alpha=0.05):
    raise NotImplementedError


def straddle_target(threshold):
    return threshold - 64


def e1c_verdict(on_entry, off_entry, *, threshold, verify_policy):
    raise NotImplementedError


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
    for conn in psutil.net_connections(kind="tcp"):
        if (
            conn.status == psutil.CONN_LISTEN
            and conn.laddr
            and conn.laddr.port in PORTS
        ):
            problems.append(f"port {conn.laddr.port} is listening (pid {conn.pid})")
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-root", required=True)
    ap.add_argument("--fork-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=("smoke", "full"), required=True)
    ap.add_argument("--pick", action="append")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args(argv)
    raise SystemExit(
        "orchestrate: the job sequence (sizing → E1c ON / OFF → E2 → E7) is built in the next commit"
    )


if __name__ == "__main__":
    main()
