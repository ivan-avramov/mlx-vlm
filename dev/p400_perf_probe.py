"""P400 §5.3 performance probe (fork-only, dev): the width-2 verifier change, old production tree vs new, ABBA.

Pre-registered (``docs/specs/p400-tq-width2-fix.md`` v4 §5.3, C188): per pick, its deployed MTP-ON tuple (the registry
overlay restoring the C177 lines, served by the fork's real CLI in-process) with ``--draft-block-size 2``, a fixed
8,192-token prompt and 256 decoded tokens at a fixed seed; legs A (old tree) B (new) B A, each 3 warmup + 5 measured
requests, ≥ 10 min idle between legs, 140 W checked before every leg. Per measured request: decode tok/s
(``generation_tps``) and the DECODE-PHASE peak (``mx.reset_peak_memory`` at the first streamed chunk, read at the end).
Per arm: the median of its 10 measured requests (and the max peak). Bounds: median tok/s regression ≤ 3 %, median
decode-phase peak increase ≤ 1 %. The production tree is chosen by ``PYTHONPATH``; each leg records the tree it loaded.

Usage (from the stack root, detached, under the STACK venv like the orchestrator): ``.venv/bin/python
<p400>/dev/p400_perf_probe.py drive --stack-root . --old-tree <p369 wt> --new-tree <p400 wt> --python <fork venv python>
--out $STACK_WORKDIR/p400/perf/<run>``.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

DEV = Path(__file__).resolve().parent
TARGET_PROMPT = 8192
DECODE_TOKENS = 256
WARMUP, MEASURED = 3, 5
SEED = 7
ORDER = ("A", "B", "B", "A")
BOUNDS = dict(tps_regression=0.03, peak_increase=0.01)


# ------------------------------------------------------------------ pure parts (tested)
def arm_stats(legs):
    """``legs``: [{"arm", "measured": [{"tps", "decode_peak"}]}] → per arm: median tok/s, median / max decode peak, n."""
    out = {}
    for arm in ("A", "B"):
        reps = [r for leg in legs if leg["arm"] == arm for r in leg["measured"]]
        out[arm] = dict(
            n=len(reps),
            median_tps=statistics.median(r["tps"] for r in reps) if reps else None,
            median_peak=(
                statistics.median(r["decode_peak"] for r in reps) if reps else None
            ),
            max_peak=max((r["decode_peak"] for r in reps), default=None),
        )
    return out


def verdict(stats, bounds=BOUNDS):
    """Old (A) vs new (B): regression = 1 - B/A for tok/s, increase = B/A - 1 for the decode-phase peak."""
    a, b = stats["A"], stats["B"]
    if a["n"] != 2 * MEASURED or b["n"] != 2 * MEASURED:
        return dict(status="INCOMPLETE", reason=f"n A {a['n']} B {b['n']}")
    tps_reg = 1.0 - b["median_tps"] / a["median_tps"]
    peak_inc = b["median_peak"] / a["median_peak"] - 1.0
    ok = tps_reg <= bounds["tps_regression"] and peak_inc <= bounds["peak_increase"]
    return dict(
        status="WITHIN_BOUNDS" if ok else "OUTSIDE_BOUNDS",
        tps_regression=tps_reg,
        peak_increase=peak_inc,
        bounds=bounds,
    )


# ------------------------------------------------------------------ one leg (a child process)
def leg(job_path):
    job = json.loads(Path(job_path).read_text())
    sys.path.insert(0, str(DEV))
    import mlx.core as mx
    from p369_phase2 import orchestrate, worker

    import mlx_vlm

    worker.ROUTED_MODEL = worker.argv_model(job["argv"])
    client = worker.boot(job["argv"])
    result = dict(arm=job["arm"], tree=str(Path(mlx_vlm.__file__).resolve().parents[1]))
    with client:
        from mlx_vlm.models.qwen3_5 import speculative_verifier as sv

        widths = []
        real_attention = sv.Qwen3_5BatchInvariantForward._attention

        def attention(self, attn, x, *a, **k):
            widths.append(int(x.shape[1]))
            return real_attention(self, attn, x, *a, **k)

        sv.Qwen3_5BatchInvariantForward._attention = attention
        base = dict(model=job["pick"], stream=False)
        text, tokens = worker.size_prompt(
            lambda t: worker.prompt_tokens(
                client, [dict(role="user", content=t)], base
            ),
            TARGET_PROMPT,
            prefix=orchestrate.SIZE_PREFIX,
            suffix=orchestrate.SIZE_SUFFIX,
        )
        if text is None:
            raise RuntimeError(
                f"could not size the prompt to {TARGET_PROMPT} ({tokens})"
            )
        gen_module = sys.modules["mlx_vlm.generate"]
        real_stream = gen_module.stream_generate
        probe = {}

        def stream(*a, **k):
            first, last = True, None
            for chunk in real_stream(*a, **k):
                if first:  # prefill done, first token out: the decode phase starts
                    mx.synchronize()
                    mx.reset_peak_memory()
                    first = False
                last = chunk
                yield chunk
            mx.synchronize()
            probe.update(
                decode_peak=int(mx.get_peak_memory()),
                tps=float(getattr(last, "generation_tps", 0.0) or 0.0),
                tokens=int(getattr(last, "generation_tokens", 0) or 0),
            )

        gen_module.stream_generate = stream
        payload = dict(
            model=job["pick"],
            messages=[dict(role="user", content=text)],
            max_tokens=DECODE_TOKENS,
            seed=SEED,
            stream=False,
        )
        reps = []
        for i in range(WARMUP + MEASURED):
            probe.clear()
            widths.clear()
            chat = f"p400-perf-{uuid.uuid4().hex}"
            r = client.post(
                "/v1/chat/completions",
                headers={"X-MLX-VLM-Chat-Id": chat},
                json=worker.routed(payload),
                timeout=1800,
            )
            if r.status_code != 200:
                raise RuntimeError(f"request {i}: {r.status_code} {r.text[:300]}")
            worker.release_session(chat)
            if not probe or probe["tps"] <= 0:
                raise RuntimeError(f"request {i}: no decode measurement {probe}")
            if 2 not in widths:
                raise RuntimeError(
                    f"request {i}: no width-2 verify block ({sorted(set(widths))})"
                )
            reps.append(dict(probe, widths=sorted(set(widths)), warmup=i < WARMUP))
        result.update(
            prompt_tokens=tokens,
            warmup=[r for r in reps if r["warmup"]],
            measured=[r for r in reps if not r["warmup"]],
        )
    Path(job["out"]).write_text(json.dumps(result, indent=1))


# ------------------------------------------------------------------ the driver
def drive(args):
    sys.path.insert(0, str(DEV))
    from p369_phase2 import orchestrate

    stack = Path(args.stack_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    registry = (stack / "main_models.yaml").read_text()
    trees = dict(
        A=str(Path(args.old_tree).resolve()), B=str(Path(args.new_tree).resolve())
    )
    report = dict(
        order=ORDER, trees=trees, picks={}, started=time.strftime("%Y-%m-%dT%H:%M:%S")
    )
    for pick in args.pick or list(orchestrate.PICKS):
        text, _ = orchestrate.overlay_registry(registry, pick)
        overlay = out / f"overlay-{pick}.yaml"
        overlay.write_text(text)
        argv = orchestrate.worker_argv(stack, overlay, pick) + [
            "--draft-block-size",
            "2",
        ]
        legs = []
        for k, arm in enumerate(ORDER):
            if k:
                time.sleep(args.idle_s)  # >= 10 min idle between legs (AGENTS.md)
            power = orchestrate.power_state()
            if not (power["ac_ok"] and power["battery_ok"]):
                raise SystemExit(f"power not OK before leg {k}: {power}")
            job = out / f"{pick}-leg{k}-{arm}.job.json"
            res = out / f"{pick}-leg{k}-{arm}.json"
            job.write_text(
                json.dumps(dict(argv=argv, pick=pick, arm=arm, out=str(res)))
            )
            env = orchestrate.worker_env(dict(os.environ))
            env["PYTHONPATH"] = trees[arm]
            log = open(out / f"{pick}-leg{k}-{arm}.log", "w")
            rc = subprocess.run(
                [args.python, __file__, "leg", str(job)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3 * 3600,
                cwd=str(out),
            ).returncode
            log.close()
            if rc != 0:
                raise SystemExit(f"{pick} leg {k} ({arm}) failed rc {rc}; see its log")
            leg_result = json.loads(res.read_text())
            if leg_result["tree"] != trees[arm]:
                raise SystemExit(
                    f"leg {k} loaded {leg_result['tree']}, wanted {trees[arm]}"
                )
            leg_result["power"] = power
            legs.append(leg_result)
        stats = arm_stats(legs)
        report["picks"][pick] = dict(legs=legs, stats=stats, verdict=verdict(stats))
        (out / "report.json").write_text(json.dumps(report, indent=1))
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({p: v["verdict"] for p, v in report["picks"].items()}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drive")
    d.add_argument("--stack-root", required=True)
    d.add_argument("--old-tree", required=True)
    d.add_argument("--new-tree", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--pick", action="append")
    d.add_argument("--idle-s", type=float, default=600.0)
    d.add_argument(
        "--python",
        required=True,
        help="the fork venv's python (mlx + the fork's deps) for the legs",
    )
    lg = sub.add_parser("leg")
    lg.add_argument("job")
    args = ap.parse_args(argv)
    if args.cmd == "drive":
        drive(args)
    else:
        leg(args.job)


if __name__ == "__main__":
    main()
