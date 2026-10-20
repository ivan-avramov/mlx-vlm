"""P400 §5.3 performance probe (fork-only, dev): the width-2 verifier change, old production tree vs new, ABBA.

Pre-registered (``docs/specs/p400-tq-width2-fix.md`` v4 §5.3, C188): per pick, its deployed MTP-ON tuple (the registry
overlay restoring the C177 lines, served by the fork's real CLI in-process) with ``--draft-block-size 2``, a fixed
8,192-token prompt and 256 decoded tokens at a fixed seed; legs A (old tree) B (new) B A, each 3 warmup + 5 measured
requests, ≥ 10 min idle between legs, 140 W checked before every leg. Per measured request: decode tok/s
(``generation_tps``) and the DECODE-PHASE peak (``mx.reset_peak_memory`` at the first streamed chunk, read at the end).
Per arm: the median of its 10 measured requests (and the max peak). Bounds: median tok/s regression ≤ 3 %, median
decode-phase peak increase ≤ 1 %. The production tree is chosen by ``PYTHONPATH``; each leg records the tree it loaded.

Usage (from the stack root, detached, under the STACK venv like the orchestrator): ``.venv/bin/python
<p400>/dev/p400_perf_probe.py drive --stack-root . --old-tree <old wt> --old-ref <sha> --new-tree <new wt> --new-ref <sha>
--python <fork venv python> --out $STACK_WORKDIR/p400/perf/<run>``. Code round 2: each arm is a clean tree at its expected
commit (re-attested per leg), every globally consecutive leg is preceded by >= 600 s idle, every request decodes exactly 256
tokens to the length cap, and each leg asserts the served MTP tuple and the KV codec its argv names.
"""

import argparse
import json
import math
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
MIN_IDLE_S = 600.0  # AGENTS.md: >= 10 min idle between arms


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
    inputs = [a["median_tps"], b["median_tps"], a["median_peak"], b["median_peak"]]
    if not all(_positive(x) for x in inputs):  # code round 3 C9
        return dict(
            status="INVALID", reason=f"non-finite / non-positive medians {inputs}"
        )
    tps_reg = 1.0 - b["median_tps"] / a["median_tps"]
    peak_inc = b["median_peak"] / a["median_peak"] - 1.0
    ok = tps_reg <= bounds["tps_regression"] and peak_inc <= bounds["peak_increase"]
    return dict(
        status="WITHIN_BOUNDS" if ok else "OUTSIDE_BOUNDS",
        tps_regression=tps_reg,
        peak_increase=peak_inc,
        bounds=bounds,
    )


def _positive(x):
    """A finite number > 0 (code round 3 C9: inf / nan / 0 must never reach the statistic)."""
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
        and x > 0
    )


def request_problems(probe, finish_reason, widths):
    """Code round 2 C3: a measured request decodes exactly DECODE_TOKENS to the length cap and saw a width-2 block."""
    problems = []
    if not probe or not _positive(probe.get("tps")):
        problems.append(f"no decode measurement {probe}")
    elif not _positive(probe.get("decode_peak")):  # code round 3 C9
        problems.append(f"no decode-phase peak {probe}")
    elif probe.get("tokens") != DECODE_TOKENS:
        problems.append(f"decoded {probe.get('tokens')} tokens, not {DECODE_TOKENS}")
    if finish_reason != "length":
        problems.append(f"finish_reason {finish_reason!r}, not 'length'")
    if 2 not in widths:
        problems.append(f"no width-2 verify block ({sorted(set(widths))})")
    return problems


def attest_tree(path):
    """The tree's HEAD and whether its working tree is clean (code round 2 C4)."""
    git = ["git", "-C", str(path)]
    head = subprocess.run(
        git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        git + ["status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return dict(head=head, clean=not dirty)


def resolve_ref(path, ref):
    """Code round 3 C8: a ref (any spelling) → its full commit id in ``path``'s repository; never a prefix."""
    out = subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "rev-parse",
            "--verify",
            "--quiet",
            f"{ref}^{{commit}}",
        ],
        capture_output=True,
        text=True,
    )
    if not ref or out.returncode != 0:
        raise SystemExit(f"cannot resolve {ref!r} to a commit in {path}")
    return out.stdout.strip()


def tree_problems(trees, refs, attest):
    """Code round 2 C4 / round 3 C8: two distinct clean trees, each EXACTLY at its expected commit, the two commits
    different. ``refs``: full resolved ids (``resolve_ref``); ``attest``: path → attest_tree().
    """
    problems = []
    if trees["A"] == trees["B"]:
        problems.append(f"both arms load {trees['A']}")
    for arm in ("A", "B"):
        if not (
            len(refs[arm]) == 40 and all(c in "0123456789abcdef" for c in refs[arm])
        ):
            problems.append(f"arm {arm}: ref {refs[arm]!r} is not a full commit id")
    if refs["A"] == refs["B"]:
        problems.append(f"both arms expect the same commit {refs['A']}")
    for arm in ("A", "B"):
        got = attest[trees[arm]]
        if got["head"] != refs[arm]:
            problems.append(
                f"arm {arm}: {trees[arm]} at {got['head']}, not {refs[arm]}"
            )
        if not got["clean"]:
            problems.append(f"arm {arm}: {trees[arm]} has uncommitted changes")
    return problems


def leg_schedule(picks):
    """Code round 2 C5: every globally consecutive leg (a pick's first included) is preceded by the idle."""
    plan = [
        dict(pick=pick, k=k, arm=arm) for pick in picks for k, arm in enumerate(ORDER)
    ]
    for i, step in enumerate(plan):
        step["idle_before"] = i > 0
    return plan


def kv_expected(argv):
    """The KV codec the argv asks for: ``bits`` None = native KV (``--kv-bits`` absent or 0; the scheme is moot)."""
    opts = dict(zip(argv, argv[1:]))
    bits = float(opts.get("--kv-bits") or 0) or None
    return dict(
        bits=bits, scheme=opts.get("--kv-quant-scheme", "uniform") if bits else None
    )


def served_check(worker, rg, argv, pick):
    """Code round 2 C6 (+ P405): the deployed MTP tuple and the KV codec the argv names are what the worker serves."""
    from p369_phase2 import orchestrate

    state = worker.served_state()
    worker.assert_served(
        state,
        dict(drafter=True, verify_policy=orchestrate.PICKS[pick]["verify_policy"]),
    )
    want = kv_expected(argv)
    bits = float(rg.kv_bits or 0) or None
    got = dict(bits=bits, scheme=rg.kv_quant_scheme if bits else None)
    if got != want:
        raise RuntimeError(f"served KV {got} != the argv's {want}")
    return dict(state, kv=got)


# ------------------------------------------------------------------ one leg (a child process)
def leg(job_path):
    job = json.loads(Path(job_path).read_text())
    sys.path.insert(0, str(DEV))
    import mlx.core as mx
    from p369_phase2 import orchestrate, worker

    import mlx_vlm

    worker.ROUTED_MODEL = worker.argv_model(job["argv"])
    client = worker.boot(job["argv"])
    tree = Path(mlx_vlm.__file__).resolve().parents[1]
    result = dict(arm=job["arm"], tree=str(tree), attest=attest_tree(tree))
    with client:
        import mlx_vlm.server as server

        result["served"] = served_check(
            worker, server.runtime.response_generator, job["argv"], job["pick"]
        )
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
            finish = r.json()["choices"][0].get("finish_reason")
            problems = request_problems(probe, finish, widths)
            if problems:
                raise RuntimeError(f"request {i}: {problems}")
            reps.append(
                dict(
                    probe,
                    finish_reason=finish,
                    widths=sorted(set(widths)),
                    warmup=i < WARMUP,
                )
            )
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
    refs = dict(  # code round 3 C8: full ids, resolved in each arm's own tree
        A=resolve_ref(trees["A"], args.old_ref), B=resolve_ref(trees["B"], args.new_ref)
    )
    attest = {path: attest_tree(path) for path in set(trees.values())}
    problems = tree_problems(trees, refs, attest)
    if problems:
        raise SystemExit(f"arm trees: {problems}")
    report = dict(
        order=ORDER,
        trees=trees,
        refs=refs,
        attest=attest,
        idle_s=args.idle_s,
        picks={},
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    picks = args.pick or list(orchestrate.PICKS)
    argvs = {}
    for pick in picks:
        text, _ = orchestrate.overlay_registry(registry, pick)
        overlay = out / f"overlay-{pick}.yaml"
        overlay.write_text(text)
        argvs[pick] = orchestrate.worker_argv(stack, overlay, pick) + [
            "--draft-block-size",
            "2",
        ]
    legs_of = {pick: [] for pick in picks}
    last_end = None
    for step in leg_schedule(picks):
        pick, k, arm, argv = step["pick"], step["k"], step["arm"], argvs[step["pick"]]
        legs = legs_of[pick]
        if step["idle_before"]:
            time.sleep(args.idle_s)  # >= 10 min idle between legs (AGENTS.md)
        idle_before_s = None if last_end is None else time.monotonic() - last_end
        power = orchestrate.power_state()
        if not (power["ac_ok"] and power["battery_ok"]):
            raise SystemExit(f"power not OK before leg {k}: {power}")
        job = out / f"{pick}-leg{k}-{arm}.job.json"
        res = out / f"{pick}-leg{k}-{arm}.json"
        job.write_text(json.dumps(dict(argv=argv, pick=pick, arm=arm, out=str(res))))
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
        last_end = time.monotonic()
        if rc != 0:
            raise SystemExit(f"{pick} leg {k} ({arm}) failed rc {rc}; see its log")
        leg_result = json.loads(res.read_text())
        if leg_result["tree"] != trees[arm]:
            raise SystemExit(
                f"leg {k} loaded {leg_result['tree']}, wanted {trees[arm]}"
            )
        if leg_result["attest"] != attest[trees[arm]]:
            raise SystemExit(
                f"leg {k} tree {leg_result['attest']}, attested {attest[trees[arm]]}"
            )
        leg_result.update(power=power, idle_before_s=idle_before_s)
        legs.append(leg_result)
        if k == len(ORDER) - 1:
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
    d.add_argument("--old-ref", required=True, help="the commit arm A must be at")
    d.add_argument("--new-ref", required=True, help="the commit arm B must be at")
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
    if args.cmd == "drive" and args.idle_s < MIN_IDLE_S:
        ap.error(f"--idle-s must be at least {MIN_IDLE_S:.0f} (AGENTS.md)")
    if args.cmd == "drive":
        drive(args)
    else:
        leg(args.job)


if __name__ == "__main__":
    main()
