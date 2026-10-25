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

P405 R5 (``docs/specs/p405-kv-scheme-precedence.md`` v4, C192): ``codec --stack-root . --tree <wt> --ref <sha> --python <fork
venv python> --out <dir>`` runs the codec-only ABBA on ONE clean tree — arm A affine (the pick's ``kv_quant_scheme`` set to
``uniform`` in its overlay), arm B TurboQuant — at the deployed MTP-ON tuple (no forced block size), schedules ``short``
(8,192 + 256 tokens, 3 warmup + 5 measured per leg) then ``long`` (261,120 + 64, 0 + 2), ≥ 10 min idle before every leg after
the first; per request decode and prefill tok/s and the prefill / decode-phase peaks; descriptive statistics, no bound.
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
# P405 R5 (code round 1 K5): the codec-only ABBA — one attested tree, arm A affine, arm B TurboQuant (descriptive)
CODEC_ARMS = dict(A="uniform", B="turboquant")
CODEC_SCHEDULES = dict(
    short=dict(prompt=8192, tokens=256, warmup=3, measured=5),
    long=dict(
        prompt=261120, tokens=64, warmup=0, measured=2
    ),  # 255 × 1024 + 64 < the 262,144 cap
)
CODEC_METRICS = ("tps", "prefill_tps", "decode_peak", "prefill_peak")
VERIFIER_PICKS = (  # the P400 §5.3 width-2 verifier ABBA: the two Qwen3.8-27B picks (C188)
    "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed",
    "Qwen3.8-27B-mlx-uniform-4bit",
)
CODEC_PICKS = (
    "Qwen3.8-27B-mlx-uniform-4bit",
)  # code round 2 K9: quantized-KV picks only (Fable is native)


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


def _fixture():
    """The pinned golden registry of THIS tree (P369 v13 V5): the overlays come from the 16-state generator."""
    from p369_phase2 import registry_states

    return registry_states.fixture_text(
        Path(__file__).resolve().parents[1] / "mlx_vlm" / "tests" / "fixtures"
    )


def codec_overlay(text, pick, scheme):
    """The registry with exactly ``pick``'s ``kv_quant_scheme:`` line set to ``scheme`` (comment kept); refuses
    anything else."""
    import re

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
    hits = [
        i
        for i in range(start, end)
        if re.match(r"^\s*kv_quant_scheme:\s*\S+", lines[i])
    ]
    if len(hits) != 1:
        raise ValueError(f"{pick}: {len(hits)} kv_quant_scheme lines, expected 1")
    i = hits[0]
    lines[i] = re.sub(r"(kv_quant_scheme:\s*)\S+", rf"\g<1>{scheme}", lines[i], count=1)
    return "\n".join(lines)


def codec_arm_problems(argv_a, argv_b):
    """Code round 2 K9: arm A must resolve to affine and arm B to TurboQuant, with the same bits."""
    from p369_phase2 import kvmanifest

    a, b = kvmanifest.kv_from_argv(argv_a), kvmanifest.kv_from_argv(argv_b)
    problems = []
    if a["kv_bits"] is None or b["kv_bits"] is None:
        problems.append(f"native KV arm (kv_bits A {a['kv_bits']} / B {b['kv_bits']})")
    if (a["kv_quant_scheme"], b["kv_quant_scheme"]) != ("uniform", "turboquant"):
        problems.append(
            f"arms resolve to {a['kv_quant_scheme']} / {b['kv_quant_scheme']}"
        )
    if a["kv_bits"] != b["kv_bits"]:
        problems.append(f"arms differ in bits {a['kv_bits']} / {b['kv_bits']}")
    return problems


def request_record(probe, finish_reason, widths, live, *, warmup):
    """One request's result row: the measurement, its finish, the verify widths, and the FULL validated live cache
    manifest (P405 code round 3 K10)."""
    return dict(
        probe,
        finish_reason=finish_reason,
        live_cache=live,
        widths=sorted(set(widths)),
        warmup=warmup,
    )


def leg_cache_problems(observed, template, argv):
    """Code round 2 K7: the LIVE target cache of a measured request matches, layer by layer, what its argv implies."""
    from p369_phase2 import kvmanifest

    if observed is None:
        return ["no live target cache captured"]
    return kvmanifest.manifest_problems(
        observed, template, kvmanifest.kv_from_argv(argv)
    )


def codec_request_problems(probe, finish_reason, tokens):
    """A codec request decodes exactly the schedule's ``tokens`` to the length cap with finite positive prefill and
    decode rates and peaks (no width-2 requirement: the deployed tuple's own blocks)."""
    problems = [
        f"{name} {probe.get(name)!r} not finite positive"
        for name in CODEC_METRICS
        if not _positive(probe.get(name))
    ]
    if probe.get("tokens") != tokens:
        problems.append(f"decoded {probe.get('tokens')} tokens, not {tokens}")
    if finish_reason != "length":
        problems.append(f"finish_reason {finish_reason!r}, not 'length'")
    return problems


def codec_stats(legs):
    """Per schedule, per arm: n, median and [min, max] of each metric over the measured requests. Descriptive (R5): an
    input to the codec decision, never a gate."""
    out = {}
    for schedule in sorted({leg["schedule"] for leg in legs}):
        out[schedule] = {}
        for arm in ("A", "B"):
            reps = [
                r
                for leg in legs
                if leg["arm"] == arm and leg["schedule"] == schedule
                for r in leg["measured"]
            ]
            out[schedule][arm] = dict(
                n=len(reps),
                median={
                    m: statistics.median(r[m] for r in reps) if reps else None
                    for m in CODEC_METRICS
                },
                range={
                    m: (
                        [float(min(r[m] for r in reps)), float(max(r[m] for r in reps))]
                        if reps
                        else None
                    )
                    for m in CODEC_METRICS
                },
            )
    return out


def leg_schedule(picks):
    """Code round 2 C5: every globally consecutive leg (a pick's first included) is preceded by the idle."""
    plan = [
        dict(pick=pick, k=k, arm=arm) for pick in picks for k, arm in enumerate(ORDER)
    ]
    for i, step in enumerate(plan):
        step["idle_before"] = i > 0
    return plan


def served_check(worker, rg, argv, pick):
    """Code round 2 C6 (+ P405 F3): the deployed MTP tuple and EVERY resolved KV field the argv names are what the worker
    serves."""
    from p369_phase2 import kvmanifest, orchestrate

    want = kvmanifest.kv_from_argv(argv)
    state = worker.served_state()
    worker.assert_served(
        state,
        dict(
            drafter=True,
            verify_policy=orchestrate.PICKS[pick]["verify_policy"],
            kv=want,
        ),
    )
    got = kvmanifest.served_kv(rg, os.environ)
    problems = kvmanifest.kv_problems(got, want)
    if problems:
        raise RuntimeError(f"served KV differs from the argv: {problems}")
    return dict(state=state, kv=got)


# ------------------------------------------------------------------ one leg (a child process)
def registry_text(stack):
    """B4 code round 1 F1: the registry as BYTES decoded — never universal-newline normalized, so C203's raw CR check sees
    a CRLF file (as ``orchestrate.read_registry``)."""
    return (Path(stack) / "main_models.yaml").read_bytes().decode("utf-8")


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
        codec = job.get(
            "codec"
        )  # P405 R5: a codec leg (schedule dict) or the P400 verifier leg
        target = codec["prompt"] if codec else TARGET_PROMPT
        n_tokens = codec["tokens"] if codec else DECODE_TOKENS
        warmup = codec["warmup"] if codec else WARMUP
        measured = codec["measured"] if codec else MEASURED
        base = dict(model=job["pick"], stream=False)
        text, tokens = worker.size_prompt(
            lambda t: worker.prompt_tokens(
                client, [dict(role="user", content=t)], base
            ),
            target,
            prefix=orchestrate.SIZE_PREFIX,
            suffix=orchestrate.SIZE_SUFFIX,
        )
        if text is None:
            raise RuntimeError(f"could not size the prompt to {target} ({tokens})")
        gen_module = sys.modules["mlx_vlm.generate"]
        real_stream = gen_module.stream_generate
        probe = {}
        from p369_phase2 import kvmanifest

        from mlx_vlm.generate import dispatch

        real_step, captured = dispatch.generate_step, {}

        def capturing_step(
            *a, **k
        ):  # code round 2 K7: the live target cache of the request
            captured["cache"] = k.get("prompt_cache")
            yield from real_step(*a, **k)

        dispatch.generate_step = capturing_step
        template = kvmanifest.manifest(
            server.runtime.response_generator.model.language_model.make_cache()
        )

        def stream(*a, **k):
            first, last = True, None
            mx.synchronize()
            mx.reset_peak_memory()  # the prefill phase starts (P405 R5)
            for chunk in real_stream(*a, **k):
                if first:  # prefill done, first token out: the decode phase starts
                    mx.synchronize()
                    probe["prefill_peak"] = int(mx.get_peak_memory())
                    mx.reset_peak_memory()
                    first = False
                last = chunk
                yield chunk
            mx.synchronize()
            probe.update(
                decode_peak=int(mx.get_peak_memory()),
                tps=float(getattr(last, "generation_tps", 0.0) or 0.0),
                prefill_tps=float(getattr(last, "prompt_tps", 0.0) or 0.0),
                tokens=int(getattr(last, "generation_tokens", 0) or 0),
            )

        gen_module.stream_generate = stream
        payload = dict(
            model=job["pick"],
            messages=[dict(role="user", content=text)],
            max_tokens=n_tokens,
            seed=SEED,
            stream=False,
        )
        reps = []
        for i in range(warmup + measured):
            probe.clear()
            widths.clear()
            captured.clear()
            chat = f"p400-perf-{uuid.uuid4().hex}"
            r = client.post(
                "/v1/chat/completions",
                headers={"X-MLX-VLM-Chat-Id": chat},
                json=worker.routed(payload),
                timeout=7200 if codec else 1800,  # a 261K prefill
            )
            if r.status_code != 200:
                raise RuntimeError(f"request {i}: {r.status_code} {r.text[:300]}")
            cache = captured.get("cache")
            live = None if cache is None else kvmanifest.manifest(cache)
            cache_problems = leg_cache_problems(live, template, job["argv"])
            if cache_problems:
                raise RuntimeError(f"request {i}: live cache {cache_problems}")
            worker.release_session(chat)
            finish = r.json()["choices"][0].get("finish_reason")
            problems = (
                codec_request_problems(probe, finish, n_tokens)
                if codec
                else request_problems(probe, finish, widths)
            )
            if problems:
                raise RuntimeError(f"request {i}: {problems}")
            reps.append(request_record(probe, finish, widths, live, warmup=i < warmup))
        result.update(
            cache_template=template,  # K10: once per leg
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
    registry = registry_text(stack)
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
    picks = args.pick or list(VERIFIER_PICKS)
    argvs = {}
    for pick in picks:
        text, _ = orchestrate.overlay_registry(registry, pick, _fixture())
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


def drive_codec(args):
    """P405 R5: per pick and schedule (short, then long), ABBA legs A = affine, B = TurboQuant on ONE clean tree at
    ``--ref``, ≥ ``--idle-s`` before every globally consecutive leg, power checked per leg; every leg asserts the served
    codec its overlay names (``served_check``)."""
    sys.path.insert(0, str(DEV))
    from p369_phase2 import orchestrate

    stack = Path(args.stack_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    registry = registry_text(stack)
    tree = str(Path(args.tree).resolve())
    ref = resolve_ref(tree, args.ref)
    attest = attest_tree(tree)
    if attest["head"] != ref or not attest["clean"]:
        raise SystemExit(f"tree {tree}: {attest}, wanted clean at {ref}")
    picks = args.pick or list(CODEC_PICKS)
    argvs = {}
    for pick in picks:
        for arm, scheme in CODEC_ARMS.items():
            on, _ = orchestrate.overlay_registry(registry, pick, _fixture())
            text = codec_overlay(
                on, pick, scheme
            )  # the MTP-ON registry, then the arm's codec
            overlay = out / f"overlay-{pick}-{arm}.yaml"
            overlay.write_text(text)
            argvs[pick, arm] = orchestrate.worker_argv(stack, overlay, pick)
        problems = codec_arm_problems(argvs[pick, "A"], argvs[pick, "B"])
        if problems:
            raise SystemExit(f"{pick}: codec arms {problems}")
    report = dict(
        mode="codec",
        arms=CODEC_ARMS,
        schedules=CODEC_SCHEDULES,
        order=ORDER,
        tree=tree,
        ref=ref,
        attest=attest,
        idle_s=args.idle_s,
        picks={},
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    plan = [
        dict(pick=pick, schedule=schedule, k=k, arm=arm)
        for pick in picks
        for schedule in CODEC_SCHEDULES
        for k, arm in enumerate(ORDER)
    ]
    legs_of = {pick: [] for pick in picks}
    last_end = None
    for i, step in enumerate(plan):
        pick, schedule, k, arm = step["pick"], step["schedule"], step["k"], step["arm"]
        if i:
            time.sleep(args.idle_s)  # >= 10 min idle between legs (AGENTS.md)
        idle_before_s = None if last_end is None else time.monotonic() - last_end
        power = orchestrate.power_state()
        if not (power["ac_ok"] and power["battery_ok"]):
            raise SystemExit(f"power not OK before {pick} {schedule} leg {k}: {power}")
        name = f"{pick}-{schedule}-leg{k}-{arm}"
        job, res = out / f"{name}.job.json", out / f"{name}.json"
        job.write_text(
            json.dumps(
                dict(
                    argv=argvs[pick, arm],
                    pick=pick,
                    arm=arm,
                    out=str(res),
                    codec=CODEC_SCHEDULES[schedule],
                )
            )
        )
        env = orchestrate.worker_env(dict(os.environ))
        env["PYTHONPATH"] = tree
        with open(out / f"{name}.log", "w") as log:
            rc = subprocess.run(
                [args.python, __file__, "leg", str(job)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=6 * 3600,
                cwd=str(out),
            ).returncode
        last_end = time.monotonic()
        if rc != 0:
            raise SystemExit(f"{name} failed rc {rc}; see its log")
        leg_result = json.loads(res.read_text())
        if leg_result["tree"] != tree or leg_result["attest"] != attest:
            raise SystemExit(
                f"{name} loaded {leg_result['tree']} {leg_result['attest']}"
            )
        leg_result.update(schedule=schedule, power=power, idle_before_s=idle_before_s)
        legs_of[pick].append(leg_result)
        report["picks"][pick] = dict(
            legs=legs_of[pick], stats=codec_stats(legs_of[pick])
        )
        (out / "report.json").write_text(json.dumps(report, indent=1))
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({p: v["stats"] for p, v in report["picks"].items()}, indent=1))


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
    c = sub.add_parser("codec")  # P405 R5
    c.add_argument("--stack-root", required=True)
    c.add_argument("--tree", required=True)
    c.add_argument("--ref", required=True, help="the commit the tree must be at")
    c.add_argument("--out", required=True)
    c.add_argument("--pick", action="append")
    c.add_argument("--idle-s", type=float, default=600.0)
    c.add_argument("--python", required=True)
    lg = sub.add_parser("leg")
    lg.add_argument("job")
    args = ap.parse_args(argv)
    if args.cmd in ("drive", "codec") and args.idle_s < MIN_IDLE_S:
        ap.error(f"--idle-s must be at least {MIN_IDLE_S:.0f} (AGENTS.md)")
    if args.cmd == "drive":
        drive(args)
    elif args.cmd == "codec":
        drive_codec(args)
    else:
        leg(args.job)
        # P429: a completed leg skips interpreter finalization — a background thread's thread-local MLX compile cache
        # re-entered a finalized interpreter and segfaulted after the result was written (rc -11).
        import logging

        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
