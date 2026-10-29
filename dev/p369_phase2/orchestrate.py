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
from contextlib import contextmanager
from pathlib import Path

try:
    from p369_phase2 import kvmanifest, registry_states, telemetry
except ImportError:  # run as a script: this directory is sys.path[0]
    import kvmanifest
    import registry_states
    import telemetry  # code round 2 M2: stdlib-only at import

PRODUCTION = "b26fa7c9"  # S′ (C236 / C242): the fork at c688e722 / 1c7c8fa1, production tree = b26fa7c9; was 7207179b, 3395cef3, da27f0ef, 4e6d0e03
_DENSE = dict(
    layer="Qwen3_5DecoderLayer",
    depth=1,
    experts=None,
    quant=[(4, 64)],
    text_type="qwen3_5_text",
    block_size=3,  # every certified drafter's configured block (two drafts + the bonus); no override deployed
)
PICKS = {  # P369 v13 §3.1: every MTP-certified registry entry; one entry per campaign (V9)
    "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed": dict(
        kv="native16", verify_policy="joint_v1", drafter=_DENSE
    ),
    "Qwen3.8-27B-mlx-uniform-4bit": dict(kv="tq4", verify_policy=None, drafter=_DENSE),
    # no C84 bundles of their own: E2 / E7 use the reviewed borrowed panel of the uniform pick's items (V4)
    "Qwen3.6-27B-Opus-Distill-OptiQ-4bit": dict(
        kv="tq4",
        verify_policy=None,
        items_from="Qwen3.8-27B-mlx-uniform-4bit",
        drafter=_DENSE,
    ),
    "Ornith-1.0-35B-mlx-uniform-4bit": dict(
        kv="native16",
        verify_policy=None,
        items_from="Qwen3.8-27B-mlx-uniform-4bit",
        drafter=dict(  # the transplanted MoE head (M27): routed experts, never dense layers (V12 / V14)
            layer="Qwen3_5MoeDecoderLayer",
            depth=1,
            experts=(256, 8),
            quant=[(4, 64)],
            text_type="qwen3_5_moe_text",
            block_size=3,
            banks=("QuantizedSwitchLinear", 4, 64),
        ),
    ),
}
THRESHOLDS = (1024, 8192, 32768, 65536)
SMOKE_THRESHOLDS = (1024, 65536)  # the positive block rides the last (code round 2 C2)
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


# ------------------------------------------------------------------ pure parts (tested)
def read_registry(path):
    """The registry as BYTES decoded (code round 1 M1): never universal-newline normalized, so a CRLF or mixed-newline
    file is not mistaken for an accepted LF state."""
    return Path(path).read_bytes().decode("utf-8")


def draft_inventory_problems(pick, name, dk):
    """Code round 1 M2: the drafter MoE-path inventory itself — MoE, at least one invocation, each running every
    part exactly once, with selected experts — never only its pass flag. Code round 2 M2: each invocation names a
    ``drafter.`` block and a non-empty list of unique integer expert ids within the entry's expert count.
    """
    where = f"{pick} {name}.draft_kernels"
    if not isinstance(dk, dict) or not isinstance(dk.get("invocations"), list):
        raise MalformedEvidence(f"malformed evidence: {where} = {dk!r}")
    problems = []
    if not exact_bool(dk.get("pass"), f"{where}.pass"):
        problems.append(f"{pick} {name}: drafter MoE path {dk.get('problems')}")
    if dk.get("moe") is not True:
        problems.append(f"{pick} {name}: the drafter inventory is not a MoE one")
    if not dk["invocations"]:
        problems.append(f"{pick} {name}: no drafter MoE-block invocation")
    want = sorted(telemetry.DRAFT_MOE_PARTS)
    n_experts = PICKS[pick]["drafter"]["experts"][0]
    for inv in dk["invocations"]:
        if not isinstance(inv, dict) or inv.get("parts") != want:
            problems.append(f"{pick} {name}: drafter MoE invocation parts {inv!r}")
            continue
        block, experts = inv.get("block"), inv.get("selected_experts")
        if not (
            isinstance(block, str) and block.startswith("drafter.") and len(block) > 8
        ):
            problems.append(f"{pick} {name}: drafter MoE invocation block {block!r}")
        if not (
            isinstance(experts, list)
            and experts
            and all(type(e) is int and 0 <= e < n_experts for e in experts)
            and len(set(experts)) == len(experts)
        ):
            problems.append(
                f"{pick} {name}: drafter MoE invocation selected experts {experts!r}"
            )
    return problems


def needs_draft_inventory(pick):
    """v13 V15: an entry whose certified drafter is MoE must show its drafter's routed path in the smoke."""
    return bool(PICKS[pick]["drafter"].get("experts"))


def campaign_pick_problems(picks):
    """P369 v13 V9: exactly one known entry per campaign (smoke and full bind one entry each)."""
    if not picks or len(picks) != 1:
        return [f"a campaign carries exactly one --pick, got {picks!r}"]
    if picks[0] not in PICKS:
        return [f"unknown pick {picks[0]!r}"]
    return []


def argv_scan_problems(pick, argv):
    """P405 round 3 T15: the argv's verify scan (the CLI default ``per_query`` when absent) is the entry's."""
    opts = dict(zip(argv, argv[1:]))
    got = opts.get("--mtp-verify-scan", "per_query")
    want = PICKS[pick]["verify_policy"] or "per_query"
    return (
        [] if got == want else [f"{pick}: argv verify scan {got!r}, expected {want!r}"]
    )


PANEL = Path(__file__).resolve().parent / "borrowed_panel.json"


def items_source(pick):
    """``(the pick whose C84 bundles the items come from, its kv label)`` — the borrowed entries use the uniform
    pick's items (v13 V4)."""
    source = PICKS[pick].get("items_from", pick)
    return source, PICKS[source]["kv"]


def messages_sha(messages):
    blob = json.dumps(
        messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_panel(source, bundles, items, entries):
    """The borrowed-panel fixture (v13 V4): the source bundles' sha256, per item its seed, dataset-content hash and
    built-message sha256, and per borrowed entry its templated ``{name: {tokens, sha256}}`` (from ``--phase panel``).
    """
    return dict(
        source_pick=source,
        bundles=dict(sorted(bundles.items())),
        items=[
            dict(
                name=i["name"],
                seed=i["seed"],
                item_sha256=i["item_sha256"],
                corpus_sha256=i[
                    "corpus_sha256"
                ],  # code round 1 M9: the dataset content, per axis
                messages_sha256=messages_sha(i["messages"]),
            )
            for i in items
        ],
        entries=entries,
    )


def panel_problems(panel, pick, bundles, items, counts=None):
    """Live reconstruction vs the reviewed panel: bundles, items (names, seeds, content, messages) and — when
    given — the entry's live templated counts / digests must be exactly the pinned ones.
    """
    problems = []
    if pick not in panel.get("entries", {}):
        return [f"{pick}: not in the borrowed panel"]
    if (
        panel.get("source_pick") != items_source(pick)[0]
    ):  # code round 1 M9: derived independently
        return [
            f"{pick}: panel source {panel.get('source_pick')!r} != {items_source(pick)[0]!r}"
        ]
    if dict(sorted(bundles.items())) != panel["bundles"]:
        problems.append("source bundles differ from the panel")
    live = build_panel(panel["source_pick"], bundles, items, {})["items"]
    if live != panel["items"]:
        diff = [
            a["name"] for a, b in zip(live, panel["items"]) if a != b
        ] or "the item set"
        problems.append(f"reconstructed items differ from the panel: {diff}")
    if counts is not None and counts != panel["entries"][pick]:
        diff = sorted(
            n
            for n in set(counts) | set(panel["entries"][pick])
            if counts.get(n) != panel["entries"][pick].get(n)
        )
        problems.append(
            f"{pick}: templated counts / digests differ from the panel: {diff}"
        )
    return problems


def borrowed_panel(pick, path):
    """P369 v13 V4: the reviewed borrowed-panel fixture for an entry without C84 bundles (None for an entry with its
    own); absent → Refused, never a silent reconstruction."""
    if "items_from" not in PICKS[pick]:
        return None
    path = Path(path)
    if not path.exists():
        raise Refused(f"refused: {pick} needs its reviewed borrowed panel {path}")
    return json.loads(path.read_text())


def overlay_registry(text, pick, fixture):
    """P369 v13 V5 / V10: ``pick`` MTP-ON in ``text`` via the ONE byte-state generator — ``text`` must be one of the
    16 accepted states of the pinned ``fixture`` (else ValueError); the other entries keep their state and the pick's
    block equals the fixture's. Returns ``(new_text, restored fixture lines)`` (none when already ON).
    """
    return registry_states.overlay_on(text, fixture, pick)


def worker_env(base):
    env = {
        k: v
        for k, v in base.items()
        if k in ENV_ALLOW or k.startswith(ENV_ALLOW_PREFIXES)
    }
    env["MLX_VLM_CACHE_SESSION_MAX"] = "1"
    # code round 4 H9: the worker loads only the cached (attested) snapshots — no mid-campaign download can
    # change the weights; network access does not change generation
    env["HF_HUB_OFFLINE"] = "1"
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
    if (
        reviewed != head
    ):  # code round 3 H9: full ids only (resolve_commit), never a prefix
        return [f"--reviewed-head {reviewed} is not HEAD {head}"]
    return []


def env_digest(env):
    """The worker env's variable names and ONE sha256 over its sorted items — values are never recorded."""
    items = json.dumps(sorted(env.items()), separators=(",", ":"))
    return dict(names=sorted(env), sha256=hashlib.sha256(items.encode()).hexdigest())


def resolve_commit(repo, value):
    """``value`` (any commit-ish) as the full commit id in ``repo``; anything else refuses."""
    out = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "--verify",
            "--quiet",
            f"{value}^{{commit}}",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise SystemExit(f"--reviewed-head {value!r} is not a commit in {repo}")
    return out.stdout.strip()


def hf_hub_cache(env, home):
    """The Hugging Face hub cache as ``huggingface_hub`` resolves it: HF_HUB_CACHE, else
    HUGGINGFACE_HUB_CACHE, else ``$HF_HOME/hub``, else ``$XDG_CACHE_HOME/huggingface/hub``, else
    ``~/.cache/huggingface/hub``."""
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(key):
            return Path(env[key])
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "hub"
    if env.get("XDG_CACHE_HOME"):
        return Path(env["XDG_CACHE_HOME"]) / "huggingface" / "hub"
    return Path(home) / ".cache" / "huggingface" / "hub"


def stack_bench_state(stack, bundles):
    """The stack bench inputs as bound: the committed ``benchmark/bench`` tree and each bundle file's sha256."""
    tree = subprocess.run(
        ["git", "-C", str(stack), "rev-parse", "HEAD:benchmark/bench"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return dict(
        tree=tree,
        bundles={rel: sha256_file(Path(stack) / rel) for rel in sorted(bundles)},
    )


def stack_bench_drift(stack, want):
    """Code round 4 H9: the stack bench inputs still equal the manifest (a clean checkout that MOVED is
    drift too)."""
    have = stack_bench_state(stack, want["bundles"])
    return [
        f"stack bench {key} drifted from the manifest"
        for key in ("tree", "bundles")
        if have[key] != want[key]
    ]


def stack_bench_problems(stack, paths):
    """Code round 3 H9: the stack inputs the C84 reconstruction imports (``benchmark/bench``, the bundles)
    must be clean in the worktree — the binding hashes their committed trees / files."""
    porcelain = subprocess.run(
        [
            "git",
            "-C",
            str(stack),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *paths,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = [line for line in porcelain.splitlines() if line.strip()]
    return [f"stack bench inputs not clean: {lines[:10]}"] if lines else []


def lock_path(env):
    """Code round 3 H21: ONE canonical campaign lock, independent of ``--out``."""
    if not env.get("STACK_WORKDIR"):
        raise SystemExit("STACK_WORKDIR is not set (source the machine config)")
    return Path(env["STACK_WORKDIR"]) / "p369" / "phase2" / ".lock"


def out_problems(out, env):
    """Out-of-repo artifacts belong under ``$STACK_WORKDIR`` (AGENTS.md); the campaign's under p369/phase2."""
    root = (
        Path(env.get("STACK_WORKDIR", "/nonexistent")) / "p369" / "phase2"
    ).resolve()
    out = Path(out).resolve()
    if not out.is_relative_to(root) or out == root:
        return [f"--out {out} is not a run directory under {root}"]
    return []


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


STRADDLE_OFFSETS = (
    64,
    65,
    66,
)  # P537 (stack C252): pre-registered prompt_end offsets below a threshold, in order


def straddle_target(threshold, offset=STRADDLE_OFFSETS[0]):
    return threshold - offset


def e2_prompts(sized, accepted):
    """P537: E2's per-threshold straddle prompt is the one E1c accepted (its offset may not be 64)."""
    return {t: accepted.get(t, text) for t, text in sized.items()}


def registry_off_problems(picks, off):
    """P527 (stack C252): the OFF arm is the stack root's ``main_models.yaml`` as committed, so every pick must be
    MTP-OFF there — a pick ON in it would compare ON with ON."""
    return [
        f"{pick} is MTP-ON in the stack root registry: the OFF arm would be ON (use an all-OFF stack root)"
        for pick in picks
        if pick not in off
    ]


class MalformedEvidence(RuntimeError):
    """Code round 6 H34 (C186): evidence not of its exact type — a harness defect, never an MTP outcome."""


def exact_bool(value, what):
    """Code round 6 H34: an evidence flag is exactly ``True`` or ``False``; anything else RAISES."""
    if type(value) is not bool:
        raise MalformedEvidence(f"malformed evidence: {what} = {value!r}")
    return value


def pick_outcome(verdicts):
    """Q37 precedence per pick (code round 2 H15): any C1 failure / incompleteness ABORTs (no E1c verdict,
    no bound); else an ESTABLISHED same-instance row / state difference FAILs (that pick's E1c finished for
    the bound, E2 / E7 skipped); else invalid evidence (kernels, engagement, prompt_end, straddle) is
    INCOMPLETE — never reported as non-neutrality; else PASS. The cross-load / configuration comparison (C2)
    never decides E1c: ``identical`` or ``unestablished``.
    """

    def flag(name, v, path):
        node = v
        for key in path.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        return exact_bool(node, f"{name}.{path}")

    def rejected(name, v):  # P399: attempts retried for validity are observations too
        attempts = v.get("rejected") if isinstance(v, dict) else None
        if not isinstance(attempts, list):
            raise MalformedEvidence(
                f"malformed evidence: {name}.rejected = {attempts!r}"
            )
        return [
            {
                key: exact_bool(
                    a.get(key) if isinstance(a, dict) else None,
                    f"{name}.rejected[{k}].{key}",
                )
                for key in ("neutral", "c1")
            }
            for k, a in enumerate(attempts)
        ]

    keys = ("c1.identical", "neutral", "evidence_ok", "pass_", "c2.identical")
    values = [  # code round 6 H34: every cell's flags validated before any decides
        dict({key: flag(name, v, key) for key in keys}, rejected=rejected(name, v))
        for name, v in verdicts.items()
    ]
    if any(
        not v["c1.identical"] or any(not a["c1"] for a in v["rejected"]) for v in values
    ):
        e1c = "ABORT"
    elif any(
        not v["neutral"] or any(not a["neutral"] for a in v["rejected"]) for v in values
    ):
        e1c = "FAIL"
    elif any(not v["evidence_ok"] or not v["pass_"] for v in values):
        e1c = "INCOMPLETE"
    else:
        e1c = "PASS"
    cross = "identical" if all(v["c2.identical"] for v in values) else "unestablished"
    return dict(e1c=e1c, run_e2_e7=e1c == "PASS", cross_load=cross)


def e2_verdict(*, c1_identical, engaged, all_identical):
    """C185 (plan v8 §3 E2) precedence per pick: the within-instance control OFF1 vs OFF2 differing ABORTs
    (the instrument cannot decide; never an MTP finding); an ON request not engaged (argmax readout, kernel
    verdict, MTP rounds) or a prompt digest problem is INCOMPLETE; else PASS iff ON equals OFF1 on every
    request, FAIL otherwise. Every flag is an exact boolean (code round 6 H34)."""
    c1 = exact_bool(c1_identical, "e2_verdict.c1_identical")
    engaged = exact_bool(engaged, "e2_verdict.engaged")
    identical = exact_bool(all_identical, "e2_verdict.all_identical")
    if not c1:
        return "ABORT"
    if not engaged:
        return "INCOMPLETE"
    return "PASS" if identical else "FAIL"


def _exact_int(value):
    return type(value) is int


def is_sha256(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def e2_turn_checked(where, turn, *, readout=False):
    """C185 review round 1 E1: a turn's evidence is typed — token ids ``list[int]``, a ``str`` finish reason,
    exact-int MTP rounds (and, for an ON turn, the argmax readout count) — else RAISES.
    """
    final = turn.get("final") if isinstance(turn, dict) else None
    ids = final.get("token_ids") if isinstance(final, dict) else None
    ok = (
        type(ids) is list
        and all(_exact_int(t) for t in ids)
        and type(final.get("finish_reason")) is str
        and _exact_int(turn.get("mtp_rounds"))
        and isinstance(turn.get("prompt", {}), dict)  # round 2 E1: containers too
    )
    if ok and readout:
        telemetry = turn.get("telemetry")
        ok = isinstance(telemetry, dict) and _exact_int(
            telemetry.get("speculative_argmax_from_hidden")
        )
    if not ok:
        raise MalformedEvidence(f"malformed evidence: E2 {where} turn = {turn!r}")
    return turn


def e2_summary(pair, cross, *, digests):
    """C185: E2 from the same-instance triples (``pair``: name -> the ``e2_pair`` entry, ``turns[0]`` = ON,
    ``off`` = [OFF1, OFF2]) and the cross-load OFF arm (``cross``: name -> the ``e2`` entry). The verdict is
    ON vs OFF1 under the C1-E2 control; ``cross_load`` (OFF arm vs OFF1) is reported and never decides. The
    four trajectories of a request share one templated prompt; C84 items are bound to the reconstruction.
    """

    def bad(what, value):
        raise MalformedEvidence(f"malformed evidence: E2 {what} = {value!r}")

    if not isinstance(pair, dict) or not pair:
        bad("requests", pair)
    if not isinstance(cross, dict) or not cross:  # round 3 E1
        bad("cross-load requests", cross)
    if set(cross) != set(pair):  # round 4 E9: every cross-load result has its triple
        bad("cross-load request names", sorted(set(cross) ^ set(pair)))

    def first_turn(where, entry):  # round 2 E1: an entry with a non-empty turn list
        turns = entry.get("turns") if isinstance(entry, dict) else None
        if not isinstance(turns, list) or not turns:
            bad(f"{where} entry", entry)
        return turns[0]

    # round 2 E5: the generation OUTCOME — token ids AND finish reason
    outcome = lambda t: (
        t["final"]["token_ids"],
        t["final"]["finish_reason"],
    )  # noqa: E731
    identity, c1, cross_per, engaged, problems = {}, {}, {}, {}, []
    for n, entry in pair.items():
        on = first_turn(n, entry)
        offs = entry.get("off")
        if not isinstance(offs, list) or len(offs) != 2:
            bad(f"{n}.off", offs)
        if n not in cross:
            bad(f"{n} cross-load entry", None)
        (off1, off2), far = offs, first_turn(f"{n} cross-load", cross[n])
        e2_turn_checked(f"{n} ON", on, readout=True)
        for label, turn in (("OFF1", off1), ("OFF2", off2), ("cross-load", far)):
            e2_turn_checked(f"{n} {label}", turn)
        verdict = entry.get("verdict")
        kernel = exact_bool(
            verdict.get("pass") if isinstance(verdict, dict) else verdict,
            f"E2 {n}.verdict.pass",
        )
        c1[n] = outcome(off1) == outcome(off2)
        identity[n] = outcome(on) == outcome(off1)
        cross_per[n] = outcome(far) == outcome(off1)
        engaged[n] = (
            on["telemetry"]["speculative_argmax_from_hidden"] > 0
            and kernel
            and on["mtp_rounds"] > 0
        )
        shas = [(t.get("prompt") or {}).get("sha256") for t in (on, off1, off2, far)]
        if not all(
            is_sha256(x) for x in shas
        ):  # round 1 E2: None == None binds nothing
            problems.append(f"{n}: a trajectory lacks a valid prompt digest: {shas}")
        elif len(set(shas)) != 1:
            problems.append(f"{n}: the trajectories' prompt digests differ: {shas}")
    problems += prompt_digest_problems(digests, pair)
    problems += prompt_digest_problems(digests, cross)
    all_identical = all(identity.values())
    c1_identical = all(c1.values())
    argmax_ok = all(engaged.values())
    return dict(
        identity=identity,
        all_identical=all_identical,
        c1=c1,
        c1_identical=c1_identical,
        cross_load=dict(
            per_request=cross_per,
            summary="identical" if all(cross_per.values()) else "unestablished",
        ),
        argmax_and_kernels=argmax_ok,
        prompt_digest_problems=problems,
        verdict=e2_verdict(
            c1_identical=c1_identical,
            engaged=argmax_ok and not problems,
            all_identical=all_identical,
        ),
    )


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
        kernel_ok = exact_bool(  # code round 6 H34
            (entry.get("verdict") or {}).get("pass"), f"{name}.verdict.pass"
        )
        turn = entry["turns"][0]
        first = _first_terminal(turn)
        if turn.get("mtp_rounds", 0) <= 0 and not first:
            not_engaged.append(name)
            continue
        digest_bad = digests is not None and (turn.get("prompt") or {}).get(
            "sha256"
        ) != digests.get(name)
        kernel_bad = not first and not kernel_ok
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
    kernel_ok = exact_bool(  # code round 6 H34
        (entry.get("verdict") or {}).get("pass"), f"{name}.verdict.pass"
    )
    follow_ups = []  # the worker writes a continuity record for every e7b turn > 0
    for k, turn in enumerate(turns[1:], 1):
        record = turn.get("continuity")
        what = f"{name} turn {k} continuity"
        if not isinstance(record, dict):
            raise MalformedEvidence(f"malformed evidence: {what} = {record!r}")
        follow_ups.append((exact_bool(record.get("continuity"), what), record))
    reasons = []
    if (turns[0].get("prompt") or {}).get("sha256") != digest:
        reasons.append("the first prompt's digest differs from the reconstruction")
    if not kernel_ok:
        reasons.append("the kernel verdict failed")
    reasons += [
        f"turn {t} ran no MTP rounds"
        for t, turn in enumerate(turns)
        if turn.get("mtp_rounds", 0) <= 0 and not _first_terminal(turn)
    ]
    reasons += [
        record.get("reason") or "no continuity reason"
        for ok, record in follow_ups
        if not ok
    ]
    return dict(
        name=name,
        continuity=all(ok for ok, _ in follow_ups),
        censored=bool(reasons),
        reason=reasons[0] if reasons else None,
        events=[ev["cls"] for t in turns for ev in t["events"]],
        mtp_rounds=[t.get("mtp_rounds") for t in turns],
    )


def e7b_status(sessions):
    return (
        "OBSERVED"
        if any(not exact_bool(s["censored"], "e7b.censored") for s in sessions)
        else "INCOMPLETE"
    )


E7B_TIMEOUT_KEYS = {"name", "censored", "reason"}
E7B_SESSION_KEYS = {"name", "continuity", "censored", "reason", "events", "mtp_rounds"}


def e7b_sessions_checked(pick, e7b):
    """Code round 6 H38: an E7b report is a list of sessions, each EXACTLY a timeout censor
    ``{name, censored: True, reason: "timeout"}`` or an ``e7b_session`` record with typed fields (the
    censored flag agreeing with the reason), and its status re-derived by ``e7b_status``; else RAISES.
    """

    def bad(what, value):
        raise MalformedEvidence(f"malformed evidence: {pick} e7b {what} = {value!r}")

    if not isinstance(e7b, dict) or not isinstance(e7b.get("sessions"), list):
        bad("report", e7b)
    for s in e7b["sessions"]:
        keys = set(s) if isinstance(s, dict) else None
        if keys == E7B_TIMEOUT_KEYS:
            ok = (
                type(s["name"]) is str
                and s["censored"] is True
                and s["reason"] == "timeout"
            )
        elif keys == E7B_SESSION_KEYS:
            ok = (
                type(s["name"]) is str
                and type(s["continuity"]) is bool
                and type(s["censored"]) is bool
                and (s["reason"] is None or type(s["reason"]) is str)
                and s["censored"] is (s["reason"] is not None)
                and type(s["events"]) is list
                and all(type(e) is str for e in s["events"])
                and type(s["mtp_rounds"]) is list
                and all(type(r) is int for r in s["mtp_rounds"])
            )
        else:
            ok = False
        if not ok:
            bad("session", s)
    if e7b.get("status") != e7b_status(e7b["sessions"]):
        bad("status", e7b.get("status"))
    return e7b["sessions"]


def kv_positive_kind(census, where):
    """Smoke 9 / P405: the KV positive the live leaves require, from the E1c entry's leaf census (never the registry
    label — the server may build a different codec than the registry names)."""
    if not isinstance(census, dict) or not census:
        raise MalformedEvidence(f"malformed evidence: {where}.leaf_census = {census!r}")
    names = list(census)
    if any(n.endswith("TurboQuantKVCache") for n in names):
        return "tq_norm"
    if any(n.endswith("QuantizedKVCache") for n in names):
        return "quantized_scale"
    if any(n.endswith("KVCache") for n in names):
        return "old_prefix_k"
    raise MalformedEvidence(
        f"malformed evidence: {where}.leaf_census has no KV leaf: {census!r}"
    )


def leaf_manifest_problems(pick, name, cell, kv):
    """P405 F3: the positive cell's live cache matches, layer path by layer path, what its argv's KV codec implies."""
    where = f"{pick} {name}"
    lm = cell.get("leaf_manifest")
    if not (
        isinstance(lm, dict)
        and isinstance(lm.get("observed"), list)
        and isinstance(lm.get("template"), list)
        and lm["template"]
    ):
        raise MalformedEvidence(f"malformed evidence: {where}.leaf_manifest = {lm!r}")
    if not isinstance(kv, dict) or set(kv) != set(kvmanifest.KV_FIELDS) | {
        "tq_kv_quant_mode"
    }:
        raise MalformedEvidence(f"malformed evidence: {pick}.kv = {kv!r}")
    try:
        found = kvmanifest.manifest_problems(lm["observed"], lm["template"], kv)
    except ValueError as exc:  # a codec the manifest does not model: refused
        found = [str(exc)]
    dm = cell.get("drafter_manifest")  # code round 1 K3
    if not (
        isinstance(dm, dict)
        and isinstance(dm.get("observed"), list)
        and isinstance(dm.get("template"), list)
        and dm["template"]
    ):
        raise MalformedEvidence(
            f"malformed evidence: {where}.drafter_manifest = {dm!r}"
        )
    found += kvmanifest.drafter_problems(dm["observed"], dm["template"])
    return [f"{where}: leaf manifest {p}" for p in found]


def smoke_gate(picks):
    """Every live positive fired (TurboQuant's required where TurboQuant KV is served), every E1c cell
    passed, E2 passed, exactly the first reconstructed C84 item's E7b session ran and reported (its
    continuity is descriptive; code round 6 H38), and the projected full run fits ESTIMATE_LIMIT_S. Pure
    (code round 6 H36): ``main`` recomputes it from a loaded smoke summary; malformed evidence RAISES.
    """
    problems, e7b_status_of = [], {}
    projected = 0
    if not picks:
        problems.append("no pick ran")
    for pick, record in picks.items():
        if record["outcome"]["e1c"] != "PASS":
            problems.append(f"{pick}: E1c {record['outcome']}")
        for name, v in (record.get("e1c") or {}).items():
            if not exact_bool(v.get("pass_"), f"{pick} {name}.pass_"):
                problems.append(f"{pick} {name}: {v.get('problems')}")
            sp = v.get("smoke_positives")
            if sp is not None:
                where = f"{pick} {name}.smoke_positives"
                fired = sp.get("fired")
                if not isinstance(fired, dict):
                    raise MalformedEvidence(
                        f"malformed evidence: {where}.fired = {fired!r}"
                    )
                baseline = exact_bool(sp.get("baseline_pass"), f"{where}.baseline_pass")
                seen = {k for k, ok in fired.items() if exact_bool(ok, f"{where}.{k}")}
                need = {"raw_ulp", "in_support", "recurrent", "probe_logit"}
                need.add(kv_positive_kind(v.get("leaf_census"), f"{pick} {name}"))
                if not baseline or not need <= seen:
                    problems.append(f"{pick}: smoke positives {sp}")
                problems += leaf_manifest_problems(pick, name, v, record.get("kv"))
                if needs_draft_inventory(pick):  # v13 V15 / code round 1 M2
                    problems += draft_inventory_problems(
                        pick, name, v.get("draft_kernels")
                    )
        # code round 2 C2: exactly the scheduled cells, the positive block on the last one only
        cells = record.get("e1c") or {}
        scheduled = [f"T{t}" for t in SMOKE_THRESHOLDS]
        if sorted(cells) != sorted(scheduled):
            problems.append(f"{pick}: E1c smoke cells {sorted(cells)} != {scheduled}")
        for name in scheduled:
            has = (cells.get(name) or {}).get("smoke_positives") is not None
            if has != (name == scheduled[-1]):
                problems.append(f"{pick} {name}: smoke positives present={has}")
        e2 = record.get("e2")
        if e2 is None or e2.get("verdict") != "PASS":
            problems.append(f"{pick}: E2 smoke {e2}")
        e7b = record.get("e7b")
        if e7b is None:
            problems.append(f"{pick}: E7b smoke session did not run")
        else:
            names = [s["name"] for s in e7b_sessions_checked(pick, e7b)]
            expected = next(iter(record.get("reconstruction") or {}), None)
            if expected is None or names != [expected]:
                problems.append(
                    f"{pick}: E7b smoke ran {names}, expected exactly [{expected!r}]"
                )
            e7b_status_of[pick] = e7b["status"]
        wall = record.get("wall_s")
        if type(wall) not in (int, float):
            raise MalformedEvidence(f"malformed evidence: {pick}.wall_s = {wall!r}")
        steps = record.get("step_s")
        if not (
            isinstance(steps, dict)
            and set(steps) >= {"prep", "e1c", "e2", "e7b"}
            and all(  # code round 1 M10: finite and non-negative
                type(steps[k]) in (int, float)
                and math.isfinite(steps[k])
                and steps[k] >= 0
                for k in ("prep", "e1c", "e2", "e7b")
            )
        ):
            raise MalformedEvidence(f"malformed evidence: {pick}.step_s = {steps!r}")
        projected += phase_projection(steps)["nominal_s"]  # v13 V13 (was wall × 6)
    return dict(
        problems=problems,
        projected_full_s=projected,
        projected_with_tail_s=projected + TAIL_RESERVE_S * len(picks),
        fits=projected <= ESTIMATE_LIMIT_S,
        pass_=not problems,
        e7b=e7b_status_of,
    )


def smoke_authorization_problems(summary, projection_ruling=None):
    """Code round 6 H36: a full run is authorized only by a smoke summary (as loaded from its JSON) whose
    phase is smoke, status ``SMOKE_PASS`` with exit code 0, whose stored gate holds exact ``True`` for
    ``pass_`` and ``fits``, and which equals the gate recomputed from the summary's own evidence.
    C193: with a ``projection_ruling`` from ``PROJECTION_RULINGS`` whose pick is the summary's only pick, a
    ``SMOKE_NOT_PASS`` / exit 2 smoke whose gate PASSED but does not fit authorizes too — nothing else is lifted.
    """
    if not isinstance(summary, dict):
        return ["the smoke summary is not an object"]
    problems = []
    if summary.get("phase") != "smoke":
        problems.append(f"phase {summary.get('phase')!r} is not smoke")
    code = summary.get("exit_code")
    ruled = None
    if projection_ruling is not None:
        ruled = (PROJECTION_RULINGS.get(projection_ruling) or {}).get("pick")
        if ruled is None:
            return problems + [f"unknown projection ruling {projection_ruling!r}"]
        if list(summary.get("picks") or {}) != [ruled]:
            return problems + [
                f"ruling {projection_ruling} covers {ruled} only, not {list(summary.get('picks') or {})}"
            ]
    want = ("SMOKE_NOT_PASS", 2) if ruled else ("SMOKE_PASS", 0)
    if (summary.get("status"), code) != want or type(code) is not int:
        problems.append(f"status {summary.get('status')!r} / exit code {code!r}")
    gate = summary.get("smoke_gate")
    if not isinstance(gate, dict):
        return problems + ["the smoke summary carries no smoke gate"]
    try:
        passed = exact_bool(gate.get("pass_"), "smoke_gate.pass_")
        fits = exact_bool(gate.get("fits"), "smoke_gate.fits")
        if not (passed and (fits or ruled)):
            problems.append(f"the smoke gate did not pass / fit: {gate}")
        if (
            ruled and fits
        ):  # code round 2 K8: the ruling lifts `fits` only when it is false
            problems.append(
                f"ruling {projection_ruling} covers a smoke that does not fit; this one fits"
            )
        again = smoke_gate(summary.get("picks") or {})
        if again != gate:
            problems.append(
                f"the stored smoke gate {gate} != its recomputation {again}"
            )
    except MalformedEvidence as exc:
        problems.append(str(exc))
    except (KeyError, TypeError, AttributeError) as exc:
        problems.append(
            f"malformed evidence: the smoke summary ({type(exc).__name__}: {exc})"
        )
    return problems


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
    """The phase status and the process exit code (code rounds 1 H10, 2 H10; P394 (2) refined): 0 = PASS
    (full: every pick's E1c PASS, E2 PASS and E7a OBSERVED; E7b — scripted chat continuity, descriptive by
    design — is reported and never gates), 2 = completed without PASS (a verdict for the operator); harness
    errors record ``HARNESS_ERROR`` and exit 1."""
    if summary.get("phase") == "smoke":
        gate = summary.get("smoke_gate") or {}
        passed = exact_bool(gate.get("pass_"), "smoke_gate.pass_")  # code round 6 H34
        fits = exact_bool(gate.get("fits"), "smoke_gate.fits")
        ok = passed and fits
        return ("SMOKE_PASS", 0) if ok else ("SMOKE_NOT_PASS", 2)
    picks = summary.get("picks") or {}
    ok = bool(picks) and all(
        (r.get("outcome") or {}).get("e1c") == "PASS"
        and (r.get("e2") or {}).get("verdict") == "PASS"
        and (r.get("e7a") or {}).get("status") == "OBSERVED"
        for r in picks.values()
    )
    return ("PASS", 0) if ok else ("NOT_PASS", 2)


def e1c_verdict(
    on_entry, off_entry, *, threshold, verify_policy, offset=STRADDLE_OFFSETS[0]
):
    """Per-request E1c rule (plan §2.2; code round 2 H15): NEUTRALITY = rows and states equal; EVIDENCE =
    kernels, engagement, the exact prompt and the straddle. ``pass_`` needs both; ``pick_outcome`` reports
    a neutrality failure as FAIL and an evidence-only failure as INCOMPLETE. Malformed comparator / kernel
    evidence (a missing or non-boolean ``pass``) RAISES — a harness defect, never an MTP finding (code round
    5 H34)."""

    def verdict_of(entry, key):
        value = (entry.get(key) or {}).get("pass") if isinstance(entry, dict) else None
        return exact_bool(value, f"comparator {key}.pass")

    rows, states = verdict_of(off_entry, "rows"), verdict_of(off_entry, "states")
    forced = verdict_of(off_entry, "forced")  # round 4 E8
    kernels = verdict_of(on_entry, "verdict")
    neutrality, evidence = [], []
    if not rows:
        neutrality.append("rows")
    if not states:
        neutrality.append("states")
    if not forced:
        neutrality.append("forced schedule")
    if not kernels:
        evidence.append("kernels")
    if not on_entry.get("draft_rounds"):
        evidence.append("no MTP rounds")
    if threshold is not None:
        if on_entry.get("prompt_end") != straddle_target(threshold, offset):
            evidence.append(
                f"prompt_end {on_entry.get('prompt_end')} != {straddle_target(threshold, offset)}"
            )
        if verify_policy == "joint_v1":
            timings = (on_entry.get("response") or {}).get("timings") or {}
            if not timings.get("verify_blocks_straddle"):
                evidence.append("no straddle counter")
            if timings.get("verify_fallback_reasons"):
                evidence.append(
                    f"fallback reasons {timings['verify_fallback_reasons']}"
                )
        elif not exact_bool(on_entry.get("straddle"), "straddle"):  # code round 6 H34
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


LAUNCHER_MODULES = (
    "bench.p426_leg",
    "bench.p426_wave1",
)  # P452: the only refused ancestors allowed


def foreign_process_problems(procs, me, ancestors):
    """P452: refused processes beside the run — never the caller, and never an ANCESTOR that is the P426 leg wrapper or
    sequencer that launched it (code round 1 F2: any other refused ancestor, e.g. a benchmark parent, still refuses).
    """
    problems = []
    for pid, argv in procs:
        if pid == me:
            continue
        why = refused_process(argv or [])
        if (
            why
            and pid in ancestors
            and why in tuple(f"module {m}" for m in LAUNCHER_MODULES)
        ):
            continue
        if why:
            problems.append(f"process {pid} ({why}): {' '.join(argv or [])[:120]}")
    return problems


def resolve_production_ref(fork_root, ref, reviewed_head):
    """P452 code round 1 F3: the production ref, resolved ONCE to a full commit sha that is a STRICT ancestor of the
    reviewed head (never the head itself — that would make the diff tautological); the same sha is diffed and recorded.
    """

    def git(*a):
        return subprocess.run(
            ["git", "-C", str(fork_root), *a], capture_output=True, text=True
        )

    sha = git("rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()
    head = git("rev-parse", "--verify", f"{reviewed_head}^{{commit}}").stdout.strip()
    if not sha or not head:
        raise ValueError(
            f"production ref {ref!r} / reviewed head {reviewed_head!r} is not a commit"
        )
    if sha == head or git("merge-base", "--is-ancestor", sha, head).returncode != 0:
        raise ValueError(
            f"production ref {sha[:12]} is not a strict ancestor of the reviewed head {head[:12]}"
        )
    return sha


def ancestor_pids():
    import psutil

    return {p.pid for p in psutil.Process(os.getpid()).parents()}


def production_diff_problem(fork_root, ref):
    """P452: the production tree (``mlx_vlm`` minus its tests) must equal ``ref``; None when it does."""
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(fork_root),
            "diff",
            "--stat",
            ref,
            "--",
            "mlx_vlm",
            ":!mlx_vlm/tests",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"production code differs from {ref}:\n{diff}" if diff else None


def preflight(args, out):
    import psutil

    me = os.getpid()
    ancestors = ancestor_pids()
    problems = foreign_process_problems(
        [
            (p.info["pid"], p.info["cmdline"])
            for p in psutil.process_iter(["pid", "cmdline"])
        ],
        me,
        ancestors,
    )
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
    diff = production_diff_problem(
        args.fork_root, getattr(args, "production_ref", None) or PRODUCTION
    )
    if diff:
        problems.append(diff)
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


class Refused(RuntimeError):
    """A pre-job refusal inside the campaign (the smoke does not bind the full run)."""


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


@contextmanager
def deferred_signals(signals=None):
    """Code round 4 H27: TERM / INT arriving while a worker is spawned and its cleanup ownership is published
    are RECORDED (``pending``), not raised — Python runs handlers in the main thread only, so no other
    thread can raise here either. No signal mask is touched, so the worker inherits none; the caller
    replays ``pending`` once the worker is owned."""
    import signal

    signals = signals or (signal.SIGTERM, signal.SIGINT)
    pending = []
    old = {
        s: signal.signal(s, lambda signum, frame: pending.append(signum))
        for s in signals
    }
    try:
        yield pending
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


PS_ENV = {"LC_ALL": "C", "LANG": "C"}
PS_MARGIN_S = 5.0  # how far a bounded cleanup's last ps call may run past its deadline


def ps_table(*, retries=3, backoff_s=0.2, timeout_s=10.0, deadline=None):
    """Code rounds 4 H28, 5 H33: ``{pid: {ppid, pgid, stat, start}}`` of every process (``ps``,
    ``LC_ALL=C`` so ``lstart`` is "Mon Oct 19 14:03:12 2026"). A failure, a timeout, an unparseable line or a
    table without this process RAISES after ``retries`` or at ``deadline`` (monotonic; each attempt's
    timeout and backoff are capped to the time left) — process state is never assumed empty.
    """
    last = None
    for _ in range(retries):
        left = None if deadline is None else deadline - time.monotonic()
        if left is not None and left <= 0:
            break
        try:
            out = subprocess.run(
                ["ps", "-A", "-o", "pid=,ppid=,pgid=,stat=,lstart="],
                capture_output=True,
                text=True,
                timeout=timeout_s if left is None else max(0.01, min(timeout_s, left)),
                check=True,
                env=dict(os.environ, **PS_ENV),
            ).stdout
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            OSError,
        ) as exc:
            last = exc
            left = None if deadline is None else deadline - time.monotonic()
            time.sleep(backoff_s if left is None else max(0.0, min(backoff_s, left)))
            continue
        table = {}
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split(None, 4)
            if len(parts) != 5 or not all(p.isdigit() for p in parts[:3]):
                raise RuntimeError(f"unparseable ps line: {line!r}")
            table[int(parts[0])] = dict(
                ppid=int(parts[1]),
                pgid=int(parts[2]),
                stat=parts[3],
                start=" ".join(parts[4].split()),
            )
        if os.getpid() not in table:
            raise RuntimeError("the ps table lacks this process (partial output)")
        return table
    raise RuntimeError(f"ps failed {retries} times: {last}")


class Descendants:
    """Code rounds 3 H22, 4 H30: the identities (pid + start time) of a worker and of every process
    descended from it, scanned at each supervisor poll, so a stop or a normal exit also reaches one that
    left the worker's group (``setsid``). An identity counts only while its recorded start time still
    matches: a stale (reused) pid is never a traversal frontier and never signalled. The worker's process
    group (pgid = its pid) is used only while that pid is still the worker or absent. Residual (disclosed):
    a helper that both spawns and escapes between two polls; pid reuse within one ``lstart`` second.
    """

    def __init__(self, root, *, table=None):
        table = ps_table() if table is None else table
        row = table.get(root)
        if row is None:
            raise RuntimeError(f"worker {root} is not in the process table")
        self.root, self.root_start, self.seen = root, row["start"], {}

    def _ours(self, pid, table):
        row = table.get(pid)
        if row is None:
            return False
        want = self.root_start if pid == self.root else self.seen.get(pid)
        return row["start"] == want

    def root_state(self, table, *, reaped=False):
        """``alive`` | ``exited`` (an unreaped zombie: its pid, hence the group id, stays anchored — code
        round 5 H30) | ``reaped`` (we waited on it). Any other state is impossible for our own unreaped
        child and RAISES."""
        row = table.get(self.root)
        if row is None:
            if reaped:
                return "reaped"
            raise RuntimeError(
                f"worker {self.root} vanished from the process table unreaped"
            )
        if row["start"] != self.root_start:
            if reaped:
                return "reaped"  # the pid was reused after we reaped it
            raise RuntimeError(f"worker {self.root} changed identity while unreaped")
        return "exited" if row["stat"].startswith("Z") else "alive"

    def scan(self, table=None, *, deadline=None):
        table = ps_table(deadline=deadline) if table is None else table
        frontier = {pid for pid in [self.root, *self.seen] if self._ours(pid, table)}
        grew = True
        while grew:
            grew = False
            for pid, row in table.items():
                if row["ppid"] in frontier and pid not in frontier:
                    frontier.add(pid)
                    self.seen[pid] = row["start"]
                    grew = True
        return table

    def alive(self, table):
        """Tracked identities still alive (non-zombie)."""
        return sorted(
            pid
            for pid in self.seen
            if self._ours(pid, table) and not table[pid]["stat"].startswith("Z")
        )

    def group_members(self, table):
        """Live members of the worker's process group — refused once the worker's pid belongs to another
        process (the cached group id is no longer ours)."""
        row = table.get(self.root)
        if row is not None and row["start"] != self.root_start:
            raise RuntimeError(
                f"process group {self.root} is no longer ours (pid reused)"
            )
        return sorted(
            pid
            for pid, r in table.items()
            if r["pgid"] == self.root and not r["stat"].startswith("Z")
        )


def _owned_live(proc, tracker, table):
    """Every live owned process: tracked identities; and, while the worker is UNREAPED (our zombie or alive
    child anchors its pid and group id), its group members and the worker itself. After we reaped it the
    numeric group id is never trusted again."""
    live = set(tracker.alive(table))
    if proc.returncode is None:
        live |= set(tracker.group_members(table))
        if tracker.root_state(table) == "alive":
            live.add(proc.pid)
    return live


def _signal_each(pids, sig):
    for pid in sorted(pids):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _kill_unverified(proc):
    """No verifiable process state (ps failing): SIGKILL the group ``launch_worker`` created — safe while the
    worker is unreaped (its pid, hence the group id, cannot be reused) — and the worker. The caller then
    fails closed: extinction is NOT established."""
    import signal

    if proc.returncode is None:
        for kill in (lambda: os.killpg(proc.pid, signal.SIGKILL), proc.kill):
            try:
                kill()
            except (ProcessLookupError, PermissionError):
                pass


def _stop(proc, grace_s, tracker=None):
    """SIGTERM to every live owned process, up to ``grace_s`` for the worker to exit (observed, never
    reaped), then ``reap``. Without verifiable process state the new group is SIGKILLed and the failure
    RAISES (code round 5 H31: never a silent clean exit)."""
    import signal

    try:
        if tracker is None:
            tracker = Descendants(proc.pid)  # our unreaped child is always in the table
        _signal_each(_owned_live(proc, tracker, tracker.scan()), signal.SIGTERM)
        until = time.monotonic() + grace_s
        while time.monotonic() < until:
            # P396: the scan keeps PS_MARGIN_S past the grace end (as reap's do) — bounded by ``until`` alone,
            # the last one ran ps with the few ms left and a TERM-ignoring worker became a harness error
            table = tracker.scan(deadline=until + PS_MARGIN_S)
            if proc.returncode is not None or tracker.root_state(table) != "alive":
                break
            time.sleep(0.05)
    except RuntimeError:
        _kill_unverified(proc)
        raise
    reap(proc, tracker)


def stop_worker(proc, tracker, *, grace_s):
    """Code round 5 H32: ``_stop`` that a further TERM / INT cannot interrupt — signals arriving during the
    cleanup are deferred and replayed only once extinction is established."""
    with deferred_signals() as again:
        _stop(proc, grace_s, tracker)
    if again:
        raise SystemExit(128 + again[0])


def reap(proc, tracker, *, wait_s=10.0):
    """Code rounds 2 H17, 3 H22, 4 H29, 5 H30 / H33: SIGKILL the worker, every live member of its group and
    every tracked identity, re-scanning and re-signalling until NONE is alive; only then is the worker reaped
    (until then its zombie anchors the group id). Anything still alive at ``wait_s`` RAISES — the next launch
    never shares the GPU with it; each ps call is bounded by the same deadline plus ``PS_MARGIN_S``.
    """
    import signal

    deadline = time.monotonic() + wait_s
    live = None
    while True:
        if live and time.monotonic() >= deadline:
            raise RuntimeError(f"worker processes survived {wait_s} s: {sorted(live)}")
        try:
            table = tracker.scan(deadline=deadline + PS_MARGIN_S)
            live = _owned_live(proc, tracker, table)
        except RuntimeError:
            _kill_unverified(proc)
            raise
        if not live:
            proc.wait()
            return
        _signal_each(live, signal.SIGKILL)
        time.sleep(0.05)


def supervise(
    proc,
    job_dir,
    *,
    boot_timeout_s,
    deadline,
    job_deadline=None,
    poll_s=2.0,
    grace_s=30.0,
    tracker=None,
):
    """Watch one worker until it exits; stop it (SIGTERM, then SIGKILL after ``grace_s``) when its current
    request (``progress.json``) overruns its deadline, when no heartbeat came within ``boot_timeout_s``, at
    the job deadline or at the campaign ``deadline`` (monotonic). Every stop writes ``timeout.json`` first.
    A request whose entry (``req-N.json``) is already durable is a ``hang_after``, never censored (H20).
    ``tracker`` (``Descendants``) is scanned at every poll and killed on a stop (H22).
    """
    job_dir = Path(job_dir)
    started = time.monotonic()
    if tracker is None:
        # our unreaped child is always in the table: a failure here is unknown process state, which
        # RAISES (code round 5 H31) — never an untracked "exited"
        tracker = Descendants(proc.pid)
    proc.p369_tracker = tracker  # the caller reaps with the same identities
    while True:
        table = tracker.scan()
        if tracker.root_state(table) == "exited":
            # observed, NOT reaped (H30): the exit code comes from reap once extinction is proven
            return dict(outcome="exited")
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
            if phase == "request" and _read_json(
                job_dir / f"req-{progress['index']}.json"
            ):
                phase = "after"  # the entry landed before the heartbeat moved on (H20)
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
            stop_worker(proc, tracker, grace_s=grace_s)  # H32: uninterruptible
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
        proc = tracker = None
        try:
            # H27: TERM / INT during the spawn and the tracker's creation are deferred, then replayed
            # below with the worker owned; a signal after the deferral is inside this try as well
            with deferred_signals() as pending:
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
                tracker = Descendants(proc.pid)  # unknown state RAISES (H31)
            if pending:
                raise SystemExit(128 + pending[0])
            outcome = supervise(
                proc,
                job_dir,
                boot_timeout_s=BOOT_TIMEOUT_S,
                deadline=deadline,
                job_deadline=time.monotonic() + job["timeout_s"],
                tracker=tracker,
            )
            if outcome["outcome"] == "exited":
                # H17 / H22 / H29 / H32: nothing the worker spawned outlives it; uninterruptible
                with deferred_signals() as again:
                    reap(proc, tracker)
                if again:
                    raise SystemExit(128 + again[0])
                outcome["returncode"] = proc.returncode
        except BaseException:
            if proc is not None:
                with (
                    deferred_signals()
                ):  # H32: the primary interruption / error propagates after
                    _stop(proc, 5, tracker)
            raise
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
TAIL_RESERVE_S = 7200  # v13 V13: one full-budget E7 runaway, REPORTED (C194: never enforced as a cap)


def phase_projection(step_s):
    """P369 v13 V13: the full run per phase from the smoke's measured steps × the full cardinalities — prep (sizing
    four thresholds, reconstructing 20 items) × 1.5; E1c × the threshold-token ratio (+ the short cell, negligible);
    E2 × 10 items + its four straddles at the E1c rate; E7a = 20 requests at E7b's per-turn time (3 turns); E7b × 5.
    Retries are not projected: a cell's further seeds and P537's further offsets (up to 15 attempts and two re-sizes
    per thresholded cell) can take a full run past this projection (no wall cap is enforced, C184 / C194).
    """
    e1c = step_s["e1c"] * sum(THRESHOLDS) / sum(SMOKE_THRESHOLDS)
    nominal = (
        step_s["prep"] * 1.5
        + e1c
        + step_s["e2"] * 10
        + e1c
        + step_s["e7b"] * (20 / 3 + 5)
    )
    return dict(
        nominal_s=nominal,
        tail_reserve_s=TAIL_RESERVE_S,
        with_tail_s=nominal + TAIL_RESERVE_S,
    )


# C193 (operator 2026-10-20): recorded rulings that lift ONLY the smoke gate's `fits` for one pick's full run; the
# P369 v13 phase-based projection replaces this in the v13 build. C194 ("no hard stop"): a ruling brings NO campaign
# wall cap — C184 stands; the per-request watchdogs remain the only automatic stops.
PROJECTION_RULINGS = {"C193": dict(pick="Qwen3.8-27B-mlx-uniform-4bit")}


def positive_seconds(raw):
    """``--wall-limit-s``: finite seconds > 0 (P405 code round 3 K11: nan made the stop a silent no-op)."""
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"expected finite seconds > 0, got {raw!r}")
    return value


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


SNAPSHOT_SKIP = (
    "README.md",
    ".gitattributes",
)  # documentation only (v13 V11: verified on the Hub)


def snapshot_files(snap):
    """P369 v13 V6: every file of a snapshot the loader can read — weights, configs, tokenizer, chat template,
    sidecars in subdirectories — as relative posix paths; only README / .gitattributes / ``evaluation/`` excluded.
    """
    snap = Path(snap)
    out = []
    for p in sorted(snap.rglob("*")):
        rel = p.relative_to(snap).as_posix()
        if not p.is_file() or rel in SNAPSHOT_SKIP or rel.startswith("evaluation/"):
            continue
        out.append(rel)
    return out


def weights_record(hf_path):
    root = hf_hub_cache(
        os.environ, Path.home()
    )  # code round 2 H9: the worker's own resolution
    repo = root / ("models--" + hf_path.replace("/", "--"))
    ref = (repo / "refs" / "main").read_text().strip()
    snap = repo / "snapshots" / ref
    return dict(
        hf_path=hf_path,
        revision=ref,
        files={
            rel: sha256_file((snap / rel).resolve()) for rel in snapshot_files(snap)
        },
    )


PINS = Path(__file__).resolve().parent / "pins.json"


def load_pins(path=PINS):
    """P369 v13 V6 / V11: the reviewed per-entry target / drafter revisions and file sha256s."""
    return json.loads(Path(path).read_text())


def pins_problems(pins, pick, target, drafter):
    """Every loaded snapshot must be EXACTLY the pinned one: revision, the file set and each sha256."""
    if pick not in pins:
        return [f"{pick}: no reviewed pins"]
    problems = []
    for role, got in (("target", target), ("drafter", drafter)):
        want = pins[pick][role]
        if got["hf_path"] != want["hf_path"] or got["revision"] != want["revision"]:
            problems.append(
                f"{pick} {role}: {got['hf_path']}@{got['revision']} != pinned "
                f"{want['hf_path']}@{want['revision']}"
            )
        if got["files"] != want["files"]:
            diff = sorted(
                p
                for p in set(got["files"]) | set(want["files"])
                if got["files"].get(p) != want["files"].get(p)
            )
            problems.append(f"{pick} {role}: files differ from the pins: {diff}")
    return problems


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
    import hashlib
    corpus_sha = hashlib.sha256(json.dumps(sorted(corpus.items()), sort_keys=True, default=str).encode()).hexdigest()
    for row in rows:
        item = corpus.get(str(row["id"]))
        if item is None:
            raise SystemExit(f"{axis}/{row['id']}: not in the corpus")
        import hashlib
        content = hashlib.sha256(json.dumps(item, sort_keys=True, default=str).encode()).hexdigest()
        items.append(dict(name=f"{axis}/{row['id']}", messages=benchmarks.build_messages(axis, item),
                          seed=row.get("sampler_seed"), prompt_tokens=row.get("prompt_tokens"),
                          item_sha256=content, corpus_sha256=corpus_sha))
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
            *items_source(
                pick
            ),  # v13 V4: a borrowed entry reads the source pick's bundles
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
        problems = campaign_pick_problems(args.pick)  # v13 V9
        if problems:
            raise SystemExit(f"refused: {problems}")
        self.picks = list(args.pick)
        # code round 5 H23: construction reads nothing — the registry and params load in prepare(), inside
        # the persistence boundary, once the run directory exists
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
        self._out_created = True  # from here a stop can record its status (H23)
        import yaml

        self.registry = read_registry(
            self.stack / "main_models.yaml"
        )  # code round 1 M1
        entries = {e["name"]: e for e in yaml.safe_load(self.registry)["models"]}
        self.params = {
            p: {
                k: entries[p]["generation_defaults"][k]
                for k in ("temperature", "top_p", "top_k", "min_p")
            }
            for p in self.picks
        }
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
                (self.stack / "benchmark" / "results" / items_source(pick)[0]).glob(
                    f"*.c84bundle-after-{items_source(pick)[1]}-20260914.jsonl"
                )
            )
        }
        self.bundles = bundles  # v13 V4: bound into the borrowed-panel check
        self.bench_paths = ["benchmark/bench", *sorted(bundles)]
        problems = stack_bench_problems(self.stack, self.bench_paths)
        if problems:  # code round 3 H9: the reconstruction imports these files
            raise Refused(f"refused: {problems}")
        manifest = dict(
            production=PRODUCTION,
            fork_head=self.head,
            harness_tree=rev("HEAD:dev/p369_phase2"),
            reviewed_head=self.args.reviewed_head,
            projection_ruling=getattr(self.args, "projection_ruling", None),  # C193
            versions=fork_versions(self.fork),
            env_keys=sorted(self.env),
            env_dropped=dropped_env(dict(os.environ)),
            env_values={k: self.env[k] for k in ENV_RECORDED_VALUES if k in self.env},
            env_digest=env_digest(self.env),
            stack_bench=stack_bench_state(self.stack, bundles),
            picks={},
        )
        self.stack_bench = manifest["stack_bench"]
        # P369 v13 V5: the live registry must be one of the 16 byte states of the pinned fixture (refused otherwise)
        fixture = registry_states.fixture_text(
            self.fork / "mlx_vlm" / "tests" / "fixtures"
        )
        off = registry_states.classify(self.registry, fixture)
        if off is None:
            raise Refused(
                "refused: the live registry is not one of the 16 accepted byte states"
            )
        manifest["registry_off"] = sorted(off)
        problems = registry_off_problems(self.picks, off)  # P527 (stack C252)
        if problems:
            raise Refused(f"refused: {problems}")
        for pick in self.picks:
            text, restored = overlay_registry(self.registry, pick, fixture)
            overlay = self.out / f"overlay-{pick}.yaml"
            overlay.write_text(text)
            on = worker_argv(self.stack, overlay, pick)
            problems = argv_scan_problems(pick, on)  # T15
            if problems:
                raise Refused(f"refused: {problems}")
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
            problems = pins_problems(  # v13 V6 / V11
                load_pins(), pick, entry["target"], entry["drafter"]
            )
            if problems:
                raise Refused(f"refused: {problems}")
            self.attested[pick] = {
                role: dict(revision=entry[role]["revision"], files=entry[role]["files"])
                for role in ("target", "drafter")
            }
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        return manifest

    def job(self, pick, name, mode, arm, requests, **extra):
        attested = self.attested[pick]
        revisions = {
            "--model": attested["target"]
        }  # {revision, files}: re-hashed by the worker
        if arm == "on":
            revisions["--draft-model"] = attested["drafter"]
        expect = dict(
            drafter=arm == "on",
            verify_policy=PICKS[pick]["verify_policy"] if arm == "on" else None,
            revisions=revisions,  # code round 2 H9: what the worker must actually load
            kv=kvmanifest.kv_from_argv(self.arms[pick][arm]),  # P405 F3
            drafter_shape=PICKS[pick]["drafter"] if arm == "on" else None,  # v13 V11
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
    def size(self, pick, thresholds, offset=STRADDLE_OFFSETS[0], strict=True):
        """Exact prompts ending ``offset`` tokens below each threshold; ``strict=False`` (a P537 re-size) leaves a
        threshold without an exact prompt out instead of raising."""
        base = dict(model=pick, max_tokens=1, stream=False)
        requests = [
            dict(
                name=f"T{t}",
                target=straddle_target(t, offset),
                prefix=SIZE_PREFIX,
                suffix=SIZE_SUFFIX,
                payload=base,
                timeout_s=600,
            )
            for t in thresholds
        ]
        name = "size" if offset == STRADDLE_OFFSETS[0] else f"size-o{offset}"
        result = self.run(self.job(pick, name, "size", "off", requests))
        prompts = {}
        for r, t in zip(result["requests"], thresholds):
            if r["text"] is None:
                if not strict:
                    continue
                raise RuntimeError(
                    f"{pick}: no exact prompt for T={t} (closest {r['tokens']})"
                )
            prompts[t] = r["text"]
        return prompts

    def e1c(self, pick, prompts, *, smoke):
        cells = [(t, text) for t, text in prompts.items()] + (
            [] if smoke else [(None, SHORT_PROMPT)]
        )
        accepted, attempts, size_failures = {}, {}, {}
        pending, parked = list(cells), []
        # P537 (stack C252): an invalid straddle cell takes the next seed; a cell invalid on every seed is
        # re-sized to the next pre-registered offset and takes the seeds again (the short cell never retries).
        # A cell with no exact prompt at an offset is parked for the next one (code review round 1).
        schedule = [(offset, seed) for offset in STRADDLE_OFFSETS for seed in SEEDS]
        attempt = 0
        for offset, seed in schedule:
            if offset != STRADDLE_OFFSETS[0] and seed == SEEDS[0]:
                waiting = [t for t, _ in pending + parked if t is not None]
                sized = (
                    self.size(pick, waiting, offset=offset, strict=False)
                    if waiting
                    else {}
                )
                pending = [(t, sized[t]) for t in waiting if t in sized]
                parked = [(t, None) for t in waiting if t not in sized]
                for t, _ in parked:
                    size_failures.setdefault(f"T{t}", []).append(offset)
            if not pending:
                if not parked:
                    break
                continue
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
            attempt += 1
            self.run(self.job(pick, name, "e1c_pair", "on", requests))
            still = []
            for i, ((t, text), req) in enumerate(zip(pending, requests)):
                path = self.out / pick / name / f"req-{i}.json"
                entry = json.loads(path.read_text())
                n = len(entry["final"]["token_ids"])
                valid = t is None or (
                    exact_bool(entry.get("straddle"), f"{req['name']}.straddle")
                    and n >= 65
                )
                # P399: every attempt is a same-instance observation, valid or not
                seen = e1c_verdict(
                    entry,
                    entry["pair"],
                    threshold=t,
                    verify_policy=PICKS[pick]["verify_policy"],
                    offset=offset,
                )
                attempts.setdefault(req["name"], []).append(
                    dict(
                        seed=seed,
                        offset=offset,
                        path=str(path),
                        valid=valid,
                        neutral=seen["neutral"],
                        c1=exact_bool(
                            entry["c1"]["acceptance"].get("pass"),
                            f"{req['name']}.c1.acceptance.pass",
                        ),
                        problems=seen["problems"],
                    )
                )
                if valid:
                    accepted[req["name"]] = dict(
                        threshold=t,
                        req=req,
                        dir=path.parent,
                        index=i,
                        offset=offset,
                        text=text,
                    )
                else:
                    still.append((t, text))
            pending = still
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
        if requests:
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
                offset=cell["offset"],
            )
            c1_ok = exact_bool(  # code round 6 H34
                on["c1"]["acceptance"].get("pass"), f"{name}.c1.acceptance.pass"
            )
            c2_rows = exact_bool(c2["rows"].get("pass"), f"{name}.c2.rows.pass")
            c2_states = exact_bool(c2["states"].get("pass"), f"{name}.c2.states.pass")
            c2_forced = exact_bool(
                (c2.get("forced") or {}).get("pass"), f"{name}.c2.forced.pass"
            )
            if not c1_ok:
                v["pass_"] = False
                v["problems"].append(f"C1: {on['c1']['acceptance']['problems']}")
            v.update(
                pair=on["pair"],
                c1=dict(identical=c1_ok, **on["c1"]),
                c2=dict(
                    identical=c2_rows and c2_states and c2_forced,
                    rows=c2["rows"],
                    states=c2["states"],
                ),
                kernels=on["verdict"],
                smoke_positives=on.get("smoke_positives"),
                forced_positions=on.get("forced_positions"),  # P398: not compared
                leaf_census=on.get(
                    "leaf_census"
                ),  # smoke 9: what the hash / positives saw
                leaf_manifest=on.get("leaf_manifest"),  # P405 F3
                drafter_manifest=on.get("drafter_manifest"),  # P405 code round 1 K3
                draft_kernels=on.get("draft_verdict"),  # v13 V15
                attempts=attempts[name],
                rejected=[a for a in attempts[name] if not a["valid"]],
                offset=cell["offset"],
                size_failures=size_failures.get(name, []),
            )
            verdicts[name] = v
        # P537: E2's per-threshold straddle requests use the prompt E1c accepted (code review round 1)
        if not hasattr(self, "accepted_prompts"):
            self.accepted_prompts = {}
        self.accepted_prompts[pick] = {
            cell["threshold"]: cell["text"]
            for cell in accepted.values()
            if cell["threshold"] is not None
        }
        for (
            t,
            _,
        ) in (
            pending + parked
        ):  # round 3 E7: invalid on every seed — the attempts still decide
            name = f"T{t}" if t else "short"
            rejected = list(attempts[name])
            problems = [f"invalid on all {len(rejected)} attempts"] + [
                f"no exact prompt at offset {o}" for o in size_failures.get(name, [])
            ]
            verdicts[name] = dict(
                pass_=False,
                neutral=True,
                evidence_ok=False,
                problems=problems,
                neutrality_problems=[],
                evidence_problems=list(problems),
                size_failures=size_failures.get(name, []),
                c1=dict(identical=True),
                c2=dict(identical=False),
                attempts=rejected,
                rejected=rejected,
            )
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

    def bundles_of(self, pick):
        source, kv = items_source(pick)
        return {
            p: sha
            for p, sha in self.bundles.items()
            if p.startswith(f"benchmark/results/{source}/")
            and p.endswith(f".c84bundle-after-{kv}-20260914.jsonl")
        }

    def make_panel(self, pick):
        """v13 V4 (``--phase panel``): the borrowed entry's panel — the source pick's reconstructed items counted in
        the entry's OFF worker (no generation); written for review, used only once committed to ``dev/``.
        """
        if "items_from" not in PICKS[pick]:
            raise Refused(f"refused: {pick} has its own C84 bundles; no panel")
        items = c84_items(self.stack, pick)
        self.check_bench(pick)
        requests = [
            dict(
                name=i["name"],
                payload=dict(model=pick, messages=i["messages"], max_tokens=1),
                timeout_s=120,
            )
            for i in items
        ]
        result = self.run(self.job(pick, "panel-count", "count", "off", requests))
        counts = {
            r["name"]: dict(tokens=r["tokens"], sha256=r["sha256"])
            for r in result["requests"]
        }
        panel = build_panel(
            items_source(pick)[0], self.bundles_of(pick), items, {pick: counts}
        )
        (self.out / "borrowed_panel.json").write_text(
            json.dumps(panel, indent=1) + "\n"
        )
        return panel

    def check_reconstruction(self, pick, items, panel=None):
        requests = [
            dict(
                name=i["name"],
                payload=dict(model=pick, messages=i["messages"], max_tokens=1),
                timeout_s=120,
            )
            for i in items
        ]
        result = self.run(self.job(pick, "c84-count", "count", "off", requests))
        if (
            panel is not None
        ):  # v13 V4: the borrowed entry's counts / digests are the reviewed panel's
            live = {
                r["name"]: dict(tokens=r["tokens"], sha256=r["sha256"])
                for r in result["requests"]
            }
            pinned = {n: panel["entries"][pick][n] for n in live}
            if live != pinned:
                raise RuntimeError(f"{pick}: templated prompts differ from the panel")
        else:
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
        """C185 (plan v8 §3 E2): the same-instance triple (ON, OFF1, OFF2) per request in the ON worker
        (``e2_pair``, one request budget for the three), then the cross-load OFF arm once.
        """
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
        requests = reqs + straddles
        triples = [dict(r, timeout_s=3 * r["timeout_s"]) for r in requests]
        loaded = {}
        for name, mode, arm, batch in (
            ("e2-pair", "e2_pair", "on", triples),
            ("e2-off", "e2", "off", requests),
        ):
            self.run(self.job(pick, name, mode, arm, batch))
            loaded[name] = {
                r["name"]: json.loads(
                    (self.out / pick / name / f"req-{i}.json").read_text()
                )
                for i, r in enumerate(batch)
            }
        return e2_summary(
            loaded["e2-pair"], loaded["e2-off"], digests=self.digests.get(pick, {})
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

    def e7b(self, pick, items, sessions=5):
        """Scripted multi-turn chat continuity (plan §3 E7b, descriptive): ``sessions`` items, each followed by
        the fixed follow-ups; the smoke runs one (P394 (2) refined) so the echo / re-tokenization / hash-chain
        mechanics are exercised before the full run."""
        count, sessions = sessions, []
        for item in items[:count]:
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
    def check_bench(self, pick):
        """Code rounds 3–4 H9: the stack bench inputs are clean AND still the bound ones."""
        problems = stack_bench_problems(self.stack, self.bench_paths)
        problems += stack_bench_drift(self.stack, self.stack_bench)
        if problems:
            raise RuntimeError(f"{pick}: {problems}")

    def run_pick(self, pick, *, smoke):
        """One pick under the Q37 precedence. A supervisor stop (``JobTimeout``) can never pass: in E1c the
        pick ABORTs (no verdict, no bound), in E2 the identity is INCOMPLETE; E7 censors per request.
        """
        started = time.monotonic()
        record = self.summary["picks"].setdefault(pick, {})
        record["kv"] = kvmanifest.kv_from_argv(self.arms[pick]["on"])  # P405 F3

        mark_at = [started]

        def save(step=None):  # C184: every finished step reaches summary.json at once
            now = time.monotonic()
            record["wall_s"] = round(now - started)
            if step:  # v13 V13: per-phase timings for the projection
                record.setdefault("step_s", {})[step] = round(now - mark_at[0], 1)
            mark_at[0] = now
            self.write_summary()

        thresholds = SMOKE_THRESHOLDS if smoke else THRESHOLDS
        prompts = self.size(pick, thresholds)
        self.check_bench(pick)  # still exactly the bound bench inputs, before...
        items = c84_items(self.stack, pick)
        self.check_bench(pick)  # ...and after the reconstruction imported them (H9)
        panel = borrowed_panel(
            pick, PANEL
        )  # v13 V4: None for an entry with its own bundles
        if panel is not None:
            problems = panel_problems(panel, pick, self.bundles_of(pick), items)
            if problems:
                raise Refused(f"refused: {problems}")
        record["reconstruction"] = self.check_reconstruction(
            pick, items[:2] if smoke else items, panel=panel
        )
        save("prep")
        try:
            record["e1c"] = self.e1c(pick, prompts, smoke=smoke)
            record["outcome"] = pick_outcome(record["e1c"])
        except JobTimeout as exc:
            record["outcome"] = dict(
                e1c="ABORT", run_e2_e7=False, reason=f"E1c worker stopped: {exc.info}"
            )
        save("e1c")
        if record["outcome"]["run_e2_e7"]:  # Q37 precedence
            try:
                record["e2"] = self.e2(
                    pick,
                    items,
                    e2_prompts(
                        prompts, getattr(self, "accepted_prompts", {}).get(pick, {})
                    ),
                    smoke=smoke,
                )
            except JobTimeout as exc:
                record["e2"] = dict(
                    verdict="INCOMPLETE", reason=f"E2 worker stopped: {exc.info}"
                )
            save("e2")
            if (
                not smoke
            ):  # code round 2 H16: E7a reaches summary.json before E7b starts
                record["e7a"] = self.e7a(pick, items)
                save("e7a")
                record["e7b"] = self.e7b(pick, items)
                save("e7b")
            else:  # P394 (2) refined: one E7b session exercises the continuity mechanics in the smoke
                record["e7b"] = self.e7b(pick, items, sessions=1)
                save("e7b")
        return record

    def execute(self):
        """The campaign under ONE persistence boundary (code round 3 H23): after ``prepare()`` (which runs no
        job), any stop — the binding refusal, a wall limit, a harness error, the final tree guard, an
        interruption (SIGTERM → SystemExit) — writes its status to ``summary.json`` before propagating.
        """

        def stop(status, exc):
            self.summary["status"], self.summary["exit_code"] = status, 1
            self.summary["error"] = f"{type(exc).__name__}: {exc}"
            if getattr(
                self, "_out_created", False
            ):  # never into a run dir prepare did not create
                self.write_summary()

        try:  # code round 4 H23: preparation inside the boundary too
            manifest = self.prepare()
            self.summary["binding"] = binding(manifest)
            return self._execute(manifest)
        except Refused as exc:
            stop("REFUSED", exc)
            raise SystemExit(str(exc)) from None
        except (
            WallLimit
        ) as exc:  # code round 1 H8: the worker was stopped; nothing more runs
            self.summary["aborted"] = f"wall_limit: {exc}"
            stop("ABORTED", exc)
            raise
        except Exception as exc:  # code rounds 2 H16, 3 H23
            stop("HARNESS_ERROR", exc)
            raise
        except (
            BaseException
        ) as exc:  # SystemExit from the SIGTERM handler, KeyboardInterrupt
            stop("INTERRUPTED", exc)
            raise

    def _execute(self, manifest):
        if self.args.smoke_summary:  # code round 1 H9: before any GPU job
            smoke_summary = json.loads(Path(self.args.smoke_summary).read_text())
            problems = binding_problems(smoke_summary, manifest)
            if problems:
                raise Refused(f"full: the smoke does not bind this run: {problems}")
        estimate = 20 * 60 if self.args.phase == "smoke" else int(ESTIMATE_LIMIT_S)
        smoke = self.args.phase == "smoke"
        if self.args.phase == "panel":  # v13 V4: count-only, no generation
            with Monitor(self.out / "monitor.log", 20 * 60) as self.monitor:
                for pick in self.picks:
                    self.summary.setdefault("panels", {})[pick] = self.make_panel(pick)
            self.summary["status"], self.summary["exit_code"] = "PANEL_WRITTEN", 0
            self.write_summary()
            return self.summary
        with Monitor(self.out / "monitor.log", estimate) as self.monitor:
            for pick in self.picks:
                self.run_pick(pick, smoke=smoke)
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
        return smoke_gate(self.summary["picks"])

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
    ap.add_argument("--phase", choices=("smoke", "full", "panel"), required=True)
    ap.add_argument("--pick", action="append")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument(
        "--smoke-summary", help="full runs refuse unless this passed smoke summary fits"
    )
    ap.add_argument(
        "--wall-limit-s",
        type=positive_seconds,
        help="an operator-set hard total-wall stop (default: none, C184)",
    )
    ap.add_argument(
        "--projection-ruling",
        help="C193: a recorded operator ruling (PROJECTION_RULINGS) that lifts only the smoke gate's fits",
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
    if args.reviewed_head:
        args.reviewed_head = resolve_commit(args.fork_root, args.reviewed_head)
    problems = reviewed_head_problems(args.phase, args.reviewed_head, head)
    problems += out_problems(out, os.environ)
    if problems:
        raise SystemExit(f"refused: {problems}")
    if args.phase == "full":
        if not args.smoke_summary:
            raise SystemExit("full: --smoke-summary of a passed smoke is required")
        smoke = json.loads(Path(args.smoke_summary).read_text())
        problems = smoke_authorization_problems(  # code round 6 H36; C193
            smoke, projection_ruling=args.projection_ruling
        )
        if problems:
            raise SystemExit(f"full: the smoke summary does not authorize: {problems}")
    checks = preflight(args, out.parent)
    print(json.dumps(dict(preflight=checks), indent=1))
    if args.plan_only:
        return
    if checks["problems"]:
        raise SystemExit(f"preflight refused: {checks['problems']}")
    lock = lock_path(os.environ)
    lock.parent.mkdir(parents=True, exist_ok=True)
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
