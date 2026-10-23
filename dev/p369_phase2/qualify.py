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
PRODUCTION = (
    "7207179b"  # the production pin every qualification and its P369 evidence run on
)
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


class UnknownKey(ValueError):
    pass


def arm_for(name, predictor):
    """P426 B7: the arm ``name`` is measured in — C199's shipped arm without ``--predictor``; ``off`` serves the committed
    registry (``mtp_off`` for an MTP-certified entry, ``served`` otherwise); ``on`` is the shipped MTP-ON arm.
    """
    shipped = ENTRIES[name]
    if predictor is None:
        return shipped
    if predictor == "off":
        return "served" if shipped == "served" else "mtp_off"
    if shipped == "served":
        raise ValueError(f"{name}: no MTP-ON arm (not MTP-certified)")
    return shipped


def default_entries():
    """Code round 1 D8: the full qualification is all seven (Qwen3.6 then needs its P369 evidence); a subset only via
    ``--entry``."""
    return list(ENTRIES)


def entry_problems(names, *, p369_summary, predictor=None):
    """Every name is one of the seven; ``Qwen3.6-27B-Opus-Distill-OptiQ-4bit`` MTP-ON only with a P369 FULL-run PASS for it
    (its ``--predictor off`` arm needs none, P426 B7)."""
    problems = [f"unknown entry {n!r}" for n in names if n not in ENTRIES]
    for name in names:
        if name not in ENTRIES:
            continue
        try:
            arm = arm_for(name, predictor)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if arm != "mtp_on_gated":
            continue
        summary = json.loads(Path(p369_summary).read_text()) if p369_summary else None
        pick = ((summary or {}).get("picks") or {}).get(name) or {}
        binding = (summary or {}).get("binding") or {}
        manifest = (summary or {}).get("manifest") or {}
        passed = (
            summary is not None
            and summary.get("phase") == "full"
            and summary.get("status") == "PASS"
            and (pick.get("outcome") or {}).get("e1c") == "PASS"
            and (pick.get("e2") or {}).get("verdict") == "PASS"
            # code round 1 D10: the run's own provenance — the production pin and a bound reviewed head
            and binding.get("production") == PRODUCTION
            and manifest.get("production") == PRODUCTION
            and bool(binding.get("reviewed_head"))
            and binding.get("reviewed_head") == binding.get("fork_head")
            and list(summary.get("picks") or {}) == [name]
        )
        if not passed:
            problems.append(f"{name}: MTP-ON only after its P369 full-run PASS (C199)")
    return problems


def summary_artifact_problems(path, summary, *, root, is_commit, contains_reviewed):
    """Code round 2 D10: the P369 summary is a real campaign artifact — under the campaign root, beside its own
    ``manifest.json`` naming the same fork head, and that head a real fork commit containing the reviewed v13 head.
    """
    problems = []
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        problems.append(f"summary {path} is not under {root}")
    head = ((summary or {}).get("binding") or {}).get("fork_head")
    try:
        manifest = json.loads((path.parent / "manifest.json").read_text())
    except (OSError, ValueError):
        manifest = None
    if not manifest or manifest.get("fork_head") != head:
        problems.append(
            "the summary's manifest.json is missing or names another fork head"
        )
    if not head or not is_commit(head):
        problems.append(f"fork head {head!r} is not a fork commit")
    elif not contains_reviewed(head):
        problems.append(f"fork head {head} does not contain the reviewed v13 head")
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
        # code round 1 D1: production holds NO policy object for the default per-query scan (served_state → None)
        verify_policy=(
            values["mtp_verify_scan"]
            if drafter and values["mtp_verify_scan"] not in ("", "per_query")
            else None
        ),
        kv_quant_mode=values["kv_quant_mode"] or None,
        quantized_kv_start=int(values["quantized_kv_start"]),
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


def landing_offsets(landings, prompt_tokens):
    """R7 full run 1: the absolute offsets production's chunk loop lands on exactly (``generate/ar.py``): the snapshot
    offset, the retention boundary (only with a prompt-end capture, and only inside the prompt), the APC checkpoints.
    """
    out = []
    if landings.get("snapshot_at_offset") is not None:
        out.append(int(landings["snapshot_at_offset"]))
    retain = landings.get("retain_at_offset")
    if (
        landings.get("prompt_end_offset")
        and retain is not None
        and 0 < int(retain) < prompt_tokens
    ):
        out.append(int(retain))
    out += [int(n) for n in (landings.get("checkpoint_lengths") or [])]
    return sorted(set(out))


def chunk_sequence(prompt_tokens, step, landings=()):
    """The chunked prefill's forwards for a fresh prompt (``generate/ar.py``): chunks of ``min(step, left − 1)`` while
    more than one token is left, each shrunk to end exactly on a landing offset it would cross
    (``_adjust_chunk_for_snapshot_landing``: strict crossing only); the last token goes through the prompt step.
    """
    out, done, pending = [], 0, sorted(set(landings))
    while prompt_tokens - done > 1:
        n = min(step, prompt_tokens - done - 1)
        for target in pending:
            if done < target < done + n:
                n = target - done
                break
        out.append(n)
        done += n
        pending = [t for t in pending if t > done]
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
    if (
        rec.get("delegates_to_cli_main") is not True
    ):  # code round 1 D4 (proven in the worker)
        problems.append("mlx_vlm.server.main does not run mlx_vlm.server.cli.main")
    if rec.get("boundary_seen") is not True:
        problems.append("no prefill / decode boundary observed (Q11)")
    if rec.get("prompt_step_tokens") != 1:
        problems.append(f"prompt step {rec.get('prompt_step_tokens')} tokens, not 1")
    landings = rec.get("landings")
    if landings is None:
        problems.append("no landing kwargs recorded for the chunk sequence")
    elif rec.get("chunks") != chunk_sequence(
        target, expected["prefill_step"], landing_offsets(landings, target)
    ):
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
    # code round 1 D3: the measured cache's exact leaf layout (P405 F3) against the REGISTRY-derived codec, and every
    # KV leaf's allocated capacity exactly the registry's preallocation floor
    from p369_phase2 import kvmanifest

    observed, template = rec.get("manifest"), rec.get("template")
    if not observed or not template:
        problems.append("no leaf manifest / template of the measured cache")
    else:
        problems += kvmanifest.manifest_problems(
            observed, template, expected_kv(expected)
        )
        caps = rec.get("capacities_by_path") or {}
        kv_paths = [
            o["path"] for o in observed if o.get("kind") in ("tq", "affine", "native")
        ]
        if not kv_paths:
            problems.append("no KV leaf in the measured cache")
        for path in kv_paths:
            if caps.get(path) != expected["prealloc"]:
                problems.append(
                    f"layer {path} capacity {caps.get(path)} != {expected['prealloc']}"
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
    want_limit = int(min(limits)) if limits else None
    if want_limit is None or memory.get("set_memory_limit") != [want_limit]:
        problems.append(
            f"memory limit {memory.get('set_memory_limit')} != {want_limit}"
        )
    caches = memory.get("set_cache_limit") or []
    if expected["cache_limit_gb"] > 0:
        want_cache = int(expected["cache_limit_gb"] * 2**30)
    else:  # code round 1 D9: mlx-vlm's _derive_cache_limit_gb from the model's heads
        want_cache = expected_cache_limit(memory.get("heads"), expected)
    if caches != [want_cache]:
        problems.append(f"cache limit {caches} != [{want_cache}]")
    checks = rec.get("drafter_live_checks")
    if expected["drafter"]:
        if not checks or any(checks):
            problems.append(f"drafter live checks {checks}")
    elif checks:
        problems.append(f"drafter checks without a drafter {checks}")
    return problems


def expected_kv(expected):
    """The KV codec the REGISTRY asks for, in ``kvmanifest.kv_from_argv``'s shape (the CLI defaults where unset)."""
    from p369_phase2 import kvmanifest

    return dict(
        kv_bits=float(expected["kv_bits"]) if expected["kv_bits"] else None,
        kv_key_bits=None,
        kv_value_bits=None,
        kv_key_scheme=None,
        kv_value_scheme=None,
        kv_quant_scheme=expected["kv_quant_scheme"]
        or kvmanifest.CLI_DEFAULTS["kv_quant_scheme"],
        kv_group_size=kvmanifest.CLI_DEFAULTS["kv_group_size"],
        quantized_kv_start=expected["quantized_kv_start"],
        tq_kv_quant_mode=expected["kv_quant_mode"] or kvmanifest.DEFAULT_TQ_MODE,
    )


def expected_cache_limit(heads, expected):
    """mlx-vlm ``server/cli.py`` ``_derive_cache_limit_gb``: ceil(heads × prefill step × max KV × 2 B / 1e9) + 2 GB
    (heads default 32), applied as ``int(gb × 2**30)`` bytes; None without a cap or a step.
    """
    if not (expected["cap"] and expected["prefill_step"]):
        return None
    gb = (
        math.ceil((heads or 32) * expected["prefill_step"] * expected["cap"] * 2 / 1e9)
        + 2.0
    )
    return int(gb * 2**30)


def entry_point_problems(info, venv_bin):
    """Code round 1 D4: the shipping console script runs the main the worker boots — its shebang is the stack venv's
    python, its entry point ``mlx_vlm.server:main``, and that main delegates to ``mlx_vlm.server.cli.main``.
    """
    problems = []
    if info.get("shebang") != str(Path(venv_bin) / "python"):
        problems.append(f"console script interpreter {info.get('shebang')!r}")
    if info.get("entry") != "mlx_vlm.server:main":
        problems.append(f"console script entry point {info.get('entry')!r}")
    if info.get("delegates_to_cli_main") is not True:
        problems.append("mlx_vlm.server.main does not run mlx_vlm.server.cli.main")
    return problems


def guard_problems(start, now, *, reviewed):
    """Code round 1 D5: the fork is clean at the reviewed head and the stack HEAD has not moved."""
    problems = []
    if now.get("fork_head") != reviewed:
        problems.append(
            f"fork at {now.get('fork_head')}, not the reviewed head {reviewed}"
        )
    if not now.get("fork_clean"):
        problems.append("fork tree not clean")
    if now.get("stack_head") != start.get("stack_head"):
        problems.append(
            f"stack HEAD moved {start.get('stack_head')} -> {now.get('stack_head')}"
        )
    return problems


class MonitorHealth:
    """Code round 3 D12: wraps a monitor's ``tick`` in place — every tick is counted and every failure recorded
    (sticky), then re-raised; a failed tick can never be hidden by a later successful one.
    """

    def __init__(self, monitor):
        self.ticks, self.errors = 0, []
        real = monitor.tick

        def tick(*a, **k):
            try:
                out = real(*a, **k)
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                raise
            self.ticks += 1
            return out

        monitor.tick = tick


def monitor_problems(*, alive, last_tick_age_s, period_s=300, errors=()):
    """Code round 1–3 D12: the 5-minute monitor is alive, ticked within one period (+ 60 s), and no tick failed."""
    problems = [f"a monitor tick failed: {e}" for e in errors]
    if not alive:
        problems.append("the monitor thread is dead")
    if last_tick_age_s is None or last_tick_age_s > period_s + 60:  # round 2 D12
        problems.append(f"the monitor's last tick is {last_tick_age_s} s old")
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
    "weights",
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


def load_references(path, *, table=None, blob_exists=None):
    """The reviewed reference table; every reference names its source and that source's git blob id, which must exist
    in the stack repository (code round 1 D11: an immutable locator, not a mutable path).
    """
    table = json.loads(Path(path).read_text()) if table is None else table
    for name, ref in table.items():
        if name not in ENTRIES:
            raise ValueError(f"reference for an unknown entry {name!r}")
        if ref is None:
            continue
        if not (ref.get("source") and _finite_positive(ref.get("peak_gb"))):
            raise ValueError(
                f"{name}: a reference needs a source and a positive peak_gb"
            )
        if not ref.get("source_path"):
            raise ValueError(f"{name}: a reference needs its source_path")
        if blob_exists is not None and not blob_exists(
            ref.get("source_blob"), ref["source_path"]
        ):  # code round 2 D11: the blob is a version OF that path
            raise ValueError(
                f"{name}: blob {ref.get('source_blob')!r} is not a version of {ref['source_path']}"
            )
    return table


def battery_intervals(log_text, start, end):
    """P418: the on-battery intervals (``pmset -g log`` "Using Batt" … "Using AC") overlapping ``[start, end]`` (epoch
    seconds) — exact, unlike 5-minute ticks or before / after samples; an interval still open at the end has ``None``.
    """
    import datetime
    import re

    events = []
    for line in log_text.splitlines():
        m = re.match(
            r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4}) .*Using (AC|Batt)\b", line
        )
        if m:
            at = datetime.datetime.strptime(
                m.group(1), "%Y-%m-%d %H:%M:%S %z"
            ).timestamp()
            events.append((at, m.group(1), m.group(2)))
    out, opened = [], None
    for at, text, source in sorted(events):
        if source == "Batt" and opened is None:
            opened = (at, text)
        elif source == "AC" and opened is not None:
            if opened[0] <= end and at >= start:
                out.append((opened[1], text))
            opened = None
    if opened is not None and opened[0] <= end:
        out.append((opened[1], None))
    return out


def power_log_problems(returncode, log_text):
    """Code round 3 F1: the power log must have been read and must carry power-source history."""
    import re

    if returncode != 0:
        return [f"pmset -g log failed (rc {returncode})"]
    if not re.search(r"Using (AC|Batt)\b", log_text or ""):
        return ["pmset -g log has no power-source history"]
    return []


def idle_before(names):
    return [i > 0 for i in range(len(names))]


def entry_status(*, pre, post, error, problems, monitor=(), power_events=()):
    """C198 (3): interference / power before or after an entry ⇒ INVALID (and a dead / stale monitor, code round 1
    D12); an error or a problem ⇒ FAILED."""
    if monitor:
        return dict(status="INVALID", reason=f"monitor: {'; '.join(monitor)}")
    if power_events:  # P418: any on-battery interval inside the entry
        return dict(
            status="INVALID",
            reason=f"on battery during the entry: {list(power_events)}",
        )
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


def interference_problems(orchestrate):
    """The per-entry process census: refused processes other than this run's own children and (P452) its P426
    launcher ancestors — the SAME rule as the preflight (``orchestrate.foreign_process_problems``).
    """
    import psutil

    mine = {c.pid for c in psutil.Process().children(recursive=True)}
    procs = [
        (p.info["pid"], p.info["cmdline"])
        for p in psutil.process_iter(["pid", "cmdline"])
        if p.info["pid"] not in mine
    ]
    return orchestrate.foreign_process_problems(
        procs, os.getpid(), orchestrate.ancestor_pids()
    )


def interference(orchestrate):
    """C198 (3): power and the process scan, once before and once after each entry."""
    power = orchestrate.power_state()
    if not (power["ac_ok"] and power["battery_ok"]):
        return dict(ok=False, why=f"power {power}", power=power)
    problems = interference_problems(orchestrate)
    if problems:
        return dict(ok=False, why=problems[0], power=power)
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


def _tree_state(fork, stack):
    import subprocess

    dirty = subprocess.run(
        ["git", "-C", str(fork), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return dict(
        fork_head=_git_head(fork), fork_clean=not dirty, stack_head=_git_head(stack)
    )


def console_script_info(stack):
    """Code round 1 D4: the stack venv's ``mlx_vlm.server`` console script — its interpreter and entry point (the
    stack-side mlx-vlm is the pre-bump submodule; its location is recorded, the worker runs the fork tree).
    """
    import subprocess

    script = Path(stack) / ".venv" / "bin" / VISION_SERVER
    shebang = script.read_text().splitlines()[0].removeprefix("#!").strip()
    code = (
        "import importlib.metadata as m, json, mlx_vlm\n"
        "eps = [e.value for e in m.entry_points(group='console_scripts') if e.name == 'mlx_vlm.server']\n"
        "print(json.dumps(dict(entry=eps[0] if len(eps) == 1 else eps, stack_mlx_vlm=mlx_vlm.__file__)))\n"
    )
    out = subprocess.run(
        [str(Path(stack) / ".venv" / "bin" / "python"), "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=stack,
    )
    return dict(json.loads(out.stdout.strip().splitlines()[-1]), shebang=shebang)


def drive(args):
    import sys

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from p369_phase2 import kvmanifest, orchestrate, registry_states

    stack, fork, out = (
        Path(p).resolve() for p in (args.stack_root, args.fork_root, args.out)
    )
    names = list(args.entry or default_entries())
    predictor = getattr(args, "predictor", None)
    problems = entry_problems(
        names, p369_summary=args.p369_summary, predictor=predictor
    )
    if problems:
        raise SystemExit(f"refused: {problems}")
    reviewed = _resolve_ref(fork, args.reviewed_head)
    if args.p369_summary and any(
        arm_for(n, predictor) == "mtp_on_gated" for n in names
    ):
        problems = summary_artifact_problems(  # code round 2 D10
            args.p369_summary,
            json.loads(Path(args.p369_summary).read_text()),
            root=Path(os.environ.get("STACK_WORKDIR", "/nonexistent"))
            / "p369"
            / "phase2",
            is_commit=lambda c: _is_commit(fork, c),
            contains_reviewed=lambda c: _is_ancestor(fork, V13_REVIEWED, c),
        )
        if problems:
            raise SystemExit(f"refused: {problems}")
    out.mkdir(parents=True, exist_ok=False)
    lock = orchestrate.acquire_lock(orchestrate.lock_path(dict(os.environ)))
    try:  # code round 2 D13: the lock is released whatever the cleanup below does
        return _drive_locked(
            args,
            names,
            reviewed,
            stack,
            fork,
            out,
            orchestrate,
            kvmanifest,
            registry_states,
            yaml,
        )
    finally:
        orchestrate.release_lock(lock)


def _drive_locked(
    args,
    names,
    reviewed,
    stack,
    fork,
    out,
    orchestrate,
    kvmanifest,
    registry_states,
    yaml,
):
    import signal
    import subprocess
    import time

    predictor = getattr(args, "predictor", None)
    arms = {n: arm_for(n, predictor) for n in names}  # P426 B7
    report = dict(
        predictor=predictor,
        entries={n: dict(status="PENDING", arm=arms[n]) for n in names},
    )
    stop = dict(signal=None)
    monitor = None
    boot = orchestrate.BOOT_TIMEOUT_S
    orchestrate.BOOT_TIMEOUT_S = (
        BOOT_S  # code round 2 E2: the qualification's own boot limit
    )
    try:  # code round 1 D13: everything after the lock is inside the cleanup scope
        if getattr(args, "production_ref", None) not in (None, orchestrate.PRODUCTION):
            try:  # P452 code round 1 F3: one resolved sha, diffed by the preflight and recorded in the report
                args.production_ref = orchestrate.resolve_production_ref(
                    fork, args.production_ref, reviewed
                )
            except ValueError as exc:
                raise SystemExit(f"refused: {exc}")
        pre = orchestrate.preflight(args, out)
        if pre["problems"]:
            raise SystemExit(f"refused: {pre['problems']}")
        start = _tree_state(fork, stack)
        problems = guard_problems(start, start, reviewed=reviewed)  # D5
        registry = orchestrate.read_registry(stack / "main_models.yaml")
        registry_sha = orchestrate.sha256_file(stack / "main_models.yaml")
        fixture = registry_states.fixture_text(fork / "mlx_vlm" / "tests" / "fixtures")
        off_entries = registry_states.classify(registry, fixture)
        if off_entries is None:
            problems.append(
                "the live registry is not one of the 16 accepted byte states"
            )
        console = console_script_info(stack)
        problems += entry_point_problems(
            dict(console, delegates_to_cli_main=True), stack / ".venv" / "bin"
        )  # the delegation itself is proven inside the worker (``check_server_main``)
        if problems:
            raise SystemExit(f"refused: {problems}")

        def blob_exists(blob, source_path):  # code round 2 D11: a version OF that path
            if not blob:
                return False
            log = subprocess.run(
                [
                    "git",
                    "-C",
                    str(stack),
                    "log",
                    "--all",
                    "--format=%H",
                    f"--find-object={blob}",
                    "--",
                    source_path,
                ],
                capture_output=True,
                text=True,
            )
            return log.returncode == 0 and bool(log.stdout.strip())

        references = load_references(REFERENCES, blob_exists=blob_exists)
        report.update(
            started=time.strftime("%Y-%m-%dT%H:%M:%S"),
            reviewed_head=reviewed,
            start=start,
            production=getattr(args, "production_ref", None) or orchestrate.PRODUCTION,
            registry_sha256=registry_sha,
            versions=orchestrate.fork_versions(fork),
            console_script=console,
        )
        _atomic(out / "report.json", report)

        def on_signal(signum, _frame):
            stop["signal"] = signum
            raise SystemExit(128 + signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, on_signal)
        monitor = orchestrate.Monitor(out / "monitor.log", estimate_s=len(names) * 1200)
        health = MonitorHealth(monitor)  # code round 3 D12: before the thread starts
        monitor.__enter__()
        for i, name in enumerate(names):
            if i:
                time.sleep(args.idle_s)
            state = report["entries"][name]
            state.update(status="RUNNING", started=time.strftime("%Y-%m-%dT%H:%M:%S"))
            _atomic(out / "report.json", report)
            window_start = time.time()
            before = interference(orchestrate)
            error, problems, after, rec, expected = None, [], None, None, None
            if before["ok"]:
                try:
                    problems = guard_problems(
                        start, _tree_state(fork, stack), reviewed=reviewed
                    )
                    if arms[name] == "mtp_off" and name not in (off_entries or ()):
                        problems.append(  # P426 B7: OFF means the committed registry's OFF block
                            f"--predictor off: the committed registry serves {name} MTP-ON"
                        )
                    if arms[name] in ("served", "mtp_off"):
                        path, text = stack / "main_models.yaml", registry
                    else:  # C199: the shipped MTP-ON form, the 16-state generator's exact block
                        text, _ = orchestrate.overlay_registry(registry, name, fixture)
                        path = out / f"overlay-{name}.yaml"
                        path.write_text(text)
                    entry = {e["name"]: e for e in yaml.safe_load(text)["models"]}[name]
                    expected = expected_config(entry)
                    cmd, module_file = full_command(stack, path, name)
                    problems += binding_problems(
                        stack, module_file, registry_sha, orchestrate.sha256_file
                    ) + command_problems(expected, cmd, stack / ".venv" / "bin")
                    if not problems:
                        job = _job(
                            name, cmd[1:], expected, fork, out, orchestrate, kvmanifest
                        )
                        result = orchestrate.run_job(
                            job,
                            fork,
                            orchestrate.worker_env(dict(os.environ)),
                            monitor,
                            deadline=math.inf,
                        )
                        rec = dict(
                            result["requests"][0]["qualify"],
                            snapshots=result.get("snapshots"),
                            weights=_weights_digest(job["expect"]["revisions"]),
                            harness=reviewed,
                            versions=report["versions"],
                        )
                        problems = request_problems(rec, expected)
                        # D5: the bindings again after the job
                        problems += guard_problems(
                            start, _tree_state(fork, stack), reviewed=reviewed
                        )
                        problems += binding_problems(
                            stack, module_file, registry_sha, orchestrate.sha256_file
                        )
                        if console_script_info(stack) != console:  # code round 2 D5
                            problems.append(
                                "the stack console script changed during the job"
                            )
                except SystemExit:
                    raise
                except (
                    Exception
                ) as exc:  # recorded per entry; the run continues (plan v2 Q5)
                    error = f"{type(exc).__name__}: {exc}"
                after = interference(orchestrate)
            rc, log_text = _pmset_log()
            events = battery_intervals(log_text, window_start, time.time())  # P418
            events += [
                ("power log", p) for p in power_log_problems(rc, log_text)
            ]  # code round 3 F1: an unreadable log is never "no outage"
            last = _last_tick_age(out / "monitor.log")
            mon = monitor_problems(
                alive=monitor._thread.is_alive(),
                last_tick_age_s=last,
                errors=health.errors,
            )
            # code round 1 D6: every piece of evidence is final BEFORE the terminal status is written
            final = dict(problems=problems[:50], before=before, after=after, record=rec)
            if rec and not error and not problems:
                peak = capacity_gb(rec["peaks"])
                ref = references.get(name)
                final.update(
                    capacity_gb=peak,
                    vs_guidance_gb=round(peak - 48.0, 3),
                    reference=(
                        None
                        if ref is None
                        else reference_comparison(
                            ref, peak, fingerprint_of(rec, expected)
                        )
                    ),
                )
            final.update(
                entry_status(
                    pre=before,
                    post=after,
                    error=error,
                    problems=problems,
                    monitor=mon,
                    power_events=events,
                ),
                power_events=events,
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            state.update(final)
            _atomic(out / "report.json", report)
        report["end"] = _tree_state(fork, stack)
        report["end_problems"] = guard_problems(start, report["end"], reviewed=reviewed)
        try:  # code round 3 D5: an unreadable script at exit is a problem, never an escape
            if console_script_info(stack) != console:  # code round 2 D5
                report["end_problems"].append(
                    "the stack console script changed during the run"
                )
        except Exception as exc:
            report["end_problems"].append(
                f"the stack console script is unreadable at exit: {type(exc).__name__}: {exc}"
            )
        if report["end_problems"]:
            for state in report["entries"].values():
                if state["status"] == "COMPLETED":
                    state.update(
                        status="INVALID", reason=f"at exit: {report['end_problems']}"
                    )
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
        orchestrate.BOOT_TIMEOUT_S = boot
        if monitor is not None:
            monitor.__exit__(None, None, None)
    return exit_code(report["entries"])


V13_REVIEWED = (
    "3ae8770e"  # C199: the reviewed v13 head every P369 campaign head must contain
)


def _weights_digest(revisions):
    import hashlib

    return {
        flag: hashlib.sha256(
            json.dumps(v["files"], sort_keys=True).encode()
        ).hexdigest()
        for flag, v in revisions.items()
    }


def _is_commit(path, ref):
    import subprocess

    return (
        subprocess.run(
            ["git", "-C", str(path), "cat-file", "-e", f"{ref}^{{commit}}"],
            capture_output=True,
        ).returncode
        == 0
    )


def _is_ancestor(path, ancestor, ref):
    import subprocess

    return (
        subprocess.run(
            ["git", "-C", str(path), "merge-base", "--is-ancestor", ancestor, ref],
            capture_output=True,
        ).returncode
        == 0
    )


def _resolve_ref(path, ref):
    import subprocess

    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _pmset_log():
    import subprocess

    got = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True)
    return got.returncode, got.stdout


def _last_tick_age(path):
    import time

    try:
        return time.time() - Path(path).stat().st_mtime
    except OSError:
        return None


def _job(name, argv, expected, fork, out, orchestrate, kvmanifest):
    model = argv[argv.index("--model") + 1]
    revisions = {"--model": orchestrate.weights_record(model)}
    if expected["drafter"]:
        revisions["--draft-model"] = orchestrate.weights_record(
            argv[argv.index("--draft-model") + 1]
        )
    return dict(
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
                orchestrate.PICKS[name]["drafter"] if expected["drafter"] else None
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
        snapshot=dict(
            rec.get("snapshots") or {}
        ),  # code round 2 E1: check_snapshots' {flag: revision}
        weights=dict(
            rec.get("weights") or {}
        ),  # D11: the worker-verified file hashes, digested per flag
        harness=rec.get("harness"),  # D11: the reviewed head the run is bound to
        context_cap=expected["cap"],
        memory_limits=dict(
            memory=(rec.get("memory") or {}).get("set_memory_limit"),
            cache=(rec.get("memory") or {}).get("set_cache_limit"),
        ),
        shrink="on" if expected["shrink"] else "off",
        output_tokens=rec.get("generation_tokens"),
        backend=VISION_SERVER,
        prompt_identity=((rec.get("prompt") or {}).get("sized") or {}).get("sha256"),
    )


def build_parser():
    import argparse
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1])
    )  # as drive(): run as a script, not a package
    from p369_phase2 import orchestrate

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack-root", required=True)
    ap.add_argument("--fork-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--entry", action="append")
    ap.add_argument("--p369-summary")
    ap.add_argument("--reviewed-head", required=True)  # code round 1 D5
    ap.add_argument(  # P426 B7: the M1 memory axis measures both predictor states
        "--predictor", choices=("on", "off"), default=None
    )
    ap.add_argument("--idle-s", type=float, default=IDLE_S)
    ap.add_argument(  # P452: the production tree this run attests (P426: stack S carries B1's log-only counters)
        "--production-ref", default=orchestrate.PRODUCTION
    )
    return ap


def main(argv=None):
    import sys

    ap = build_parser()
    args = ap.parse_args(argv)
    if args.idle_s < IDLE_S:
        ap.error(f"--idle-s must be at least {IDLE_S:.0f} (AGENTS.md)")
    sys.exit(drive(args))


if __name__ == "__main__":
    main()
