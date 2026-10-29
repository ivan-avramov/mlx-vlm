"""P415 (plan v5, C196 / C198 / C199): the R7 256K operational qualification — pure parts (fork-only, dev).

Per entry, the configuration it SHIPS (C199: the MTP-certified entries MTP-ON, the others as served), the largest prompt the
server accepts (context cap − ``MIN_OUTPUT_TOKENS``) and the registry's own request settings (C198: no override but the
seed). The expectation is derived from the registry entry independently of mlx-serve's argv builder (plan v2 Q1): a
reviewed key → flag table with mlx-serve's own defaults; an unknown registry key refuses.
"""

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # run as a script: the package is not on sys.path
from p369_phase2 import registry_states  # noqa: E402

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


class UnknownKey(ValueError):
    pass


# the entry's MTP registry keys (bench.p426_leg.MTP_KEYS / C177); the fork cannot import the stack
MTP_KEYS = ("draft_kind", "draft_model", "mtp_verify_scan")
MTP_ON_ARMS = ("served_on", "mtp_on_restored")  # the arms the C199 gate applies to


def serves_mtp(entry):
    """The committed block is served MTP-ON: ``expected_config``'s drafter (``draft_kind: mtp`` with a ``draft_model``).
    A block with only some MTP keys, or a non-MTP ``draft_kind``, refuses (spec v1.2).
    """
    drafter = expected_config(entry)["drafter"]
    if not drafter and any(k in entry for k in MTP_KEYS):
        raise ValueError(f"{entry.get('name')}: a partial or non-MTP draft block")
    return drafter


def arm_for(name, predictor, *, registry_entry, registry_state):
    """C254: the arm ``name`` is measured in, derived from its COMMITTED registry block. No ``predictor``: served as
    committed (``served_on`` / ``served_off``); ``off``: an MTP-ON entry gets a derived registry without its MTP keys
    (``mtp_off``); ``on``: ``served_on`` as committed, or ``mtp_on_restored`` for an OFF entry the 16-state generator
    restores. ``registry_state`` is ``registry_states.classify``'s result, used only to decide a restore.
    """
    on = serves_mtp(registry_entry)
    if predictor is None:
        return "served_on" if on else "served_off"
    if predictor == "off":
        return "mtp_off" if on else "served_off"
    if on:
        return "served_on"
    if name not in registry_states.ENTRY_OFF:
        raise ValueError(f"{name}: no MTP drafter to restore")
    if registry_state is None:
        raise ValueError(
            f"{name}: the registry is not one of the 16 accepted states (a restore needs the pinned fixture)"
        )
    if name not in registry_state:
        raise ValueError(
            f"{name}: no MTP drafter to restore (the registry does not classify "
            "it as MTP-OFF)"
        )
    return "mtp_on_restored"


def default_entries(registry_entries):
    """The registry's entries that are P369 picks (the MTP-certified ones), in registry order."""
    from p369_phase2 import orchestrate

    return [n for n in registry_entries if n in orchestrate.PICKS]


def mtp_off_registry(text, name):
    """``text`` with exactly ``name``'s MTP-key lines removed (its block's span, every other byte unchanged). Refuses when
    the result is not the entry's settings minus those keys with everything else equal.
    """
    import yaml

    old = registry_states.parse(text)  # the strict parse first
    lines = text.split("\n")
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    models = next(v for k, v in root.value if getattr(k, "value", None) == "models")
    items = [i for i in models.value if registry_states._node_name(i) == name]
    if len(items) != 1:
        raise ValueError(f"{name}: {len(items)} registry blocks")
    item = items[0]
    for key in _merged_mtp_keys(item):  # inherited keys have no line of the entry's own
        raise ValueError(
            f"{name}: {key.value} (line {key.start_mark.line + 1}): inherited "
            "through a merge key; it cannot be removed line-wise"
        )
    drop = set()
    for key, value in item.value:
        if getattr(key, "value", None) not in MTP_KEYS:
            continue
        where = f"{name}: {key.value} (line {key.start_mark.line + 1})"
        if item.flow_style:
            raise ValueError(f"{where}: a flow-style entry block cannot be edited")
        first = key.start_mark.line
        last = value.end_mark.line - (1 if value.end_mark.column == 0 else 0)
        tail = (
            lines[last][value.end_mark.column :] if value.end_mark.line == last else ""
        ).strip()
        if lines[first][: key.start_mark.column].strip() or (
            tail and not tail.startswith("#")
        ):
            raise ValueError(f"{where}: shares its line with other content")
        drop.update(range(first, last + 1))
    derived = "\n".join(l for i, l in enumerate(lines) if i not in drop)
    new = registry_states.parse(derived)
    want = dict(
        old,
        models=[_without_mtp(m) if m["name"] == name else m for m in old["models"]],
    )
    if new != want:
        keys = [k for k in MTP_KEYS if k in registry_states.settings(text)[name]]
        raise ValueError(
            f"{name}: the MTP keys {keys} could not be removed textually "
            "(the result differs from the entry minus those keys)"
        )
    return derived


def _merged_mtp_keys(node, seen=None):
    """The MTP key nodes a mapping node inherits through its ``<<`` merge keys (recursively), in source order."""
    seen = seen if seen is not None else set()
    found = []
    for key, value in node.value:
        if getattr(key, "tag", None) != "tag:yaml.org,2002:merge":
            continue
        sources = value.value if str(value.tag).endswith(":seq") else [value]
        for source in sources:
            if id(source) in seen or not isinstance(source.value, list):
                continue
            seen.add(id(source))
            found += [
                k for k, _ in source.value if getattr(k, "value", None) in MTP_KEYS
            ]
            found += _merged_mtp_keys(source, seen)
    return found


def _without_mtp(entry):
    return {k: v for k, v in entry.items() if k not in MTP_KEYS}


def summary_arguments(raw, names):
    """``--p369-summary`` values -> (``{entry: path}``, problems): ``name=path`` (split on the first ``=``, the name one of
    the run's entries) or a bare path, which is for a single entry only."""
    if raw is None:
        raw = []
    elif isinstance(raw, (str, Path)):
        raw = [raw]
    mapping, problems = {}, []
    for value in map(str, raw):
        head, sep, rest = value.partition("=")
        if sep and head and (head in names or "/" not in head):
            key, path = head, rest  # an --entry name may itself contain "/"
            if key not in names:
                problems.append(
                    f"--p369-summary names {key!r}, which is not an --entry"
                )
                continue
        elif len(names) == 1:
            key, path = names[0], value
        else:
            problems.append(
                f"--p369-summary {value!r}: a bare path needs exactly one entry; use name=path"
            )
            continue
        if key in mapping:
            problems.append(f"--p369-summary given twice for {key!r}")
        mapping[key] = path
    return mapping, problems


def _production_matches(value, production, resolve):
    """The summary's production, resolved to a full fork sha (``resolve``), equals THIS run's resolved production."""
    if not (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{7,40}", value)):
        return False
    try:
        return resolve(value) == production
    except ValueError:
        return False


def certified_argv(summary_path, name):
    """The worker argv the entry's P369 run certified MTP-ON: ``picks[name].argv_on`` of the ``manifest.json`` beside the
    summary (None when absent or malformed)."""
    try:
        manifest = json.loads((Path(summary_path).parent / "manifest.json").read_text())
        argv = manifest["picks"][name]["argv_on"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    ok = isinstance(argv, list) and all(isinstance(a, str) for a in argv)
    return argv if ok else None


def argv_binding_problems(name, certified, argv):
    """The MTP-ON argv qualify derives must equal the certified one; the first differing flag is named."""
    if list(argv) == list(certified):
        return []
    try:
        want, got = _parse_flags(list(certified)), _parse_flags(list(argv))
    except ValueError as exc:
        return [f"{name}: {CERTIFIED_DIFFERS} (argv: {exc})"]
    first = next((f for f, v in want.items() if got.get(f) != v), None)
    first = first or next((f for f in got if f not in want), "argument order")
    return [f"{name}: {CERTIFIED_DIFFERS} ({first})"]


CERTIFIED_DIFFERS = (
    "the served configuration differs from the one its P369 run certified"
)


def entry_problems(
    names,
    *,
    registry_text,
    predictor=None,
    summaries=None,
    production,
    resolve,
    certified=None,
    classify=lambda: None,
    artifact=None,
):
    """C254 refusals, all before the run: a name outside the registry (listed), an arm without a drafter to restore, a
    registry that cannot give the restore, and every MTP-ON arm (C199, generalized) without a P369 FULL-run PASS for exactly
    that entry on THIS run's production. ``production`` is a full sha and ``resolve(ref)`` resolves a summary's ref to one
    (ValueError when it cannot); ``certified`` is ``{entry: the P369 run's argv_on or None}``, read once by the caller;
    ``classify()`` supplies the 16-state result (called only for a restore);
    ``artifact(path, summary)`` is the campaign-artifact check.
    """
    from p369_phase2 import orchestrate

    entries = registry_states.settings(registry_text)
    summaries = summaries or {}
    problems = []
    for name in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"duplicate --entry {name!r}")
    for name in names:
        if name not in entries:
            problems.append(
                f"unknown entry {name!r}: the registry's entries are {list(entries)}"
            )
            continue
        try:
            restore = predictor == "on" and not serves_mtp(entries[name])
            arm = arm_for(
                name,
                predictor,
                registry_entry=entries[name],
                registry_state=classify() if restore else None,
            )
            if arm == "mtp_off":
                mtp_off_registry(registry_text, name)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if arm not in MTP_ON_ARMS:
            continue
        if name not in orchestrate.PICKS:
            problems.append(
                f"{name}: MTP-ON needs the entry in orchestrate.PICKS (its drafter shape for the draft-kernel check)"
            )
            continue
        gate = f"{name}: MTP-ON only after its P369 full-run PASS (C199)"
        path = summaries.get(name)
        try:
            summary = json.loads(Path(path).read_text()) if path else None
        except (OSError, ValueError):
            summary = None
        if not isinstance(summary, dict):
            problems.append(gate)
            continue
        pick = (summary.get("picks") or {}).get(name) or {}
        binding = summary.get("binding") or {}
        manifest = summary.get("manifest") or {}
        passed = (
            summary.get("phase") == "full"
            and summary.get("status") == "PASS"
            and (pick.get("outcome") or {}).get("e1c") == "PASS"
            and (pick.get("e2") or {}).get("verdict") == "PASS"
            # code round 1 D10: the run's own provenance — THIS run's production and a bound reviewed head
            and _production_matches(binding.get("production"), production, resolve)
            and _production_matches(manifest.get("production"), production, resolve)
            and bool(binding.get("reviewed_head"))
            and binding.get("reviewed_head") == binding.get("fork_head")
            and list(summary.get("picks") or {}) == [name]
        )
        if not passed:
            problems.append(gate)
            continue
        if artifact is not None:
            problems += artifact(path, summary)  # code round 2 D10
        if (certified or {}).get(name) is None:
            problems.append(
                f"{name}: its P369 manifest.json has no picks[{name}].argv_on"
            )
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
    defaults and the prompt size (C198: cap − MIN_OUTPUT_TOKENS). A value that cannot be converted (``null`` for an
    integer key, say) is a ``ValueError`` naming the entry (v1.4), never a TypeError."""
    try:
        return _expected_config(entry)
    except UnknownKey:
        raise
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{entry.get('name')}: {exc}") from exc


def _expected_config(entry):
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


def _arm_registry(arm, name, registry, stack, fork, out, orchestrate):
    """``(path, text)`` of the registry an arm serves, from the snapshot ``registry``: the committed file (``served_*``),
    the entry's MTP keys removed (``mtp_off``, C254) or the 16-state overlay (``mtp_on_restored``, C199); the derived
    ones are written under ``out`` (safe ids)."""
    if arm in ("served_on", "served_off"):
        return stack / "main_models.yaml", registry
    if arm == "mtp_off":
        text = mtp_off_registry(registry, name)
        path = out_path(out, f"mtp-off-{safe_id(name)}.yaml")
    else:
        fixture = registry_states.fixture_text(fork / "mlx_vlm" / "tests" / "fixtures")
        text, _ = orchestrate.overlay_registry(registry, name, fixture)
        path = out_path(out, f"overlay-{safe_id(name)}.yaml")
    path.write_bytes(text.encode("utf-8"))
    return path, text


def safe_id(name):
    """One safe path component for a registry name: ``[A-Za-z0-9._-]`` kept, everything else ``_``, plus the first 8 hex
    of sha256(name) so two names never collide."""
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return f"{clean}-{hashlib.sha256(name.encode('utf-8')).hexdigest()[:8]}"


def out_path(out, component):
    """``out / component``, which must resolve under ``out`` (else ValueError: refuse)."""
    path = Path(out) / component
    if not path.resolve().is_relative_to(Path(out).resolve()):
        raise ValueError(f"{path} is not under {out}")
    return path


def _full_sha(fork, ref):
    """``ref`` resolved to a full commit sha in the fork (ValueError when it is not a commit)."""
    import subprocess

    try:
        return _resolve_ref(fork, ref)
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ValueError(f"{ref!r} is not a commit in {fork}") from exc


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
    predictor = getattr(args, "predictor", None)
    lock = orchestrate.acquire_lock(orchestrate.lock_path(dict(os.environ)))
    try:  # code round 2 D13: the lock is released whatever the cleanup below does
        # C254 v1.2: the ONE registry snapshot, read right after the lock; the gate, the derived / restored registries,
        # the hashes and the execution all use exactly these bytes
        registry_text = orchestrate.read_registry(stack / "main_models.yaml")
        start = _tree_state(fork, stack)
        try:
            registry_entries = registry_states.settings(registry_text)  # strict
        except ValueError as exc:
            raise SystemExit(f"refused: ['the registry does not parse: {exc}']")
        names = list(args.entry or default_entries(list(registry_entries)))
        summaries, problems = summary_arguments(args.p369_summary, names)
        reviewed = _resolve_ref(fork, args.reviewed_head)
        production = getattr(args, "production_ref", None) or orchestrate.PRODUCTION
        try:  # P452 F3 / C254: one resolved FULL sha, diffed by the preflight and recorded in the report
            production = args.production_ref = orchestrate.resolve_production_ref(
                fork, production, reviewed
            )
        except ValueError as exc:
            raise SystemExit(f"refused: {exc}")
        classified = []

        def classify():  # only a restore needs the pinned fixture
            if not classified:
                fixture = registry_states.fixture_text(
                    fork / "mlx_vlm" / "tests" / "fixtures"
                )
                classified.append(registry_states.classify(registry_text, fixture))
            return classified[0]

        certified = {n: certified_argv(p, n) for n, p in summaries.items()}  # once
        problems += entry_problems(
            names,
            registry_text=registry_text,
            predictor=predictor,
            summaries=summaries,
            production=production,
            resolve=lambda ref: _full_sha(fork, ref),
            certified=certified,
            classify=classify,
            artifact=lambda path, summary: summary_artifact_problems(
                path,
                summary,
                root=Path(os.environ.get("STACK_WORKDIR", "/nonexistent"))
                / "p369"
                / "phase2",
                is_commit=lambda c: _is_commit(fork, c),
                contains_reviewed=lambda c: _is_ancestor(fork, V13_REVIEWED, c),
            ),
        )
        if problems:
            raise SystemExit(f"refused: {problems}")
        arms = {
            n: arm_for(
                n,
                predictor,
                registry_entry=registry_entries[n],
                registry_state=(
                    classify()
                    if predictor == "on" and not serves_mtp(registry_entries[n])
                    else None
                ),
            )
            for n in names
        }
        out.mkdir(parents=True, exist_ok=False)
        commands, problems = {}, []
        for (
            n
        ) in (
            names
        ):  # C254 v1.2 / v1.4: the MTP-ON argv is bound to the certified one BEFORE the run
            if arms[n] not in MTP_ON_ARMS:
                continue
            try:
                path, _ = _arm_registry(
                    arms[n], n, registry_text, stack, fork, out, orchestrate
                )
                commands[n] = full_command(stack, path, n)
            except Exception as exc:
                problems.append(f"{n}: cannot derive the served argv: {exc}")
                continue
            problems += argv_binding_problems(n, certified[n], commands[n][0][1:])
        if problems:
            raise SystemExit(f"refused: {problems}")
        snapshot = dict(
            text=registry_text,
            sha256=hashlib.sha256(registry_text.encode("utf-8")).hexdigest(),
            start=start,
            certified=certified,  # read once; the gate and the binding use this value
            commands=commands,
        )
        return _drive_locked(
            args,
            names,
            arms,
            snapshot,
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
    arms,
    snapshot,
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
        pre = orchestrate.preflight(args, out)
        if pre["problems"]:
            raise SystemExit(f"refused: {pre['problems']}")
        start = snapshot[
            "start"
        ]  # taken with the registry snapshot: a moved stack HEAD refuses
        problems = guard_problems(start, _tree_state(fork, stack), reviewed=reviewed)
        registry, registry_sha = snapshot["text"], snapshot["sha256"]
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
                    path, text = _arm_registry(
                        arms[name], name, registry, stack, fork, out, orchestrate
                    )
                    if arms[name] not in ("served_on", "served_off"):
                        state["overlay_sha256"] = hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest()
                    entry = registry_states.settings(text)[name]
                    expected = expected_config(entry)
                    # MTP-ON arms: the argv was derived and bound to the certified one before the run (v1.4)
                    cmd, module_file = snapshot["commands"].get(name) or full_command(
                        stack, path, name
                    )
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
            if rec:  # C254: capacity AND speed per arm, whatever the entry's status
                final.update(arm=arms[name], rates=rec.get("rates"))
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
        out_dir=str(out_path(out, safe_id(name))),
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
    ap.add_argument(  # C254: a bare path (one entry) or name=path, once per MTP-ON entry
        "--p369-summary", action="append"
    )
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
