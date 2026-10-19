"""P369 phase 2 worker: ONE (pick, arm) job in a fresh, sanitized subprocess (Q20).

Usage (by the orchestrator only): ``python -I worker.py <job.json>``. The job names the worker argv (from
mlx-serve's own ``_build_command``), the reviewed source root, the expected arm state and the requests. The
worker runs the fork's real CLI (``mlx_vlm.server.cli.main``) with ``uvicorn.run`` intercepted, serves through
``TestClient`` (lifespan preload; the real endpoint, templating, cached-session path), asserts what it serves,
runs the job's mode and writes ``result.json`` atomically (plus per-request ``.npz`` row / commit files).
Claims are WORKER conformance; the mlx-serve router layer is not exercised.

Modes: ``size`` (dry-run prompt sizing through the endpoint, no generation), ``count`` (the endpoint's
prompt token count of given messages, no generation), ``e1c_on``, ``e1c_off`` (the
forced clean replay, then the comparison against the ON artifacts), ``e2`` (greedy identity material),
``e7a`` / ``e7b`` (event observation; ``e7b`` continues one chat id across scripted turns).
"""

import json
import os
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

FILLER = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
NUDGES = list("abcdefghijklmnopqrstuvwxyz0123456789")


class DryRun(BaseException):
    """Raised inside the endpoint's tokenization to measure a prompt without generating (a BaseException,
    so no generic ``except Exception`` handler on the way out can swallow it)."""

    def __init__(self, tokens):
        super().__init__(tokens)
        self.tokens = tokens


# ------------------------------------------------------------------ provenance / state checks
def check_sources(root, dev_dir):
    """Every loaded ``mlx_vlm`` module must come from the reviewed tree, ``p369_phase2`` from ``dev_dir``."""
    root, dev_dir = Path(root).resolve(), Path(dev_dir).resolve()
    bad = []
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            if not Path(path).resolve().is_relative_to(root):
                bad.append((name, path))
        if name == "p369_phase2" or name.startswith("p369_phase2."):
            if not Path(path).resolve().is_relative_to(dev_dir):
                bad.append((name, path))
    if bad:
        raise RuntimeError(f"modules outside the reviewed sources: {bad[:5]}")


def atomic_write(path, payload):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


AFTER_REQUEST_S = 900.0  # release / idle wait / the next request's setup / shutdown


def write_progress(out_dir, *, index, name, deadline_s, phase="request"):
    """Code round 1 H8: the heartbeat the orchestrator's supervisor enforces — which request (or the
    ``after`` phase following it) is running, since when, and its deadline."""
    atomic_write(
        Path(out_dir) / "progress.json",
        dict(
            index=index,
            name=name,
            phase=phase,
            started=time.time(),
            deadline_s=float(deadline_s),
        ),
    )


def boot(argv):
    """Run the fork's CLI on ``argv`` with ``uvicorn.run`` intercepted; return a (not yet entered) TestClient."""
    import uvicorn

    captured = {}
    real = uvicorn.run
    uvicorn.run = lambda app, **kwargs: captured.update(app=app, **kwargs)
    old_argv = sys.argv
    sys.argv = ["mlx_vlm.server", *argv]
    try:
        from mlx_vlm.server import cli

        cli.main()
    finally:
        uvicorn.run = real
        sys.argv = old_argv
    if captured.get("app") != "mlx_vlm.server:app":
        raise RuntimeError(f"cli.main did not hand uvicorn the app: {captured}")
    import importlib

    from fastapi.testclient import TestClient

    module, _, attr = captured["app"].partition(":")
    return TestClient(getattr(importlib.import_module(module), attr))


def served_state():
    import mlx_vlm.server as server
    from mlx_vlm.server import session_manager

    rg = server.runtime.response_generator
    policy = getattr(rg, "mtp_verify_policy", None)
    return dict(
        drafter_loaded=getattr(rg, "draft_model", None) is not None,
        draft_kind=getattr(rg, "draft_kind", None),
        verify_policy=getattr(policy, "name", None),
        apc_manager=server.runtime.apc_manager is not None,
        apc_env=os.environ.get("APC_ENABLED"),
        session_max=getattr(session_manager, "_session_cache_max", None),
        session_max_env=os.environ.get("MLX_VLM_CACHE_SESSION_MAX"),
    )


def assert_served(state, expect):
    problems = []
    if state["drafter_loaded"] != expect["drafter"]:
        problems.append(
            f"drafter_loaded={state['drafter_loaded']} expected {expect['drafter']}"
        )
    if expect["drafter"] and state["draft_kind"] != "mtp":
        problems.append(f"draft_kind={state['draft_kind']}")
    if state["verify_policy"] != expect.get("verify_policy"):
        problems.append(
            f"verify_policy={state['verify_policy']} expected {expect.get('verify_policy')}"
        )
    if state["apc_manager"] or state["apc_env"] is not None:
        problems.append("APC is active")
    if state["session_max"] != 1 or state["session_max_env"] != "1":
        problems.append(
            f"session max {state['session_max']} / env {state['session_max_env']}"
        )
    if problems:
        raise RuntimeError("served state mismatch: " + "; ".join(problems))


# ------------------------------------------------------------------ request plumbing
class Capture:
    """Final chunk (token ids, finish, counts) and backend of each request, from the production calls."""

    def __init__(self):
        self.final = None
        self.backend = None
        self.completed = (
            0  # Q38: exits of _process_cached_request seen on the GPU thread
        )
        self._patched = []

    def __enter__(self):
        from mlx_vlm.server import generation

        capture = self
        stream = sys.modules["mlx_vlm.generate"].stream_generate

        def capturing_stream(*args, **kwargs):
            last = None
            for chunk in stream(*args, **kwargs):
                last = chunk
                yield chunk
            if last is not None:
                capture.final = dict(
                    token_ids=[
                        int(t) for t in (getattr(last, "token_ids", None) or [])
                    ],
                    finish_reason=getattr(last, "finish_reason", None),
                    generation_tokens=getattr(last, "generation_tokens", None),
                    prompt_tokens=getattr(last, "prompt_tokens", None),
                    cached_tokens=getattr(last, "cached_tokens", None),
                )

        generate_fn = generation.ResponseGenerator.generate

        def capturing_generate(rg, *args, **kwargs):
            ctx, tokens = generate_fn(rg, *args, **kwargs)
            capture.backend = getattr(ctx, "backend", None)
            return ctx, tokens

        process = generation.ResponseGenerator._process_cached_request

        def acknowledged_process(rg, *args, **kwargs):
            try:
                return process(rg, *args, **kwargs)
            finally:
                capture.completed += 1

        self._patched += [
            (sys.modules["mlx_vlm.generate"], "stream_generate", stream),
            (generation.ResponseGenerator, "generate", generate_fn),
            (generation.ResponseGenerator, "_process_cached_request", process),
        ]
        sys.modules["mlx_vlm.generate"].stream_generate = capturing_stream
        generation.ResponseGenerator.generate = capturing_generate
        generation.ResponseGenerator._process_cached_request = acknowledged_process
        return self

    def reset(self):
        self.final, self.backend = None, None

    def __exit__(self, *exc):
        while self._patched:
            owner, name, value = self._patched.pop()
            setattr(owner, name, value)
        return False


def wait_ack(capture, before, *, timeout_s=60.0, poll_s=0.01):
    """Q38: the GPU thread's ``_process_cached_request`` for this request has returned."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if capture.completed > before:
            return
        time.sleep(poll_s)
    raise RuntimeError("the cached request did not acknowledge its exit")


def release_trajectory(recorder, chat_id):
    """Q35: retire one trajectory's session and cache tree BEFORE the next trajectory starts — clear the
    chat's ``PromptCacheState``, drop the session entry and the recorder's reference, and prove with weak
    references that every leaf cache was collected."""
    import gc
    import weakref

    from mlx_vlm.speculative.cache_state import iter_leaf_caches

    caches = recorder._cache
    refs = [weakref.ref(leaf) for leaf in iter_leaf_caches(caches or [])]
    release_session(chat_id)
    recorder._cache = None
    del caches
    gc.collect()
    alive = sum(1 for r in refs if r() is not None)
    if alive:
        raise RuntimeError(
            f"trajectory {chat_id}: cache tree not retired ({alive} leaves alive)"
        )


def release_session(chat_id):
    """Drop the chat's production session entry (``{"state", "last_used", "turn_hashes"}``) and clear its
    ``PromptCacheState`` (cache, token ids, snapshot ring); any other entry shape refuses.
    """
    from mlx_vlm.server import session_manager

    with session_manager._session_caches_lock:
        entry = session_manager._session_caches.pop(chat_id, None)
    if entry is None:
        return
    if not isinstance(entry, dict) or "state" not in entry:
        raise RuntimeError(
            f"unexpected session entry for {chat_id}: {type(entry).__name__}"
        )
    entry["state"].clear()


def post(client, payload, chat_id, capture, timeout_s):
    capture.reset()
    before = capture.completed
    started = time.monotonic()
    r = client.post(
        "/v1/chat/completions",
        headers={"X-MLX-VLM-Chat-Id": chat_id},
        json=payload,
        timeout=timeout_s,
    )
    elapsed = time.monotonic() - started
    if r.status_code != 200:
        raise RuntimeError(f"request failed {r.status_code}: {r.text[:500]}")
    wait_ack(capture, before)
    if capture.backend != "cached_session":
        raise RuntimeError(f"backend {capture.backend!r}, expected cached_session")
    if capture.final is None:
        raise RuntimeError("no final chunk captured")
    return dict(
        response=r.json(), final=dict(capture.final), wall_s=elapsed, chat_id=chat_id
    )


@contextmanager
def probe_counters_frozen(policy):
    """Probes are verify forwards too: keep the verify-scan counters exactly as production left them."""
    saved = (
        dict(policy._counts)
        if policy is not None and hasattr(policy, "_counts")
        else None
    )
    try:
        yield
    finally:
        if saved is not None:
            policy._counts.clear()
            policy._counts.update(saved)


# ------------------------------------------------------------------ prompt sizing (dry runs)
def prompt_tokens(client, messages, base_payload):
    """The endpoint's own prompt token count for ``messages`` (no generation)."""
    from mlx_vlm.server import generation

    real = generation.ResponseGenerator._preprocess_request

    def measuring(rg, *args, **kwargs):
        raw = real(rg, *args, **kwargs)
        raise DryRun(generation._count_prompt_tokens(raw))

    generation.ResponseGenerator._preprocess_request = measuring
    try:
        client.post(
            "/v1/chat/completions",
            headers={"X-MLX-VLM-Chat-Id": f"p369-size-{uuid.uuid4().hex}"},
            json=dict(base_payload, messages=messages, max_tokens=1),
        )
    except DryRun as stop:
        return stop.tokens
    finally:
        generation.ResponseGenerator._preprocess_request = real
    raise RuntimeError("dry run reached generation")


def size_prompt(count, target, *, prefix, suffix):
    """Find a user text ``prefix + filler + suffix`` whose templated count is exactly ``target``.

    ``count(text) -> int``. Binary search over filler words, then single-character nudges. Returns
    ``(text, tokens)`` or ``(None, closest)``."""

    def text_for(k, nudge=""):
        words = [FILLER[i % len(FILLER)] for i in range(k)]
        return " ".join([prefix, *words, nudge, suffix]).replace("  ", " ")

    lo, hi = 0, max(16, target)
    while count(text_for(hi)) < target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if count(text_for(mid)) < target:
            lo = mid + 1
        else:
            hi = mid
    best = None
    for k in (lo, lo - 1, lo - 2):
        if k < 0:
            continue
        for nudge in [""] + NUDGES + [n + m for n in NUDGES[:6] for m in NUDGES[:6]]:
            text = text_for(k, nudge)
            got = count(text)
            if got == target:
                return text, got
            if best is None or abs(got - target) < abs(best - target):
                best = got
    return None, best


# ------------------------------------------------------------------ artifacts
def save_rows(path, records):
    import numpy as np

    positions = np.array([p for p, _ in records], dtype=np.int64)
    dtypes = sorted({r["dtype"] for _, r in records})
    if len(dtypes) > 1:
        raise RuntimeError(f"mixed row dtypes {dtypes}")
    bits = np.stack([r["bits"] for _, r in records]) if records else np.zeros((0, 0))
    np.savez(
        path,
        positions=positions,
        bits=bits,
        dtype=np.array(dtypes[0] if dtypes else ""),
    )


def load_rows(path):
    import numpy as np

    data = np.load(path)
    dtype = str(data["dtype"])
    return [
        (int(p), dict(dtype=dtype, bits=data["bits"][i]))
        for i, p in enumerate(data["positions"])
    ]


def save_snapshots(path, snapshots):
    import numpy as np

    probes = [s["probe"] for s in snapshots if s["probe"] is not None]
    np.savez(
        path,
        offsets=np.array([s["offset"] for s in snapshots], dtype=np.int64),
        shas=np.array([s["sha"] for s in snapshots]),
        neutral=np.array([s["neutral"] for s in snapshots]),
        kinds=np.array([s["kind"] for s in snapshots]),
        probe_bits=(
            np.stack([p["bits"] for p in probes]) if probes else np.zeros((0, 0))
        ),
        probe_dtype=np.array(probes[0]["dtype"] if probes else ""),
    )


def load_snapshots(path):
    import numpy as np

    d = np.load(path)
    dtype = str(d["probe_dtype"])
    out = []
    for i, offset in enumerate(d["offsets"]):
        probe = (
            dict(dtype=dtype, bits=d["probe_bits"][i]) if len(d["probe_bits"]) else None
        )
        out.append(
            dict(
                offset=int(offset),
                sha=str(d["shas"][i]),
                neutral=bool(d["neutral"][i]),
                kind=str(d["kinds"][i]),
                probe=probe,
            )
        )
    return out


# ------------------------------------------------------------------ smoke positives (Q32)
def _replace_last(records, position, new):
    out = list(records)
    index = max(i for i, (p, _) in enumerate(out) if p == position)
    out[index] = (position, new)
    return out


def _mutate_and_restore(caches, array, index, sha):
    """Change one element of a live cache array, hash, restore it exactly; True when the change was seen."""
    from p369_phase2 import observe

    saved = array[index]
    array[index] = saved + 1
    seen = observe.state_hash(caches) != sha
    array[index] = saved
    if observe.state_hash(caches) != sha:
        raise RuntimeError("smoke positive: the restored cache does not hash back")
    return seen


def smoke_positives(on, off, n, params):
    """Q32 on live data: the unmodified ON / OFF data pass, and each injected change is detected."""
    import copy

    import numpy as np
    from p369_phase2 import compare, observe

    rows_on, rows_off = on.rows, off.rows
    base_rows = compare.compare_streams(rows_on, rows_off, n, params)
    base_states = observe.compare_states(on, off)
    fired = {}
    latest, _ = compare.latest_rows(rows_off, n)
    position = max(latest)
    record = latest[position]
    canon = compare.canonical(record, params)
    # raw ulp, outside the support: the raw recorder sees it, the filtered distribution cannot.
    j = int(np.flatnonzero(canon == 0)[0])
    bits = np.array(record["bits"]).copy()
    bits[j] = bits[j] + 1
    raw = compare.compare_streams(
        rows_on, _replace_last(rows_off, position, dict(record, bits=bits)), n, params
    )
    fired["raw_ulp"] = raw["raw_only_differences"] == [position] and raw["pass"]
    # in support: the comparator must fail exactly there.
    row = compare.from_record(record)
    top = int(np.argmax(canon))
    import mlx.core as mx

    bumped = mx.where(mx.arange(row.size) == top, row + 0.5, row).astype(row.dtype)
    bad = compare.compare_streams(
        rows_on, _replace_last(rows_off, position, compare.as_record(bumped)), n, params
    )
    fired["in_support"] = (not bad["pass"]) and [
        m["position"] for m in bad["mismatches"]
    ] == [position]
    # live cache components (the final OFF cache): each change must move the full-state hash.
    from mlx_vlm.speculative.cache_state import iter_leaf_caches

    caches = off._cache
    sha = observe.state_hash(caches)
    leaves = list(iter_leaf_caches(caches))
    tq = sys.modules.get("mlx_vlm.turboquant")
    for leaf in leaves:
        kind = type(leaf).__name__
        if tq is not None and isinstance(leaf, tq.TurboQuantKVCache):
            if "tq_norm" not in fired and leaf.keys is not None:
                norms = leaf.keys[0]
                fired["tq_norm"] = _mutate_and_restore(caches, norms, (..., 0), sha)
        elif hasattr(leaf, "keys") and getattr(leaf, "keys", None) is not None:
            if "old_prefix_k" not in fired and not isinstance(leaf.keys, (list, tuple)):
                fired["old_prefix_k"] = _mutate_and_restore(
                    caches, leaf.keys, (..., 0, 0), sha
                )
        elif kind == "ArraysCache" and "recurrent" not in fired:
            arrays = [a for a in leaf.state if a is not None]
            fired["recurrent"] = _mutate_and_restore(caches, arrays[-1], (..., 0), sha)
    # probe logit: one ON snapshot's probe changed must fail the state comparison there.
    if on.commits and on.commits[0]["probe"] is not None:
        changed = copy.copy(on)
        changed.commits = [dict(c) for c in on.commits]
        probe = dict(changed.commits[0]["probe"])
        probe_bits = np.array(probe["bits"]).copy()
        probe_bits[0] = probe_bits[0] + 1
        changed.commits[0]["probe"] = dict(probe, bits=probe_bits)
        states = observe.compare_states(changed, off)
        fired["probe_logit"] = (not states["pass"]) and states["probe_mismatches"] == [
            changed.commits[0]["offset"]
        ]
    return dict(
        baseline_pass=bool(base_rows["pass"] and base_states["pass"]), fired=fired
    )


# ------------------------------------------------------------------ main
def main(job_path):
    job = json.loads(Path(job_path).read_text())
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, job["dev_dir"])
    started = time.time()
    result = dict(job=job["name"], mode=job["mode"], started=started, requests=[])
    try:
        run_job(job, out_dir, result)
        result["status"] = "ok"
    except BaseException as exc:  # recorded, then the process exits nonzero
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write(out_dir / "result.json", result)
        raise
    result["finished"] = time.time()
    atomic_write(out_dir / "result.json", result)


@contextmanager
def drafter_detached(rg):
    """P391: the forced OFF replay on the SAME loaded target — the cached path reads ``draft_kind`` /
    ``draft_model`` per request (``_inline_draft_kwargs``), so ``None`` there is plain decoding.
    """
    saved = (rg.draft_model, rg.draft_kind)
    rg.draft_model, rg.draft_kind = None, None
    try:
        yield
    finally:
        rg.draft_model, rg.draft_kind = saved


def wait_idle(rg, metrics, *, timeout_s=60.0, poll_s=0.05):
    """Q38: the request queue is empty and no endpoint request is in flight (the GPU thread is idle)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if rg.requests.qsize() == 0 and getattr(metrics, "_in_flight", 0) == 0:
            return
        time.sleep(poll_s)
    raise RuntimeError("the worker did not go idle")


class DetachedGuard:
    """Q38: during a detached replay ``_inline_draft_kwargs`` must return ``{}`` and no speculative prefill
    capture may be requested; any violation is recorded (the job then fails)."""

    def __init__(self):
        self.problems = []
        self._patched = []

    def __enter__(self):
        from mlx_vlm.generate import ar
        from mlx_vlm.server import generation

        guard = self
        inline, prefill = generation._inline_draft_kwargs, ar.speculative_prefill_kwargs

        def guarded_inline(*args, **kwargs):
            out = inline(*args, **kwargs)
            if out:
                guard.problems.append(f"draft kwargs returned {sorted(out)}")
            return out

        def guarded_prefill(*args, **kwargs):
            guard.problems.append("speculative prefill kwargs requested")
            return prefill(*args, **kwargs)

        self._patched = [
            (generation, "_inline_draft_kwargs", inline),
            (ar, "speculative_prefill_kwargs", prefill),
        ]
        generation._inline_draft_kwargs = guarded_inline
        ar.speculative_prefill_kwargs = guarded_prefill
        return self

    def __exit__(self, *exc):
        while self._patched:
            owner, name, value = self._patched.pop()
            setattr(owner, name, value)
        return False


def c1_acceptance(on_part, off_parts, c1, *, retired, expected_offsets, replay_offsets):
    """Q37: the within-instance control is complete and identical — every replay emitted exactly the forced
    tokens with ON's finish reason and no reused prefix, each replay snapshotted EXACTLY the ON offsets
    (``expected_offsets``; code round 1 H4: a recorder gap common to both replays is a C1 failure, never
    an E1c FAIL), each trajectory's cache tree was retired before the next one started (Q35; ``retired``,
    from ``release_trajectory``), and OFF1 vs OFF2 rows (raw + transformed) and states (hashes + probes)
    are equal.
    """
    problems = []
    for k, offsets in enumerate(replay_offsets, 1):
        if sorted(offsets) != sorted(expected_offsets):
            missing = sorted(set(expected_offsets) - set(offsets))
            extra = sorted(set(offsets) - set(expected_offsets))
            problems.append(f"OFF{k} offsets: missing {missing}, extra {extra}")
    want = on_part["final"]["token_ids"]
    for k, part in enumerate(off_parts, 1):
        final = part["final"]
        if final["token_ids"] != want:
            problems.append(f"OFF{k} did not emit exactly the forced tokens")
        if final["finish_reason"] != on_part["final"]["finish_reason"]:
            problems.append(f"OFF{k} finish {final['finish_reason']} != ON")
        if final.get("cached_tokens"):
            problems.append(f"OFF{k} reused {final['cached_tokens']} cached tokens")
    if on_part["final"].get("cached_tokens"):
        problems.append(f"ON reused {on_part['final']['cached_tokens']} cached tokens")
    if not retired:
        problems.append("trajectory cache trees were not retired in turn")
    if not c1["rows"]["pass"]:
        problems.append("C1 rows differ")
    if not c1["states"]["pass"]:
        problems.append("C1 states differ")
    return dict(problems=problems, **{"pass": not problems})


def snapshot_holder(snaps, rows=None):
    """A comparison reference from saved snapshots: ``prompt_end`` (if recorded) and the rest as commits."""
    from types import SimpleNamespace

    first = snaps[0] if snaps and snaps[0]["kind"] == "prompt_end" else None
    return SimpleNamespace(
        prompt_end=first,
        commits=[s for s in snaps if s is not first],
        rows=rows,
    )


def e1c_request(
    client, capture, lm, policy, manifest, spec, mode, timeout, **rec_kwargs
):
    """One E1c request: the recorder (ON or forced OFF), kernel telemetry, key-length spans."""
    from p369_phase2 import observe, telemetry

    tel = telemetry.KernelTelemetry(lm)
    recorder = observe.Recorder(mode, lm=lm, **rec_kwargs)
    real_probe = observe.probe

    def frozen_probe(lm_, caches, token):
        with probe_counters_frozen(policy), tel.probe():
            return real_probe(lm_, caches, token)

    observe.probe = frozen_probe
    try:
        with recorder, tel, observe.KeyLengths() as key_lengths:
            got = post(
                client, spec["payload"], f"p369-{uuid.uuid4().hex}", capture, timeout
            )
    finally:
        observe.probe = real_probe
    part = dict(got)
    part["telemetry"] = tel.summary()
    part["verify_scope_calls"] = sum(1 for c in tel.calls if c.get("scope") == "verify")
    part["verdict"] = telemetry.verdict(tel, manifest)
    part["offsets"] = (
        [recorder.prompt_end["offset"]] if recorder.prompt_end else []
    ) + [c["offset"] for c in recorder.commits]
    part["prompt_end"] = recorder.prompt_end["offset"] if recorder.prompt_end else None
    part["draft_rounds"] = recorder.draft_rounds
    spans = sorted({(b["first_key"], b["last_key"]) for b in key_lengths.blocks})
    part["key_spans"] = dict(
        blocks=len(spans),
        first=spans[0] if spans else None,
        last=spans[-1] if spans else None,
    )
    threshold = spec.get("threshold")
    part["straddle"] = (
        observe.straddles(key_lengths.blocks, threshold) if threshold else None
    )
    return part, recorder


def compare_pair(ref, recorder, n, params):
    """Rows and states of a forced OFF ``recorder`` against a reference (ON recorder or holder)."""
    from p369_phase2 import compare, observe

    return dict(
        rows=compare.compare_streams(ref.rows, recorder.rows, n, params),
        states=observe.compare_states(ref, recorder),
    )


def run_job(job, out_dir, result):
    from p369_phase2 import telemetry

    client = boot(job["argv"])
    with client, ExitStack() as stack:
        state = served_state()
        result["served"] = state
        assert_served(state, job["expect"])
        check_sources(job["source_root"], job["dev_dir"])
        import mlx_vlm.server as server

        rg = server.runtime.response_generator
        lm = rg.model.language_model
        policy = getattr(rg, "mtp_verify_policy", None)
        manifest = telemetry.manifest_from_model(lm)
        result["manifest"] = manifest
        capture = stack.enter_context(Capture())
        if job["mode"] == "size":
            for index, spec in enumerate(job["requests"]):
                write_progress(
                    out_dir,
                    index=index,
                    name=spec["name"],
                    deadline_s=spec["timeout_s"],
                )
                count = lambda text: prompt_tokens(  # noqa: E731
                    client, [dict(role="user", content=text)], spec["payload"]
                )
                text, tokens = size_prompt(
                    count, spec["target"], prefix=spec["prefix"], suffix=spec["suffix"]
                )
                result["requests"].append(
                    dict(name=spec["name"], text=text, tokens=tokens)
                )
            return
        if job["mode"] == "count":
            for index, spec in enumerate(job["requests"]):
                write_progress(
                    out_dir,
                    index=index,
                    name=spec["name"],
                    deadline_s=spec["timeout_s"],
                )
                tokens = prompt_tokens(
                    client, spec["payload"]["messages"], spec["payload"]
                )
                result["requests"].append(dict(name=spec["name"], tokens=tokens))
            return
        probe = dict(probe_token=int(job["probe_token"]))
        params = job["params"]
        for index, spec in enumerate(job["requests"]):
            entry = dict(name=spec["name"], index=index)
            timeout = float(spec["timeout_s"])
            write_progress(out_dir, index=index, name=spec["name"], deadline_s=timeout)
            if job["mode"] == "e1c_on":
                part, rec = e1c_request(
                    client, capture, lm, policy, manifest, spec, "on", timeout, **probe
                )
                entry.update(part)
                save_rows(out_dir / f"rows-{index}.npz", rec.rows)
                save_snapshots(
                    out_dir / f"snaps-{index}.npz",
                    ([rec.prompt_end] if rec.prompt_end else []) + rec.commits,
                )
                wait_idle(rg, server.runtime.metrics)
                release_trajectory(rec, part["chat_id"])
            elif job["mode"] == "e1c_pair":
                part, on = e1c_request(
                    client, capture, lm, policy, manifest, spec, "on", timeout, **probe
                )
                entry.update(part)
                save_rows(out_dir / f"rows-{index}.npz", on.rows)
                save_snapshots(
                    out_dir / f"snaps-{index}.npz",
                    ([on.prompt_end] if on.prompt_end else []) + on.commits,
                )
                tokens, n = part["final"]["token_ids"], len(part["final"]["token_ids"])
                same = dict(model=id(rg.model), policy=id(policy))
                metrics = server.runtime.metrics
                replays = []
                wait_idle(rg, metrics)
                release_trajectory(
                    on, part["chat_id"]
                )  # Q35: before the next trajectory
                # Detached from before submission through worker idle (Q38); the same-instance OFF replay,
                # then control C1. Each trajectory is retired before the next starts (Q35).
                with drafter_detached(rg):
                    for k in range(2):
                        guard = DetachedGuard()
                        with guard:
                            off_part, off = e1c_request(
                                client,
                                capture,
                                lm,
                                policy,
                                manifest,
                                spec,
                                "off",
                                timeout,
                                force_tokens=tokens,
                                hash_offsets=part["offsets"],
                                **probe,
                            )
                            wait_idle(rg, metrics)
                        if guard.problems:
                            raise RuntimeError(f"detached replay: {guard.problems}")
                        if off_part["verify_scope_calls"] or off_part["draft_rounds"]:
                            raise RuntimeError(
                                "a detached replay ran speculative rounds"
                            )
                        if k == 0 and spec.get("smoke_positives"):
                            # on OFF1's live cache, before it is retired
                            entry["smoke_positives"] = smoke_positives(
                                on, off, n, params
                            )
                        release_trajectory(off, off_part["chat_id"])
                        replays.append((off_part, off))
                if dict(model=id(rg.model), policy=id(policy)) != same:
                    raise RuntimeError(
                        "the target / verify policy changed during the pair"
                    )
                (off1_part, off1), (off2_part, off2) = replays
                save_rows(out_dir / f"rows-off-{index}.npz", off1.rows)
                save_snapshots(
                    out_dir / f"snaps-off-{index}.npz",
                    [off1.states[o] for o in sorted(off1.states)],
                )
                entry["off_final"] = off1_part["final"]
                entry["pair"] = compare_pair(on, off1, n, params)
                c1_ref = snapshot_holder(
                    [off1.states[o] for o in sorted(off1.states)], rows=off1.rows
                )
                c1 = compare_pair(c1_ref, off2, n, params)
                entry["c1"] = dict(
                    c1,
                    acceptance=c1_acceptance(
                        part,
                        [off1_part, off2_part],
                        c1,
                        retired=True,
                        expected_offsets=part["offsets"],
                        replay_offsets=[sorted(off1.states), sorted(off2.states)],
                    ),
                )
            elif (
                job["mode"] == "e1c_off"
            ):  # control C2 (or the v4 cross-load comparison)
                ref_meta = json.loads(Path(spec["on_req"]).read_text())
                tokens = ref_meta["final"]["token_ids"]
                part, rec = e1c_request(
                    client,
                    capture,
                    lm,
                    policy,
                    manifest,
                    spec,
                    "off",
                    timeout,
                    force_tokens=tokens,
                    hash_offsets=ref_meta["offsets"],
                    **probe,
                )
                entry.update(part)
                ref = snapshot_holder(
                    load_snapshots(spec["on_snaps"]), rows=load_rows(spec["on_rows"])
                )
                entry.update(compare_pair(ref, rec, len(tokens), params))
                if spec.get("smoke_positives"):
                    entry["smoke_positives"] = smoke_positives(
                        ref, rec, len(tokens), params
                    )
                wait_idle(rg, server.runtime.metrics)
                release_trajectory(rec, part["chat_id"])
            elif job["mode"] in ("e2", "e7a", "e7b"):
                from p369_phase2 import classes

                tel = telemetry.KernelTelemetry(lm)
                observer = classes.Observer()
                turns = spec.get("turns") or [spec["payload"]]
                chat_id = f"p369-{uuid.uuid4().hex}"
                history = []
                entry["turns"] = []
                for t, payload in enumerate(turns):
                    if job["mode"] == "e7b" and t > 0:
                        payload = dict(payload, messages=history + payload["messages"])
                    observer.begin_request(f"{spec['name']}#{t}")
                    with observer, tel:
                        got = post(
                            client,
                            payload,
                            (
                                chat_id
                                if job["mode"] == "e7b"
                                else f"p369-{uuid.uuid4().hex}"
                            ),
                            capture,
                            timeout,
                        )
                    events = observer.end_request(
                        got["final"]["generation_tokens"]
                        or len(got["final"]["token_ids"]),
                        got["final"]["finish_reason"],
                    )
                    message = got["response"]["choices"][0]["message"]
                    history = payload["messages"] + [
                        dict(role="assistant", content=message.get("content") or "")
                    ]
                    entry["turns"].append(dict(got, events=events))
                    if (
                        job["mode"] != "e7b"
                    ):  # e7b keeps its one session across the turns
                        release_session(got["chat_id"])
                if job["mode"] == "e7b":
                    release_session(chat_id)
                entry["telemetry"] = tel.summary()
                entry["verdict"] = telemetry.verdict(tel, manifest)
            else:
                raise ValueError(f"unknown mode {job['mode']}")
            atomic_write(out_dir / f"req-{index}.json", entry)
            write_progress(
                out_dir,
                index=index,
                name=spec["name"],
                deadline_s=AFTER_REQUEST_S,
                phase="after",
            )
            result["requests"].append(
                {k: entry[k] for k in entry if k not in ("response", "turns")}
            )


if __name__ == "__main__":
    main(sys.argv[1])
