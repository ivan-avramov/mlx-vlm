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

        self._patched += [
            (sys.modules["mlx_vlm.generate"], "stream_generate", stream),
            (generation.ResponseGenerator, "generate", generate_fn),
        ]
        sys.modules["mlx_vlm.generate"].stream_generate = capturing_stream
        generation.ResponseGenerator.generate = capturing_generate
        return self

    def reset(self):
        self.final, self.backend = None, None

    def __exit__(self, *exc):
        while self._patched:
            owner, name, value = self._patched.pop()
            setattr(owner, name, value)
        return False


def post(client, payload, chat_id, capture, timeout_s):
    capture.reset()
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
    if capture.backend != "cached_session":
        raise RuntimeError(f"backend {capture.backend!r}, expected cached_session")
    if capture.final is None:
        raise RuntimeError("no final chunk captured")
    return dict(response=r.json(), final=dict(capture.final), wall_s=elapsed)


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


def run_job(job, out_dir, result):
    from p369_phase2 import compare, observe, telemetry

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
        result["manifest"] = telemetry.manifest_from_model(lm)
        capture = stack.enter_context(Capture())
        if job["mode"] == "size":
            for spec in job["requests"]:
                count = lambda text: prompt_tokens(  # noqa: E731
                    client,
                    [dict(role="user", content=text)],
                    spec["payload"],
                )
                text, tokens = size_prompt(
                    count, spec["target"], prefix=spec["prefix"], suffix=spec["suffix"]
                )
                result["requests"].append(
                    dict(name=spec["name"], text=text, tokens=tokens)
                )
            return
        if job["mode"] == "count":
            for spec in job["requests"]:
                tokens = prompt_tokens(
                    client, spec["payload"]["messages"], spec["payload"]
                )
                result["requests"].append(dict(name=spec["name"], tokens=tokens))
            return
        for index, spec in enumerate(job["requests"]):
            entry = dict(name=spec["name"], index=index)
            timeout = float(spec["timeout_s"])
            tel = telemetry.KernelTelemetry(lm)
            if job["mode"] in ("e1c_on", "e1c_off"):
                rec_kwargs = dict(lm=lm, probe_token=int(job["probe_token"]))
                if job["mode"] == "e1c_off":
                    on_dir = Path(job["on_dir"])
                    on_meta = json.loads((on_dir / f"req-{index}.json").read_text())
                    rec_kwargs.update(
                        force_tokens=on_meta["final"]["token_ids"],
                        hash_offsets=on_meta["offsets"],
                    )
                recorder = observe.Recorder(
                    "on" if job["mode"] == "e1c_on" else "off", **rec_kwargs
                )
                real_probe = observe.probe

                def frozen_probe(lm_, caches, token):
                    with probe_counters_frozen(policy), tel.probe():
                        return real_probe(lm_, caches, token)

                observe.probe = frozen_probe
                try:
                    with recorder, tel, observe.KeyLengths() as key_lengths:
                        got = post(
                            client,
                            spec["payload"],
                            f"p369-{uuid.uuid4().hex}",
                            capture,
                            timeout,
                        )
                finally:
                    observe.probe = real_probe
                entry.update(got)
                entry["telemetry"] = tel.summary()
                entry["verdict"] = telemetry.verdict(tel, result["manifest"])
                offsets = []
                if recorder.prompt_end is not None:
                    offsets.append(recorder.prompt_end["offset"])
                offsets += [c["offset"] for c in recorder.commits]
                entry["offsets"] = offsets
                entry["prompt_end"] = (
                    recorder.prompt_end["offset"] if recorder.prompt_end else None
                )
                entry["draft_rounds"] = recorder.draft_rounds
                spans = sorted(
                    {(b["first_key"], b["last_key"]) for b in key_lengths.blocks}
                )
                entry["key_spans"] = dict(
                    blocks=len(spans),
                    first=spans[0] if spans else None,
                    last=spans[-1] if spans else None,
                )
                threshold = spec.get("threshold")
                entry["straddle"] = (
                    observe.straddles(key_lengths.blocks, threshold)
                    if threshold
                    else None
                )
                save_rows(out_dir / f"rows-{index}.npz", recorder.rows)
                snaps = (
                    ([recorder.prompt_end] if recorder.prompt_end else [])
                    + recorder.commits
                    if job["mode"] == "e1c_on"
                    else [recorder.states[o] for o in sorted(recorder.states)]
                )
                save_snapshots(out_dir / f"snaps-{index}.npz", snaps)
                if job["mode"] == "e1c_off":
                    n = len(on_meta["final"]["token_ids"])
                    on_rows = load_rows(on_dir / f"rows-{index}.npz")
                    entry["rows"] = compare.compare_streams(
                        on_rows, recorder.rows, n, job["params"]
                    )
                    on_snaps = load_snapshots(on_dir / f"snaps-{index}.npz")
                    holder = type("Snaps", (), {})()
                    holder.prompt_end = (
                        on_snaps[0]
                        if on_snaps and on_snaps[0]["kind"] == "prompt_end"
                        else None
                    )
                    holder.commits = [s for s in on_snaps if s["kind"] != "prompt_end"]
                    off_holder = type("Snaps", (), {})()
                    off_holder.states = dict(recorder.states)
                    entry["states"] = observe.compare_states(holder, off_holder)
                atomic_write(out_dir / f"req-{index}.json", entry)
            elif job["mode"] in ("e2", "e7a", "e7b"):
                from p369_phase2 import classes

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
                entry["telemetry"] = tel.summary()
                entry["verdict"] = telemetry.verdict(tel, result["manifest"])
                atomic_write(out_dir / f"req-{index}.json", entry)
            else:
                raise ValueError(f"unknown mode {job['mode']}")
            result["requests"].append(
                {k: entry[k] for k in entry if k not in ("response", "turns")}
            )


if __name__ == "__main__":
    main(sys.argv[1])
