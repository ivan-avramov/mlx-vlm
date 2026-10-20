"""P405 F3: the KV codec a worker must serve, from its argv, and the exact per-layer leaf manifest (pure; no MLX import).

P405 served affine 4-bit KV where the argv asked TurboQuant (``runtime.config`` froze the import-time default) and the harness
saw only a leaf-class count. ``kv_from_argv`` resolves the argv the way the CLI does (its defaults where a flag is absent);
``served_kv`` reads what the ``ResponseGenerator`` actually got; ``manifest_problems`` compares the live cache, layer path by
layer path, with what the served path's policy (``generate/common.py`` ``maybe_quantize_kv_cache``) builds for that argv:
TurboQuant converts every KV layer except the last top-level cache entry when there are more than two; uniform converts every
KV layer. Only ``quantized_kv_start`` 0 is modelled (every registry entry); anything else is refused, never guessed.
"""

import math

KV_FIELDS = (
    "kv_bits",
    "kv_key_bits",
    "kv_value_bits",
    "kv_key_scheme",
    "kv_value_scheme",
    "kv_quant_scheme",
    "kv_group_size",
    "quantized_kv_start",
)
# the CLI's own defaults (mlx_vlm/server/cli.py; generate/common.py DEFAULT_*)
CLI_DEFAULTS = dict(
    kv_group_size=64, quantized_kv_start=5000, kv_quant_scheme="uniform"
)
DEFAULT_TQ_MODE = "mse"


def _bits(raw):
    return (
        float(raw or 0) or None
    )  # absent / 0 = native (the CLI's KV_BITS=0 resolves to None)


def kv_from_argv(argv):
    opts = dict(zip(argv, argv[1:]))
    split = lambda flag: float(opts[flag]) if flag in opts else None  # noqa: E731
    return dict(
        kv_bits=_bits(opts.get("--kv-bits")),
        kv_key_bits=split("--kv-key-bits"),
        kv_value_bits=split("--kv-value-bits"),
        kv_key_scheme=opts.get("--kv-key-scheme"),
        kv_value_scheme=opts.get("--kv-value-scheme"),
        kv_quant_scheme=opts.get("--kv-quant-scheme", CLI_DEFAULTS["kv_quant_scheme"]),
        kv_group_size=int(opts.get("--kv-group-size", CLI_DEFAULTS["kv_group_size"])),
        quantized_kv_start=int(
            opts.get("--quantized-kv-start", CLI_DEFAULTS["quantized_kv_start"])
        ),
        tq_kv_quant_mode=opts.get("--kv-quant-mode", DEFAULT_TQ_MODE),
    )


def served_kv(rg, environ):
    """Every resolved KV field the generator holds, plus the TurboQuant mode the worker's environment carries."""
    got = {name: getattr(rg, name) for name in KV_FIELDS}
    got["kv_bits"] = _bits(got["kv_bits"])
    for name in ("kv_key_bits", "kv_value_bits"):
        got[name] = None if got[name] is None else float(got[name])
    got["tq_kv_quant_mode"] = environ.get("TQ_KV_QUANT_MODE", DEFAULT_TQ_MODE)
    return got


def kv_problems(served, expected):
    return [
        f"{name} served {served.get(name)!r}, argv {expected[name]!r}"
        for name in expected
        if served.get(name) != expected[name]
    ]


# ------------------------------------------------------------------ the per-layer manifest
def _kind(leaf):
    """``tq`` / ``affine`` / ``native`` for a KV leaf, else the class name (linear-attention state and the like)."""
    for cls in type(leaf).__mro__:
        if cls.__name__ == "_TurboQuantAttentionMixin":
            return "tq"
    names = {cls.__name__ for cls in type(leaf).__mro__}
    if "QuantizedKVCache" in names:
        return "affine"
    if "KVCache" in names:
        return "native"
    return type(leaf).__name__


def _record(path, leaf):
    kind = _kind(leaf)
    out = dict(path=path, kind=kind, cls=type(leaf).__name__)
    if kind == "tq":
        out.update(
            bits=float(leaf.bits),
            key_bits=float(leaf.key_bits),
            value_bits=float(leaf.value_bits),
            mode=leaf._kv_quant_mode,
        )
    elif kind == "affine":
        out.update(bits=int(leaf.bits), group_size=int(leaf.group_size))
    return out


def manifest(caches, prefix=""):
    """``[record]`` in layer order; a nested ``CacheList`` contributes ``"<i>.<j>"`` paths."""
    out = []
    for i, leaf in enumerate(caches or []):
        if leaf is None:
            continue
        path = f"{prefix}{i}"
        if type(leaf).__name__ == "CacheList":
            out += manifest(leaf.caches, prefix=f"{path}.")
        else:
            out.append(_record(path, leaf))
    return out


def _tq_bits(kv):
    bits = kv["kv_bits"]
    fractional = not math.isclose(bits, round(bits), abs_tol=1e-6)
    key = (
        kv["kv_key_bits"]
        if kv["kv_key_bits"] is not None
        else (math.floor(bits) if fractional else bits)
    )
    value = (
        kv["kv_value_bits"]
        if kv["kv_value_bits"] is not None
        else (math.ceil(bits) if fractional else bits)
    )
    return float(bits), float(key), float(value)


def expected_manifest(template, kv):
    """What the served path builds from the model's unquantized cache ``template`` (a ``manifest``) for the argv ``kv``."""
    bits = kv["kv_bits"]
    if bits is None:
        return [dict(t) for t in template]
    if kv["kv_key_scheme"] is not None or kv["kv_value_scheme"] is not None:
        raise ValueError(
            "split key / value schemes (the hybrid branch) are not modelled"
        )
    if kv["quantized_kv_start"] != 0:
        raise ValueError(
            f"quantized_kv_start {kv['quantized_kv_start']} is not modelled (only 0)"
        )
    tq = kv["kv_quant_scheme"] == "turboquant" or not math.isclose(
        bits, round(bits), abs_tol=1e-6
    )
    tops = sorted({int(t["path"].split(".")[0]) for t in template})
    last = tops[-1] if len(tops) > 2 else None
    out = []
    for t in template:
        if t["kind"] != "native":
            out.append(dict(t))
            continue
        top = int(t["path"].split(".")[0])
        if tq and top == last:
            out.append(dict(path=t["path"], kind="native"))
        elif tq:
            b, k, v = _tq_bits(kv)
            out.append(
                dict(
                    path=t["path"],
                    kind="tq",
                    bits=b,
                    key_bits=k,
                    value_bits=v,
                    mode=kv["tq_kv_quant_mode"],
                )
            )
        else:
            out.append(
                dict(
                    path=t["path"],
                    kind="affine",
                    bits=int(bits),
                    group_size=kv["kv_group_size"],
                )
            )
    return out


def manifest_problems(observed, template, kv):
    """Layer-path mismatches between the live cache and what the argv implies (class names are not compared: the
    pre-allocating variants are the served path's own; kind, widths and mode are)."""
    expected = {e["path"]: e for e in expected_manifest(template, kv)}
    seen = {o["path"]: o for o in observed}
    problems = []
    missing = [p for p in expected if p not in seen]
    extra = [p for p in seen if p not in expected]
    if missing:
        problems.append(f"layers {missing} missing from the live cache")
    if extra:
        problems.append(f"layers {extra} not in the model's cache template")
    for path, want in expected.items():
        got = seen.get(path)
        if got is None:
            continue
        got = {k: v for k, v in got.items() if k != "cls"}
        want = {k: v for k, v in want.items() if k != "cls"}
        if got != want:
            problems.append(f"layer {path}: served {got}, expected {want}")
    return problems
