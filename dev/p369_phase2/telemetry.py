"""Runtime verifier-kernel outcome telemetry (P369 phase 2, Q18 / Q22 / Q29).

ENTRIES are the qwen3_5 verifier's by-value imports from ``speculative.ops.linear`` (patched in
``models.qwen3_5.speculative_verifier``); LEAVES are the implementations the dispatchers choose among
(patched in ``speculative.ops.linear``, where the dispatchers look them up). A leaf counts only when it
returned a non-``None`` result. Each entry call is classified exactly once from the leaves that
succeeded inside it:

- one or more leaf families → their sorted ``+``-joined names;
- none, entry returned ``None`` → ``declined`` (the caller then runs its own plain fallback);
- none, entry returned a value → ``plain`` (width 1: the singleton path itself) or ``plain_wide``
  (width > 1: a multi-row plain matmul, outside the exact kernels);
- the entry raised → ``error`` (re-raised).

The lm's ``speculative_argmax_from_hidden`` / ``speculative_logits_from_hidden`` calls are counted too.
Calls made inside ``probe()`` are kept apart (``probe_calls``). Each call carries its ``scope``: ``verify``
inside ``speculative.mtp._mtp_verify_target`` / ``_mtp_acceptance_walk`` (the target's verify forward and
its readout), else ``other`` (the drafter's proposals).

Per-module ATTRIBUTIONS (code round 1 H2): every entry call attributes its family to each module it ran on.
A tuple entry (``_target_verify_linears``) whose fused kernel declines dispatches ``_target_verify_linear``
per module; those nested calls (wrapped in ``speculative.ops.linear``) are classified individually, the
tuple call itself is recorded as ``per_module``, and each module carries its own family.

``manifest_from_model`` derives each module's EXPECTED families from the production eligibility predicates
(``quantized_verifier`` / ``exact_speculative_verify``) for a B = 1 verify block; ``verdict(telemetry,
manifest)`` requires, in the verify scope, every wide attribution to be one of its module's expected
families, every width-1 attribution to be expected or a plain / fused-decode singleton, and every ON-PATH
projection (``model.layers.*`` and the head) to appear in at least one wide verify block.
"""

import collections
from contextlib import contextmanager

WIDTH1_OK = {"plain", "quantized_fused_decode", "declined"}
SCOPES = ("_mtp_verify_target", "_mtp_acceptance_walk")
NESTED = "_target_verify_linear"  # dispatched per module by the tuple fallback
VERIFY_BLOCK = (
    3  # the expected-family probe width (any 1 < T <= 8 is the same eligibility)
)

ENTRIES = (
    "_target_verify_linear",
    "_target_verify_linears",
    "_target_verify_embedding_as_linear",
    "_target_verify_quantized_linear",
    "_target_verify_quantized_argmax",
)
LEAVES = {
    "_target_verify_optimized_affine_linear": "quantized_affine_optimized",
    "_target_verify_singleton_quantized_linear": "quantized_singleton",
    "_target_verify_quantized_linears": "quantized_linears_optimized",
    "_decode_quantized_linears_fused": "quantized_fused_decode",
    "_target_verify_weight": "dense_exact",
    "_target_verify_singletons": "fallback_singletons",
    "_target_verify_timewise": "fallback_timewise",
    "_target_verify_optimized_affine_argmax": "argmax_affine_optimized",
    "_target_verify_optimized_nvfp4_argmax": "argmax_nvfp4",
    "_target_verify_singleton_quantized_argmax": "argmax_singleton",
}
READOUTS = ("speculative_argmax_from_hidden", "speculative_logits_from_hidden")


def _width(x):
    shape = getattr(x, "shape", ())
    if len(shape) == 3:
        return int(shape[1])
    return int(shape[0]) if shape else 0


def _dtype(x):
    return str(getattr(x, "dtype", "?")).split(".")[-1]


class KernelTelemetry:
    def __init__(self, lm):
        self.lm = lm
        self.calls = []
        self.probe_calls = []
        self.attributions = []
        self.readouts = collections.Counter()
        self._names = {id(m): name for name, m in lm.named_modules()}
        self._frames = []
        self._probe_depth = 0
        self._scope_depth = 0
        self._patched = []

    # -- recording
    def _modules(self, arg):
        items = arg if isinstance(arg, (list, tuple)) else [arg]
        return [self._names.get(id(m), "?") for m in items]

    def _record(self, entry, family, width, dtype, modules, *, attribute=True):
        scope = "verify" if self._scope_depth else "other"
        target = self.probe_calls if self._probe_depth else self.calls
        target.append(
            dict(
                entry=entry,
                family=family,
                width=width,
                dtype=dtype,
                modules=modules,
                scope=scope,
            )
        )
        if attribute and not self._probe_depth:
            kind = "argmax" if entry == "_target_verify_quantized_argmax" else "linear"
            for module in modules:
                self.attributions.append(
                    dict(
                        entry=entry,
                        family=family,
                        width=width,
                        module=module,
                        kind=kind,
                        scope=scope,
                    )
                )

    @contextmanager
    def probe(self):
        self._probe_depth += 1
        try:
            yield
        finally:
            self._probe_depth -= 1

    def summary(self):
        by_family = collections.Counter(c["family"] for c in self.calls)
        by_entry = collections.Counter(
            (c["entry"], c["family"], c["width"]) for c in self.calls
        )
        return dict(
            calls=len(self.calls),
            probe_calls=len(self.probe_calls),
            by_family=dict(by_family),
            by_entry_family_width={
                "|".join(map(str, k)): v for k, v in sorted(by_entry.items())
            },
            **{name: self.readouts[name] for name in READOUTS},
        )

    # -- wrappers
    def _classified_call(self, name, real, target, x, args, kwargs):
        """Run ``real`` in its own frame and record it exactly once (see the module docstring)."""
        frame = dict(families=[], subs=0)
        self._frames.append(frame)
        modules = self._modules(target)
        try:
            out = real(target, x, *args, **kwargs)
        except BaseException:
            self._frames.pop()
            self._record(name, "error", _width(x), _dtype(x), modules)
            raise
        self._frames.pop()
        if frame["families"]:
            family = "+".join(sorted(set(frame["families"])))
        elif frame["subs"]:
            # the tuple fallback: its nested per-module calls carry the attributions
            self._record(
                name, "per_module", _width(x), _dtype(x), modules, attribute=False
            )
            return out
        elif out is None:
            family = "declined"
        else:
            family = "plain" if _width(x) <= 1 else "plain_wide"
        self._record(name, family, _width(x), _dtype(x), modules)
        return out

    def _entry(self, name, real):
        telemetry = self

        def wrapped(target, x, *args, **kwargs):
            return telemetry._classified_call(name, real, target, x, args, kwargs)

        return wrapped

    def _nested(self, name, real):
        """``ops.linear._target_verify_linear`` as the tuple fallback calls it: classified per module when it
        runs inside an entry; transparent otherwise."""
        telemetry = self
        label = f"_target_verify_linears>{name}"

        def wrapped(target, x, *args, **kwargs):
            if not telemetry._frames:
                return real(target, x, *args, **kwargs)
            telemetry._frames[-1]["subs"] += 1
            return telemetry._classified_call(label, real, target, x, args, kwargs)

        return wrapped

    def _leaf(self, family, real):
        telemetry = self

        def wrapped(*args, **kwargs):
            out = real(*args, **kwargs)
            if out is not None and telemetry._frames:
                telemetry._frames[-1]["families"].append(family)
            return out

        return wrapped

    def _readout(self, name, real):
        telemetry = self

        def wrapped(lm, hidden, *args, **kwargs):
            if not telemetry._probe_depth:
                telemetry.readouts[name] += 1
            return real(lm, hidden, *args, **kwargs)

        return wrapped

    def _scope(self, real):
        telemetry = self

        def wrapped(*args, **kwargs):
            telemetry._scope_depth += 1
            try:
                return real(*args, **kwargs)
            finally:
                telemetry._scope_depth -= 1

        return wrapped

    def _patch(self, owner, name, value):
        self._patched.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def __enter__(self):
        from mlx_vlm.models.qwen3_5 import speculative_verifier as sv
        from mlx_vlm.speculative import mtp
        from mlx_vlm.speculative.ops import linear as ops_linear

        for name in SCOPES:
            self._patch(mtp, name, self._scope(getattr(mtp, name)))

        for name in ENTRIES:
            self._patch(sv, name, self._entry(name, getattr(sv, name)))
        self._patch(
            ops_linear, NESTED, self._nested(NESTED, getattr(ops_linear, NESTED))
        )
        for name, family in LEAVES.items():
            self._patch(ops_linear, name, self._leaf(family, getattr(ops_linear, name)))
        cls = type(self.lm)
        for name in READOUTS:
            if hasattr(cls, name):
                self._patch(cls, name, self._readout(name, getattr(cls, name)))
        return self

    def __exit__(self, *exc):
        while self._patched:
            owner, name, value = self._patched.pop()
            setattr(owner, name, value)
        return False


def _dense_exact_eligible(weight, dtype):
    """``exact_speculative_verify_weight``'s guard (production, ``models/exact_speculative_verify.py``) for a
    B = 1 verify block: kernel present, outputs >= 4 and % 4, K < 16 * outputs, weight dtype = activations.
    """
    from mlx_vlm.models.exact_speculative_verify import (
        exact_speculative_verify_dense_available,
    )

    outputs, dimensions = int(weight.shape[0]), int(weight.shape[-1])
    return (
        exact_speculative_verify_dense_available()
        and outputs >= 4
        and outputs % 4 == 0
        and dimensions < 16 * outputs
        and weight.dtype == dtype
    )


def expected_families(module, dtype):
    """The families production dispatch can return for ``module`` in a B = 1 verify block (width
    ``VERIFY_BLOCK``) with ``dtype`` activations: ``{"linear": [...], "argmax": [...]}`` (argmax only for
    quantized heads). Derived from the production eligibility predicates, never from a run.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models import quantized_verifier as qv

    if isinstance(module, nn.QuantizedLinear):
        k = int(module.scales.shape[-1]) * int(module.group_size)
        x = mx.zeros((1, VERIFY_BLOCK, k), dtype=dtype)
        optimized = qv._can_optimized_affine_linear(module, x)
        if optimized:
            linear = ["quantized_affine_optimized", "quantized_linears_optimized"]
        elif qv.supports_quantization(module):
            linear = ["quantized_singleton"]
        else:
            linear = ["fallback_timewise"]
        if optimized and "bias" not in module:
            argmax = ["argmax_affine_optimized"]
        elif module.mode == "nvfp4":
            argmax = ["argmax_nvfp4"]
        else:
            argmax = ["argmax_singleton"]
        return dict(linear=linear, argmax=argmax)
    if isinstance(module, nn.Linear):
        exact = "bias" not in module and _dense_exact_eligible(module.weight, dtype)
        return dict(linear=["dense_exact"] if exact else ["fallback_singletons"])
    if isinstance(module, getattr(nn, "QuantizedEmbedding", ())):
        return dict(
            linear=["fallback_timewise"]
        )  # packed weight: the dense kernel declines by dtype
    if isinstance(
        module, nn.Embedding
    ):  # a tied head through ``_target_verify_embedding_as_linear``
        exact = _dense_exact_eligible(module.weight, dtype)
        return dict(linear=["dense_exact"] if exact else ["fallback_timewise"])
    return None


def manifest_from_model(lm, dtype=None):
    """``{module name: {kind, bits, group_size, mode, expected, on_path}}`` for every linear / embedding of a
    loaded lm. ``dtype`` = the activation dtype (default: the dtype of the first dense / scale tensor).
    ``on_path``: the module is computed by every verify block (``model.layers.*`` and the head — ``lm_head``,
    or the tied ``embed_tokens``)."""
    import mlx.nn as nn

    modules = list(lm.named_modules())
    if dtype is None:
        for _, module in modules:
            if isinstance(module, nn.QuantizedLinear):
                dtype = module.scales.dtype
                break
            if isinstance(module, nn.Linear):
                dtype = module.weight.dtype
                break
    names = {name for name, _ in modules}
    head = "lm_head" if "lm_head" in names else "model.embed_tokens"
    out = {}
    for name, module in modules:
        if isinstance(module, nn.QuantizedLinear):
            entry = dict(
                kind="quantized",
                bits=int(module.bits),
                group_size=int(module.group_size),
                mode=str(getattr(module, "mode", "affine")),
            )
        elif isinstance(module, nn.Linear):
            entry = dict(kind="dense", bias="bias" in module)
        elif isinstance(module, getattr(nn, "QuantizedEmbedding", ())):
            entry = dict(kind="quantized", bits=int(module.bits))
        elif isinstance(module, nn.Embedding):
            entry = dict(kind="dense")
        else:
            continue
        entry["expected"] = expected_families(module, dtype)
        entry["on_path"] = name.startswith("model.layers.") or name == head
        out[name] = entry
    return out


def _on_path(name, info):
    if "on_path" in info:
        return bool(info["on_path"])
    return name.startswith("model.layers.") or name == "lm_head"


def verdict(telemetry, manifest):
    """Judge the verify-scope ATTRIBUTIONS (see the module docstring): some exist; every module is known; no
    error; a wide one is one of its module's expected families; a width-1 one is expected or plain /
    fused-decode / declined; every on-path module appears in a wide verify block (``missing_coverage``).
    Off-path manifest modules are listed in ``not_on_path`` (reported, not required)."""
    problems = []
    verify = [a for a in telemetry.attributions if a.get("scope") == "verify"]
    wide = [a for a in verify if a["width"] > 1]
    if not verify:
        problems.append("no verify-scope projection call")
    observed = collections.defaultdict(set)
    for a in verify:
        info = manifest.get(a["module"])
        if info is None:
            problems.append(f"unknown module in {a}")
            continue
        if a["family"] == "error":
            problems.append(f"error in {a}")
            continue
        allowed = set((info.get("expected") or {}).get(a["kind"]) or ())
        if a["width"] > 1:
            observed[a["module"]].add(a["family"])
            if a["family"] not in allowed:
                problems.append(
                    f"unexpected family {a['family']} for {a['module']} (expected {sorted(allowed)}): {a}"
                )
        elif a["family"] not in allowed | WIDTH1_OK:
            problems.append(f"unexpected width-1 family in {a}")
    on_path = sorted(m for m, info in manifest.items() if _on_path(m, info))
    missing = [m for m in on_path if m not in observed]
    if missing:
        problems.append(f"projections never observed in a verify block: {missing[:20]}")
    families = collections.Counter(a["family"] for a in wide)
    return dict(
        verify_calls=len(verify),
        wide_calls=len(wide),
        wide_families=dict(families),
        observed={m: sorted(f) for m, f in sorted(observed.items())},
        missing_coverage=missing,
        not_on_path=sorted(m for m, info in manifest.items() if not _on_path(m, info)),
        problems=problems[:50],
        problem_count=len(problems),
        **{"pass": not problems},
    )
