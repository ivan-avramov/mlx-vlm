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
Calls made inside ``probe()`` are kept apart (``probe_calls``).
"""

import collections
from contextlib import contextmanager

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
        self.readouts = collections.Counter()
        self._names = {id(m): name for name, m in lm.named_modules()}
        self._frames = []
        self._probe_depth = 0
        self._patched = []

    # -- recording
    def _modules(self, arg):
        items = arg if isinstance(arg, (list, tuple)) else [arg]
        return [self._names.get(id(m), "?") for m in items]

    def _record(self, entry, family, width, dtype, modules):
        target = self.probe_calls if self._probe_depth else self.calls
        target.append(
            dict(entry=entry, family=family, width=width, dtype=dtype, modules=modules)
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
    def _entry(self, name, real):
        telemetry = self

        def wrapped(target, x, *args, **kwargs):
            telemetry._frames.append([])
            try:
                out = real(target, x, *args, **kwargs)
            except BaseException:
                telemetry._frames.pop()
                telemetry._record(
                    name, "error", _width(x), _dtype(x), telemetry._modules(target)
                )
                raise
            successes = telemetry._frames.pop()
            if successes:
                family = "+".join(sorted(set(successes)))
            elif out is None:
                family = "declined"
            else:
                family = "plain" if _width(x) <= 1 else "plain_wide"
            telemetry._record(
                name, family, _width(x), _dtype(x), telemetry._modules(target)
            )
            return out

        return wrapped

    def _leaf(self, family, real):
        telemetry = self

        def wrapped(*args, **kwargs):
            out = real(*args, **kwargs)
            if out is not None and telemetry._frames:
                telemetry._frames[-1].append(family)
            return out

        return wrapped

    def _readout(self, name, real):
        telemetry = self

        def wrapped(lm, hidden, *args, **kwargs):
            if not telemetry._probe_depth:
                telemetry.readouts[name] += 1
            return real(lm, hidden, *args, **kwargs)

        return wrapped

    def _patch(self, owner, name, value):
        self._patched.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def __enter__(self):
        from mlx_vlm.models.qwen3_5 import speculative_verifier as sv
        from mlx_vlm.speculative.ops import linear as ops_linear

        for name in ENTRIES:
            self._patch(sv, name, self._entry(name, getattr(sv, name)))
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
