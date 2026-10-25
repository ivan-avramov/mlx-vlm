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
TUPLE_ENTRY = "_target_verify_linears"
TUPLE_FAMILY = "quantized_linears_optimized"
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
# P369 v13 V3 / V15: routed MoE projections run OUTSIDE the linear dispatcher — the verifier's ``_switch_glu`` calls the
# SwitchLinear / QuantizedSwitchLinear banks directly, ``SwitchGLU`` may take exact fused helpers that read the bank
# weights. Each is recorded per routed-bank invocation, by module identity.
ROUTED_PROJ = ("gate_proj", "up_proj", "down_proj")
# helper name -> (module, the bank(s) its first argument stands for)
ROUTED_HELPERS = {
    "exact_affine_switch_gate_up": ("fast_ops", ("gate_proj", "up_proj")),
    "exact_affine_moe_down": ("fast_ops", None),
    "exact_quantized_switch_linear": ("quantized_verifier", None),
    "exact_quantized_selected_linear": ("quantized_verifier", None),
    "exact_quantized_moe_hc_expand": ("quantized_verifier", None),
}
DRAFT_MOE_PARTS = (
    "gate",
    "shared_expert.gate_proj",
    "shared_expert.up_proj",
    "shared_expert.down_proj",
    "shared_expert_gate",
    "switch_mlp.gate_proj",
    "switch_mlp.up_proj",
    "switch_mlp.down_proj",
)


def _width(x):
    shape = getattr(x, "shape", ())
    if len(shape) == 3:
        return int(shape[1])
    return int(shape[0]) if shape else 0


def _dtype(x):
    return str(getattr(x, "dtype", "?")).split(".")[-1]


class KernelTelemetry:
    def __init__(self, lm, drafter=None):
        self.lm = lm
        self.drafter = drafter
        self.routed_frames = (
            []
        )  # v13 V3: one per routed-bank invocation (verifier ``_switch_glu`` / ``SwitchGLU``)
        self.draft_moe = (
            []
        )  # v13 V15: one per drafter MoE-block invocation in draft scope
        self._routed_stack = []
        self._helper_depth = (
            0  # code round 1 M5: a fused helper's own bank calls are not counted twice
        )
        self._draft_stack = []
        self._draft_depth = 0
        self.calls = []
        self.probe_calls = []
        self.attributions = []
        self.readouts = collections.Counter()
        self._names = {id(m): name for name, m in lm.named_modules()}
        if (
            drafter is not None
        ):  # P413: a bound drafter's references to target modules keep the target's names
            for name, m in drafter.named_modules():
                self._names.setdefault(id(m), f"drafter.{name}")
        self._frames = []
        self._probe_depth = 0
        self._scope_depth = 0
        self._patched = []

    # -- recording
    def _modules(self, arg):
        items = arg if isinstance(arg, (list, tuple)) else [arg]
        return [self._names.get(id(m), "?") for m in items]

    def _record(self, entry, family, width, dtype, modules, *, attribute=True, **extra):
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
                **extra,
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
        """Run ``real`` in its own frame and record it exactly once (see the module docstring). A tuple
        entry also records whether the production tuple kernel was eligible for this exact group
        (``tuple_expected``, code round 2 H2)."""
        extra = {}
        if name == TUPLE_ENTRY and _width(x) > 1:
            extra["tuple_expected"] = tuple_kernel_eligible(target, x)
        frame = dict(families=[], subs=0)
        self._frames.append(frame)
        modules = self._modules(target)
        try:
            out = real(target, x, *args, **kwargs)
        except BaseException:
            self._frames.pop()
            self._record(name, "error", _width(x), _dtype(x), modules, **extra)
            raise
        self._frames.pop()
        if frame["families"]:
            family = "+".join(sorted(set(frame["families"])))
        elif frame["subs"]:
            # the tuple fallback: its nested per-module calls carry the attributions
            self._record(
                name,
                "per_module",
                _width(x),
                _dtype(x),
                modules,
                attribute=False,
                **extra,
            )
            return out
        elif out is None:
            family = "declined"
        else:
            family = "plain" if _width(x) <= 1 else "plain_wide"
        self._record(name, family, _width(x), _dtype(x), modules, **extra)
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

    # -- v13 V3 / V15: routed MoE banks and the drafter's MoE path
    def _scope_name(self):
        if self._draft_depth:
            return "draft"
        return "verify" if self._scope_depth else "other"

    def _routed_note(self, module, indices, projections=None):
        """One routed-bank projection call — or a fused helper standing for ``projections`` of the container
        ``module`` — attributed to EACH underlying bank (its own family; code round 1 M5) and to the enclosing
        invocation frame (code round 1 M6: a verify-scope routed call outside a frame is a problem).
        """
        import numpy as np

        if projections is None:
            name = self._names.get(id(module), "?")
            banks = [(name.rsplit(".", 1)[-1], name, module)]
        else:  # code round 3 R2: each bank named by ITS module identity, never synthesized from the container
            banks = [
                (
                    p,
                    self._names.get(id(getattr(module, p, None)), "?"),
                    getattr(module, p),
                )
                for p in projections
            ]
        tokens = int(indices.size // indices.shape[-1]) if indices is not None else 0
        experts = (
            sorted({int(e) for e in np.array(indices).reshape(-1)})
            if indices is not None
            else []
        )
        frame = self._routed_stack[-1] if self._routed_stack else None
        if frame is not None:
            frame["projections"].extend(p for p, _, _ in banks)
            frame["modules"].extend(n for _, n, _ in banks)  # code round 2 N3
            frame["experts"].update(experts)
        if not self._probe_depth:
            for _, bank_name, bank in banks:
                family = (
                    "routed_gather_quantized"
                    if type(bank).__name__.startswith("Quantized")
                    else "routed_gather"
                )
                self.attributions.append(
                    dict(
                        entry="routed",
                        family=family,
                        width=tokens,
                        module=bank_name,
                        kind="routed",
                        scope=self._scope_name(),
                        frame=frame["id"] if frame is not None else None,
                    )
                )
        if (
            self._draft_stack
        ):  # code round 3 R2: a bank outside the frame's block is recorded as foreign
            frame = self._draft_stack[-1]
            prefix = frame["block"] + "."
            frame["parts"].extend(
                n[len(prefix) :] if n.startswith(prefix) else f"foreign:{n}"
                for _, n, _ in banks
            )
            frame["experts"].update(experts)

    def _routed_bank(self, real):
        telemetry = self

        def wrapped(module, x, indices, *args, **kwargs):
            if (
                not telemetry._helper_depth
            ):  # inside a fused helper: the helper records it once
                telemetry._routed_note(module, indices)
            return real(module, x, indices, *args, **kwargs)

        return wrapped

    def _routed_helper(self, real, projections):
        telemetry = self

        def wrapped(module, x, indices, *args, **kwargs):
            telemetry._helper_depth += 1
            try:
                out = real(module, x, indices, *args, **kwargs)
            finally:
                telemetry._helper_depth -= 1
            if (
                out is not None and not telemetry._helper_depth
            ):  # code round 2 M5: a helper nested in a helper is the outer one's
                telemetry._routed_note(module, indices, projections)
            return out

        return wrapped

    def _routed_invocation(self, real, *, positional_indices=2):
        """Opens a routed-bank invocation frame around ``_switch_glu`` / ``SwitchGLU.__call__``."""
        telemetry = self

        def wrapped(*args, **kwargs):
            x = args[positional_indices - 1]
            frame = dict(
                id=len(telemetry.routed_frames) + len(telemetry._routed_stack),
                scope=telemetry._scope_name(),
                width=int(x.shape[0] * x.shape[1]) if x.ndim == 3 else int(x.shape[0]),
                # code round 2 N3: the container whose banks this invocation must run (the argument before x)
                owner=telemetry._names.get(id(args[positional_indices - 2]), "?"),
                projections=[],
                modules=[],
                experts=set(),
            )
            telemetry._routed_stack.append(frame)
            try:
                return real(*args, **kwargs)
            finally:
                telemetry._routed_stack.pop()
                if not telemetry._probe_depth:
                    frame["experts"] = sorted(frame["experts"])
                    telemetry.routed_frames.append(frame)

        return wrapped

    def _draft_scope(self, real):
        telemetry = self

        def wrapped(*args, **kwargs):
            telemetry._draft_depth += 1
            try:
                return real(*args, **kwargs)
            finally:
                telemetry._draft_depth -= 1

        return wrapped

    def _draft_moe_block(self, real):
        telemetry = self

        def wrapped(block, x, *args, **kwargs):
            name = telemetry._names.get(id(block), "?")
            if not (telemetry._draft_depth and name.startswith("drafter.")):
                return real(block, x, *args, **kwargs)
            frame = dict(block=name, parts=[], experts=set())
            telemetry._draft_stack.append(frame)
            try:
                return real(block, x, *args, **kwargs)
            finally:
                telemetry._draft_stack.pop()
                frame["experts"] = sorted(frame["experts"])
                telemetry.draft_moe.append(frame)

        return wrapped

    def _draft_linear(self, real):
        telemetry = self

        def wrapped(module, x, *args, **kwargs):
            if telemetry._draft_stack:
                frame = telemetry._draft_stack[-1]
                name = telemetry._names.get(id(module), "?")
                prefix = frame["block"] + "."
                if name.startswith(prefix):
                    frame["parts"].append(name[len(prefix) :])
            return real(module, x, *args, **kwargs)

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
        # v13 V3 / V15: routed banks, their fused helpers, and the drafter's own MoE path
        import mlx.nn as nn

        from mlx_vlm.models import fast_ops, quantized_verifier, switch_layers

        helpers = dict(fast_ops=fast_ops, quantized_verifier=quantized_verifier)
        for bank in (switch_layers.SwitchLinear, switch_layers.QuantizedSwitchLinear):
            self._patch(bank, "__call__", self._routed_bank(bank.__call__))
        for helper, (module, projections) in ROUTED_HELPERS.items():
            owner = helpers[module]
            if hasattr(owner, helper):
                self._patch(
                    owner,
                    helper,
                    self._routed_helper(getattr(owner, helper), projections),
                )
        self._patch(
            sv.Qwen3_5BatchInvariantForward,
            "_switch_glu",
            self._routed_invocation(
                sv.Qwen3_5BatchInvariantForward._switch_glu, positional_indices=3
            ),
        )
        self._patch(
            switch_layers.SwitchGLU,
            "__call__",
            self._routed_invocation(switch_layers.SwitchGLU.__call__),
        )
        if self.drafter is not None:
            from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

            dcls = type(self.drafter)
            self._patch(dcls, "draft_block", self._draft_scope(dcls.draft_block))
            self._patch(
                Qwen3_5MoeSparseMoeBlock,
                "__call__",
                self._draft_moe_block(Qwen3_5MoeSparseMoeBlock.__call__),
            )
            for lin in (nn.Linear, nn.QuantizedLinear):
                self._patch(lin, "__call__", self._draft_linear(lin.__call__))
        return self

    def __exit__(self, *exc):
        while self._patched:
            owner, name, value = self._patched.pop()
            setattr(owner, name, value)
        return False


def tuple_kernel_eligible(linears, x):
    """``quantized_verifier.optimized_affine_linears``' guard (production, replicated with its source named)
    for the exact group and verify block: 2–4 bias-free affine QuantizedLinears of one bits / group size /
    mode, each eligible for the optimized kernel, 1 < T <= 8."""
    import mlx.nn as nn

    from mlx_vlm.models import quantized_verifier as qv

    linears = list(linears) if isinstance(linears, (list, tuple)) else [linears]
    bits = getattr(linears[0], "bits", None) if linears else None
    return bool(
        2 <= len(linears) <= 4
        and x.ndim == 3
        and 1 < x.shape[1] <= 8
        and bits in (4, 5, 8)
        and all(
            isinstance(linear, nn.QuantizedLinear)
            and linear.bits == bits
            and linear.group_size == linears[0].group_size
            and linear.mode == linears[0].mode
            and "bias" not in linear
            and qv._can_optimized_affine_linear(linear, x)
            for linear in linears
        )
    )


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
        elif type(module).__name__ in ("SwitchLinear", "QuantizedSwitchLinear"):
            quantized = type(module).__name__ == "QuantizedSwitchLinear"
            entry = dict(
                kind="routed",
                expected=dict(
                    routed=["routed_gather_quantized" if quantized else "routed_gather"]
                ),
                on_path=name.startswith("model.layers."),
            )
            out[name] = (
                entry  # v13 V3: the routed banks, invisible to the linear dispatcher
            )
            continue
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
    for (
        call
    ) in (
        telemetry.calls
    ):  # code round 2 H2: an eligible group must use the tuple kernel
        if (
            call.get("scope") == "verify"
            and call.get("tuple_expected")
            and call["family"] != TUPLE_FAMILY
        ):
            problems.append(
                f"tuple kernel expected ({TUPLE_FAMILY}) for {call['modules']}, got {call['family']}"
            )
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
    routed_on_path = [
        m
        for m, info in manifest.items()
        if info.get("kind") == "routed" and _on_path(m, info)
    ]
    verify_frames = [f for f in telemetry.routed_frames if f["scope"] == "verify"]
    if (
        routed_on_path and not verify_frames
    ):  # code round 1 M6: the invocation observer must have run
        problems.append(
            "routed banks on the path but no verify-scope routed invocation observed"
        )
    for a in verify:
        if a.get("kind") == "routed" and a.get("frame") is None:
            problems.append(
                f"verify-scope routed call outside an invocation frame: {a}"
            )
    # v13 V3: each verify-scope invocation runs every bank exactly once — code round 2 N3: its OWN container's banks
    for frame in telemetry.routed_frames:
        if frame["scope"] != "verify":
            continue
        if sorted(frame["projections"]) != sorted(ROUTED_PROJ):
            problems.append(f"routed invocation ran {frame['projections']}: {frame}")
        owned = sorted(f"{frame['owner']}.{p}" for p in ROUTED_PROJ)
        if frame["owner"] == "?" or sorted(frame["modules"]) != owned:
            problems.append(
                f"routed invocation of owner {frame['owner']} ran banks {sorted(frame['modules'])}: {frame}"
            )
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


def routed_invocations(telemetry, *, scope):
    """The routed-bank invocations recorded in ``scope`` (``verify`` / ``draft`` / ``other``)."""
    return [f for f in telemetry.routed_frames if f["scope"] == scope]


def draft_verdict(telemetry, drafter):
    """P369 v13 V15: the drafter's MoE path in its draft scope — at least one MoE-block invocation, each running the
    router, the shared expert, its gate and every routed bank exactly once, with its selected experts recorded. A
    dense drafter (no MoE block) has nothing to verify here."""
    names = [
        n
        for n, m in drafter.named_modules()
        if type(m).__name__ == "Qwen3_5MoeSparseMoeBlock"
    ]
    problems = []
    if not names:
        return dict(
            pass_=True, problems=[], invocations=[], moe=False, **{"pass": True}
        )
    if not telemetry.draft_moe:
        problems.append("no drafter MoE-block invocation in draft scope")
    for frame in telemetry.draft_moe:
        if sorted(frame["parts"]) != sorted(DRAFT_MOE_PARTS):
            problems.append(
                f"drafter MoE block ran {sorted(frame['parts'])}: {frame['block']}"
            )
        if not frame["experts"]:
            problems.append(
                f"drafter MoE block {frame['block']}: no selected experts recorded"
            )
    invocations = [
        dict(block=f["block"], parts=sorted(f["parts"]), selected_experts=f["experts"])
        for f in telemetry.draft_moe
    ]
    return dict(
        problems=problems[:50],
        invocations=invocations,
        moe=True,
        **{"pass": not problems},
    )
