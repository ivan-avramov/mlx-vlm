"""Joint MTP verification scan policy for qwen3_5 native-KV verification blocks.

Fork-only (M58). ``joint_v1`` replaces the verifier's per-query attention calls
(one call per query over a physically shortened key/value prefix) with ONE call
over the full keys/values (``mask="causal"``, or the AND of an incoming 4-D
boolean mask with the per-query step mask) when ALL of:

1. ``length >= 2`` (verification block, ``output is None``);
2. native cache (no ``bits`` attribute, not TurboQuant), plain 4-D keys/values,
   batch 1;
3. ``mask`` is None, ``"causal"`` or a 4-D boolean array covering the block;
4. ``key_length - length >= 0``;
5. ``length * (n_q // n_kv) <= VECTOR_QUERY_BOUND`` (the fused vector kernel's
   query bound, STEP 1);
6. inside the qualified domain (dtype, head dim, GQA, device class);
7. no MLX key-length threshold straddle: ``_qwen3_5_sdpa_vector_plan`` of the
   shortest and the longest per-query key length agree.

Otherwise the existing per-query code runs with its exact arguments. The
constants, thresholds and domain belong to the version name. ``per_query`` is
today's behaviour and never creates a policy object.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx

from .models.base import (
    BatchTurboQuantKVCache,
    TurboQuantKVCache,
    kv_sequence_length,
    scaled_dot_product_attention,
    slice_kv_sequence,
)

POLICY_NAMES = ("per_query", "joint_v1")
ENV_SCAN = "MLX_VLM_MTP_VERIFY_SCAN"
ENV_AB = "MLX_VLM_MTP_VERIFY_AB"
MIN_LENGTH = 2
VECTOR_QUERY_BOUND = 32  # STEP 1: 36 > 32 leaves the vector kernel (applegpu_g17s)
SELF_TEST_KEYS = 2048
SELF_TEST_STRADDLE_PREFIX = 1022  # keys 1023..1025 cross the 1024 threshold
SELF_TEST_BUDGET_S = 30.0

BLOCK_KEYS = (
    "verify_blocks_joint_v1",
    "verify_blocks_per_query",
    "verify_blocks_straddle",
    "verify_blocks_len1",
)
AB_KEYS = (
    "verify_ab_blocks",
    "verify_ab_mismatch",
    "verify_ab_straddle_blocks",
    "verify_ab_straddle_mismatch",
)
REASONS_KEY = "verify_fallback_reasons"


class VerifyScanError(RuntimeError):
    """The requested scan policy cannot be served by this model/device."""


@dataclass(frozen=True)
class Domain:
    """The shapes STEP 1 measured; outside it the per-query path runs."""

    dtypes: Tuple = (mx.bfloat16,)
    head_dim: int = 256
    gqa: Tuple = (6,)
    device_classes: Tuple = ("applegpu_g17s",)


QUALIFIED_DOMAIN = Domain()


def _gpu_enabled() -> bool:
    return mx.default_device() == mx.gpu


def _device_class() -> Optional[str]:
    if not _gpu_enabled():
        return None
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        return str(info.get("architecture", ""))
    except Exception:  # noqa: BLE001 - undiscoverable device: fail closed (not in any domain)
        return None


def _mirror_matches_live_device(live_class) -> bool:
    """The plan mirror reads a process-cached architecture suffix (`language.py`); the domain check
    reads the device live. They must agree, else the mirror could model a different device class
    than the one serving: fail closed."""
    from .models.qwen3_5 import language as _language

    try:
        cached = _language._qwen3_5_device_arch_suffix()
    except Exception:  # noqa: BLE001
        return False
    return bool(live_class) and live_class[-1:] == cached


def _fail(message: str):
    line = f"mtp-verify-scan joint_v1: {message}"
    print(line, file=sys.stderr, flush=True)
    raise VerifyScanError(line)


def step_mask(length: int, key_length: int) -> mx.array:
    """The per-query key prefixes as a mask: query i sees keys [0, prefix + i]."""
    prefix_length = key_length - length
    return (
        mx.arange(key_length)[None, None, None, :]
        < (prefix_length + mx.arange(length) + 1)[None, None, :, None]
    )


def joint_attention(queries, keys, values, *, cache, scale, mask):
    """The one joint call, shared by the production path and the AB instrument."""
    return scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=scale, mask=mask
    )


def per_query_attention(queries, keys, values, *, cache, scale, mask):
    """The verifier's per-query decomposition (the AB reference and the served
    path for straddling blocks); mirrors the verifier branch's arguments."""
    length = queries.shape[2]
    prefix_length = kv_sequence_length(keys) - length
    return mx.concatenate(
        [
            scaled_dot_product_attention(
                queries[:, :, index : index + 1, :],
                slice_kv_sequence(keys, prefix_length + index + 1),
                slice_kv_sequence(values, prefix_length + index + 1),
                cache=cache,
                scale=scale,
                mask=(
                    mask[..., index : index + 1, : prefix_length + index + 1]
                    if isinstance(mask, mx.array) and mask.ndim >= 4
                    else None
                ),
            )
            for index in range(length)
        ],
        axis=2,
    )


def _bits(array):
    return array.view({1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[array.dtype.size])


def _invalid(a, b):
    """True or a lazy bool scalar: shape/dtype inequality or any non-finite value in either."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return True
    return ~mx.all(mx.isfinite(a)) | ~mx.all(mx.isfinite(b))


def _bits_differ(a, b):
    """Lazy bool scalar: any representation-level difference (array_equal is not enough:
    -0 == +0, NaN != NaN). Only meaningful when shapes and dtypes are equal."""
    return mx.any(_bits(a) != _bits(b))


def _differs(a, b):
    """True or a lazy bool scalar: invalid (shape/dtype/non-finite) or any bit difference."""
    invalid = _invalid(a, b)
    if invalid is True:
        return True
    return _bits_differ(a, b) | invalid


@dataclass
class Decision:
    route: str  # joint | straddle | per_query | len1
    reason: Optional[str] = None
    joint_mask: object = None


@dataclass
class _Pending:
    straddle: bool
    flag: object
    layer: int
    length: int
    key_length: int
    joint: object
    other: object


class SelfTestResult:
    def __init__(self, ran=0, elapsed_s=0.0):
        self.ran = ran
        self.elapsed_s = elapsed_s


class JointV1Policy:
    name = "joint_v1"

    def __init__(self, ab: bool = False, domain: Optional[Domain] = None):
        self.ab = bool(ab)
        self.domain = domain
        self._counts = {k: 0 for k in BLOCK_KEYS + (AB_KEYS if ab else ())}
        self._reasons = {}
        self._pending = []
        self._ordinal = 0
        self._logged_shapes = set()

    # ------------------------------------------------------------ decision
    def classify(self, queries, keys, values, cache, mask) -> Decision:
        """Pure decision (rules 1-7); nothing is counted or computed."""
        length = queries.shape[2]
        if length < MIN_LENGTH:
            return Decision("len1")
        if (
            hasattr(cache, "bits")
            or isinstance(cache, (TurboQuantKVCache, BatchTurboQuantKVCache))
            or not (isinstance(keys, mx.array) and isinstance(values, mx.array))
            or keys.ndim != 4
            or values.ndim != 4
        ):
            return Decision("per_query", "cache")
        if queries.shape[0] != 1 or keys.shape[0] != 1:
            return Decision("per_query", "batch")
        key_length = keys.shape[-2]
        causal = mask is None or (isinstance(mask, str) and mask == "causal")
        array_mask = (
            isinstance(mask, mx.array)
            and mask.ndim == 4
            and mask.dtype == mx.bool_
            and mask.shape[-2] >= length
            and mask.shape[-1] >= key_length
        )
        if not (causal or array_mask):
            return Decision("per_query", "mask_form")
        if key_length - length < 0:
            return Decision("per_query", "prefix")
        n_q, n_kv = queries.shape[1], keys.shape[1]
        gqa = n_q // n_kv
        if length * gqa > VECTOR_QUERY_BOUND:
            return Decision("per_query", "gqa_bound")
        domain = self.domain or QUALIFIED_DOMAIN
        if (
            queries.dtype not in domain.dtypes
            or queries.shape[-1] != domain.head_dim
            or n_q % n_kv != 0
            or gqa not in domain.gqa
            or _device_class() not in domain.device_classes
            or not _mirror_matches_live_device(_device_class())
        ):
            return Decision("per_query", "domain")
        joint_mask = "causal"
        if array_mask:
            joint_mask = step_mask(length, key_length) & mask[..., :length, :key_length]
            if not _rows_are_contiguous_runs(joint_mask):
                return Decision("per_query", "mask_rows")
        from .models.qwen3_5 import language as _language

        plan = _language._qwen3_5_sdpa_vector_plan
        if plan(key_length - length + 1, n_q, n_kv) != plan(key_length, n_q, n_kv):
            return Decision("straddle", None, joint_mask)
        return Decision("joint", None, joint_mask)

    # -------------------------------------------------------------- dispatch
    def attend(self, *, queries, keys, values, cache, scale, mask):
        """Joint output, per-query output (straddle), or None (the verifier's
        existing code runs, call-for-call, with its exact arguments)."""
        decision = self.classify(queries, keys, values, cache, mask)
        route = decision.route
        if route == "len1":
            self._counts["verify_blocks_len1"] += 1
            return None
        if route == "per_query":
            self._counts["verify_blocks_per_query"] += 1
            self._reasons[decision.reason] = self._reasons.get(decision.reason, 0) + 1
            return None
        args = dict(cache=cache, scale=scale)
        if route == "joint":
            self._counts["verify_blocks_joint_v1"] += 1
            joint = joint_attention(queries, keys, values, mask=decision.joint_mask, **args)
            if self.ab:
                other = per_query_attention(queries, keys, values, mask=mask, **args)
                self._shadow(False, joint, other, queries, keys)
            return joint
        self._counts["verify_blocks_straddle"] += 1
        served = per_query_attention(queries, keys, values, mask=mask, **args)
        if self.ab:
            joint = joint_attention(queries, keys, values, mask=decision.joint_mask, **args)
            self._shadow(True, joint, served, queries, keys)
        return served

    def _shadow(self, straddle, joint, other, queries, keys):
        # AB counters advance only in end_block(), after the flags are materialised.
        self._ordinal += 1
        self._pending.append(
            _Pending(
                straddle, _differs(joint, other), self._ordinal, queries.shape[2],
                keys.shape[-2], joint, other,
            )
        )

    def end_block(self):
        """Materialise the round's AB flags once, then advance the counters."""
        self._ordinal = 0
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        flags = [p.flag for p in pending if isinstance(p.flag, mx.array)]
        if flags:
            mx.eval(flags)
        for entry in pending:
            self._counts["verify_ab_straddle_blocks" if entry.straddle else "verify_ab_blocks"] += 1
            bad = entry.flag if isinstance(entry.flag, bool) else bool(entry.flag.item())
            if not bad:
                continue
            if entry.straddle:
                self._counts["verify_ab_straddle_mismatch"] += 1
                continue
            self._counts["verify_ab_mismatch"] += 1
            self._log_mismatch(entry)

    def discard_pending(self):
        """Drop this round's unmaterialised comparisons (an exception aborted the round): no
        counter advances and no array references are retained."""
        self._pending = []
        self._ordinal = 0

    def _log_mismatch(self, entry):
        shape = (entry.length, entry.key_length, str(entry.joint.dtype))
        if shape in self._logged_shapes:
            return
        self._logged_shapes.add(shape)
        try:
            max_abs = float(
                mx.max(
                    mx.abs(entry.joint.astype(mx.float32) - entry.other.astype(mx.float32))
                ).item()
            )
        except Exception:  # noqa: BLE001 - shape mismatch: no elementwise diff
            max_abs = float("nan")
        line = (
            "mtp_verify_scan joint_v1 AB mismatch: "
            f"layer={entry.layer} length={entry.length} key_length={entry.key_length} "
            f"dtype={entry.joint.dtype} max_abs={max_abs:g}"
        )
        import logging

        logging.getLogger(__name__).warning(line)
        print(line, file=sys.stderr, flush=True)

    # ------------------------------------------------------------- counters
    def counters(self) -> dict:
        self.end_block()
        out = dict(self._counts)
        out[REASONS_KEY] = dict(self._reasons)
        return out

    snapshot = counters

    def since(self, snapshot: dict) -> dict:
        now = self.counters()
        out = {k: now[k] - snapshot.get(k, 0) for k in now if k != REASONS_KEY}
        before = snapshot.get(REASONS_KEY, {})
        out[REASONS_KEY] = {
            k: n - before.get(k, 0)
            for k, n in now[REASONS_KEY].items()
            if n - before.get(k, 0)
        }
        return out


def _rows_are_contiguous_runs(mask) -> bool:
    """Every row non-empty and a single contiguous run of True (rare path: only
    an incoming 4-D boolean mask reaches it; synchronises once)."""
    non_empty = mx.all(mx.any(mask, axis=-1))
    runs = mask[..., :1].astype(mx.int32).sum(-1) + (
        mask[..., 1:] & ~mask[..., :-1]
    ).astype(mx.int32).sum(-1)
    return bool((non_empty & mx.all(runs <= 1)).item())


def resolve_policy(scan: Optional[str], ab: bool = False):
    """``None`` for absent/``per_query`` (today's behaviour); a policy otherwise."""
    if scan in (None, "", "per_query"):
        if ab:
            raise ValueError("--mtp-verify-ab requires --mtp-verify-scan joint_v1")
        return None
    if scan == "joint_v1":
        return JointV1Policy(ab=ab)
    raise ValueError(
        f"unknown mtp verify scan {scan!r}; expected one of {POLICY_NAMES}"
    )


def _qualified_class():
    from .models.qwen3_5.language import Qwen3_5Attention

    return Qwen3_5Attention


def require_gpu():
    if not _gpu_enabled():
        _fail(f"default device is {mx.default_device()}, not the GPU; refusing")


def require_environment(*, draft_kind, kv_bits):
    if draft_kind != "mtp":
        _fail(f"requires --draft-kind mtp (got {draft_kind!r}); refusing")
    if kv_bits:
        _fail("quantized KV is not supported; refusing")
    if os.environ.get("MLX_SDPA_BLOCKS"):
        _fail("MLX_SDPA_BLOCKS is set (libmlx honours it, the plan mirror does not)")


def require_loaded_mtp_drafter(policy, draft_model, draft_kind):
    """After drafter resolution/compatibility handling: ``joint_v1`` serves only with a LOADED
    MTP drafter. A drafter that resolved to another kind, or the incompatibility fallback
    ``(None, None)``, must stop the load before READY (the verifier also serves suffix decoding)."""
    if policy is not None and (draft_model is None or draft_kind != "mtp"):
        _fail(
            f"requires a loaded MTP drafter after resolution (draft_model "
            f"{'loaded' if draft_model is not None else 'absent'}, kind {draft_kind!r}); refusing"
        )


def apply_to_model(model, policy) -> int:
    """Stamp ``policy`` on ``model``, its language model and every qwen3_5
    attention module (the verifier reads the module). Refuses an unsupported
    family. Returns the number of stamped modules."""
    qualified = _qualified_class()
    target = getattr(model, "language_model", model)
    config = getattr(model, "config", None) or getattr(target, "config", None)
    model_type = getattr(config, "model_type", None)
    targets = [m for _, m in target.named_modules() if type(m) is qualified]
    if model_type != "qwen3_5" or not targets:
        _fail(
            f"no qualified verifier call sites (model_type={model_type!r}); "
            "only the qwen3_5 family is supported"
        )
    for module in targets:
        module.mtp_verify_policy = policy
    model.mtp_verify_policy = policy
    target.mtp_verify_policy = policy
    return len(targets)


def self_test(
    policy, *, heads, kv_heads, head_dim, dtype, force_calls=None
) -> SelfTestResult:
    """Identity and known-positive proof on synthetic arrays at the model's shape.

    (1) joint vs per-query bitwise identical at the largest eligible length on
    2048 keys; (2) one straddling cell (keys 1023..1025): the mirror must
    PREDICT the mismatch and it must occur. A raise, an unpredicted mismatch, or
    a predicted mismatch that does not occur is a failure; so is a non-GPU
    default device (no skip). Counters are untouched. A hung call is bounded by
    the router's readiness timeout, not by this function: nothing here can
    interrupt a Metal call that never returns.
    """
    started = time.perf_counter()
    if force_calls is None:
        force_calls = _gpu_enabled()
    if not force_calls:
        _fail(f"default device is {mx.default_device()}, not the GPU; refusing")
    gqa = heads // kv_heads
    longest = VECTOR_QUERY_BOUND // gqa
    cache = _SelfTestCache()
    scale = head_dim**-0.5

    def cell(length, key_length):
        q = mx.random.normal((1, heads, length, head_dim)).astype(dtype)
        k = mx.random.normal((1, kv_heads, key_length, head_dim)).astype(dtype)
        v = mx.random.normal((1, kv_heads, key_length, head_dim)).astype(dtype)
        decision = policy.classify(q, k, v, cache, "causal")
        return q, k, v, decision

    def both(q, k, v, length, key_length):
        """True when the two paths differ at the representation level on VALID outputs. A shape or
        dtype error or any non-finite value ALWAYS rejects (never a "difference")."""
        try:
            joint = joint_attention(q, k, v, cache=cache, scale=scale, mask="causal")
            other = per_query_attention(q, k, v, cache=cache, scale=scale, mask=None)
            mx.eval(joint, other)
        except Exception as exc:  # noqa: BLE001 - any raise is fatal
            _fail(f"self-test failed at length={length} keys={key_length}: {exc}")
        bad = _invalid(joint, other)
        if bad is True or bool(bad.item()):
            _fail(
                f"self-test: invalid output (shape/dtype/non-finite) at length={length} "
                f"keys={key_length}"
            )
        return bool(_bits_differ(joint, other).item())

    q, k, v, decision = cell(max(longest, MIN_LENGTH), SELF_TEST_KEYS)
    if longest < MIN_LENGTH or decision.route != "joint":
        _fail(
            "self-test: no eligible cell for this model shape "
            f"(heads={heads} kv_heads={kv_heads} head_dim={head_dim} dtype={dtype}; "
            f"route={decision.route} reason={decision.reason})"
        )
    if both(q, k, v, longest, SELF_TEST_KEYS):
        _fail(
            f"self-test: joint vs per-query NOT bitwise identical at length={longest} "
            f"keys={SELF_TEST_KEYS}"
        )
    length = min(3, longest)
    key_length = SELF_TEST_STRADDLE_PREFIX + length
    q, k, v, decision = cell(length, key_length)
    predicted = decision.route == "straddle"
    occurred = both(q, k, v, length, key_length)
    if predicted and not occurred:
        _fail("self-test: straddle predicted but the mismatch did not occur")
    if occurred and not predicted:
        _fail("self-test: unpredicted mismatch at the straddle cell (mirror wrong)")
    if not predicted:
        _fail("self-test: the straddle cell is not a known positive on this device")
    elapsed = time.perf_counter() - started
    if elapsed > SELF_TEST_BUDGET_S:
        _fail(f"self-test took {elapsed:.1f}s (budget {SELF_TEST_BUDGET_S:.0f}s)")
    return SelfTestResult(2, elapsed)


class _SelfTestCache:
    """A native cache stand-in: no `bits`, not TurboQuant."""


def self_test_model(model, policy, **kwargs) -> list:
    """Run the self-test once per distinct (heads, kv_heads, head_dim, dtype),
    with the query dtype read where the attention computes it (M57 probe)."""
    from . import attention_policy as ap

    qualified = _qualified_class()
    target = getattr(model, "language_model", model)
    modules = [m for _, m in target.named_modules() if type(m) is qualified]
    dtypes = ap._observe_query_dtypes(target, modules)
    dims = []
    for module, dtype in zip(modules, dtypes):
        if dtype is None:
            _fail("self-test dtype probe saw no attention call on a qualified module")
        cell = (module.num_attention_heads, module.num_key_value_heads,
                module.head_dim, dtype)
        if cell not in dims:
            dims.append(cell)
    return [
        self_test(policy, heads=h, kv_heads=kv, head_dim=hd, dtype=dt, **kwargs)
        for h, kv, hd, dt in dims
    ]
