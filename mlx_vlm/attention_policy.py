"""Versioned fused-attention dispatch policy for native-KV full attention.

Fork-only (M57). ``fused_v1`` forces ``force_fused=True`` on the native branch of
``models.base.scaled_dot_product_attention`` when ALL of: native cache and no
sinks; query dtype is not float32; ``qL > 8``; and ``qL >= 128`` or the score
tensor an unfused call would build is ``>= 2**28`` bytes. The constants belong
to the policy version: changing one is a new version name.
"""

import sys
import time
from typing import Optional, Tuple

import mlx.core as mx

POLICY_NAMES = ("auto", "fused_v1")
ENV = "MLX_VLM_ATTENTION_POLICY"
SELF_TEST_KEYS = 4096
SELF_TEST_BUDGET_S = 30.0


class AttentionPolicyError(RuntimeError):
    """The requested policy cannot be served by this model/device."""


class FusedV1Policy:
    name = "fused_v1"
    max_q_len_never_forced = 8
    min_q_len_always_forced = 128
    score_bytes_threshold = 2**28
    # Largest score tensor an UNFUSED call can build under this policy.
    max_unfused_score_bytes = 2**28

    def __init__(self):
        self.forced = 0
        self.auto = 0

    def should_force(self, queries, key_length, sinks) -> bool:
        """Pure decision; rule 1's cache test is the call site's (native branch)."""
        if sinks is not None or queries.dtype == mx.float32:
            return False
        q_len = queries.shape[-2]
        if q_len <= self.max_q_len_never_forced:
            return False
        if q_len >= self.min_q_len_always_forced:
            return True
        score_bytes = queries.shape[1] * q_len * key_length * queries.dtype.size
        return score_bytes >= self.score_bytes_threshold

    def decide(self, queries, key_length, sinks) -> bool:
        forced = self.should_force(queries, key_length, sinks)
        if forced:
            self.forced += 1
        else:
            self.auto += 1
        return forced

    def counters(self) -> Tuple[int, int]:
        return (self.forced, self.auto)

    snapshot = counters

    def since(self, snapshot: Tuple[int, int]) -> Tuple[int, int]:
        return (self.forced - snapshot[0], self.auto - snapshot[1])


def resolve_policy(name: Optional[str]):
    """``None`` for absent/``auto`` (today's behaviour); a policy object otherwise."""
    if name in (None, "", "auto"):
        return None
    if name == "fused_v1":
        return FusedV1Policy()
    raise ValueError(
        f"unknown attention policy {name!r}; expected one of {POLICY_NAMES}"
    )


def _qualified_class():
    from .models.qwen3_5.language import Qwen3_5Attention

    return Qwen3_5Attention


def __getattr__(name):
    # PEP 562: `QUALIFIED_ATTENTION` resolves lazily (avoids an import cycle with
    # models.qwen3_5.language at module import).
    if name == "QUALIFIED_ATTENTION":
        return _qualified_class()
    raise AttributeError(name)


def _fail(message: str):
    line = f"attention-policy fused_v1: {message}"
    print(line, file=sys.stderr, flush=True)
    raise AttentionPolicyError(line)


def apply_to_model(model, policy) -> int:
    """Stamp ``policy`` on ``model`` and on every qualified attention module.

    Qualified = exactly ``Qwen3_5Attention`` in a ``qwen3_5`` model. Returns the
    number of stamped modules; refuses (stderr line + AttentionPolicyError) on an
    unqualified family or a model with attention sinks.
    """
    qualified = _qualified_class()
    target = getattr(model, "language_model", model)
    config = getattr(model, "config", None) or getattr(target, "config", None)
    model_type = getattr(config, "model_type", None)
    modules = [m for _, m in target.named_modules()]
    sinked = [m for m in modules if getattr(m, "sinks", None) is not None]
    if sinked:
        _fail("model has attention sinks; the policy does not apply to them")
    targets = [m for m in modules if type(m) is qualified]
    if model_type != "qwen3_5" or not targets:
        _fail(
            f"no qualified attention call sites (model_type={model_type!r}); "
            "only the qwen3_5 family is supported"
        )
    for module in targets:
        module.attention_policy = policy
    model.attention_policy = policy
    target.attention_policy = policy
    return len(targets)


def _force_calls_enabled() -> bool:
    return mx.default_device() == mx.gpu


class SelfTestResult:
    def __init__(self, ran=0, skipped_reason=None, elapsed_s=0.0):
        self.ran = ran
        self.skipped_reason = skipped_reason
        self.elapsed_s = elapsed_s


def self_test(
    policy,
    *,
    heads,
    kv_heads,
    head_dim,
    dtype,
    sdpa=None,
    force_calls=None,
) -> SelfTestResult:
    """Prove the policy never forces qL 1..8 and that every forced call runs.

    Forced calls need a GPU (no CPU fused kernel): off-GPU they are SKIPPED with a
    stderr line and the pure decision assertions still run. Counters are left
    untouched (``should_force`` is pure).
    """
    started = time.perf_counter()
    scale = head_dim**-0.5
    for q_len in range(1, 9):
        probe = _Shape(heads, q_len, dtype)
        if policy.should_force(probe, 262144, None):
            _fail(f"self-test: policy forces qL={q_len}; qL 1..8 must never be forced")
    if force_calls is None:
        force_calls = _force_calls_enabled()
    if not force_calls:
        reason = f"default device is {mx.default_device()}; forced calls need a GPU"
        print(
            f"attention-policy self-test: forced-call half SKIPPED ({reason})",
            file=sys.stderr,
            flush=True,
        )
        return SelfTestResult(0, reason, time.perf_counter() - started)
    sdpa = sdpa if sdpa is not None else mx.fast.scaled_dot_product_attention
    keys = mx.random.normal((1, kv_heads, SELF_TEST_KEYS, head_dim)).astype(dtype)
    values = mx.random.normal((1, kv_heads, SELF_TEST_KEYS, head_dim)).astype(dtype)
    ran = 0
    for q_len in (9, 127, 128, 512):
        queries = mx.random.normal((1, heads, q_len, head_dim)).astype(dtype)
        if not policy.should_force(queries, SELF_TEST_KEYS, None):
            continue
        boolean = mx.arange(SELF_TEST_KEYS)[None, :] <= (
            SELF_TEST_KEYS - q_len + mx.arange(q_len)[:, None]
        )
        for mask in ("causal", None, boolean):
            try:
                out = sdpa(
                    queries, keys, values, scale=scale, mask=mask, force_fused=True
                )
                mx.eval(out)
            except Exception as exc:  # noqa: BLE001 - any raise is fatal
                _fail(f"self-test failed at qL={q_len} mask={_mask_name(mask)}: {exc}")
            ran += 1
    elapsed = time.perf_counter() - started
    if elapsed > SELF_TEST_BUDGET_S:
        _fail(f"self-test took {elapsed:.1f}s (> {SELF_TEST_BUDGET_S:.0f}s bound)")
    return SelfTestResult(ran, None, elapsed)


def _mask_name(mask) -> str:
    return mask if isinstance(mask, str) or mask is None else "array"


class _Shape:
    def __init__(self, heads, q_len, dtype):
        self.shape = (1, heads, q_len, 0)
        self.dtype = dtype


def self_test_model(model, policy, **kwargs) -> list:
    """Run the self-test once per distinct (heads, kv_heads, head_dim, dtype)."""
    qualified = _qualified_class()
    dims = []
    for _, module in getattr(model, "language_model", model).named_modules():
        if type(module) is not qualified:
            continue
        cell = (
            module.num_attention_heads,
            module.num_key_value_heads,
            module.head_dim,
            module.q_norm.weight.dtype,
        )
        if cell not in dims:
            dims.append(cell)
    return [
        self_test(
            policy, heads=h, kv_heads=kv, head_dim=hd, dtype=dt, **kwargs
        )
        for h, kv, hd, dt in dims
    ]
