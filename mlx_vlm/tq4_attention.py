"""TQ4 production attention mode for qwen3_5 TurboQuant 4 caches (fork-only, M72 1.3).

Modes (registry ``tq4_attention``, CLI ``--tq4-attention``, env ``MLX_VLM_TQ4_ATTENTION``):

- ``m72_v1``: TQ4 prefill chunks through p2s (short chunks L 2..5 at T_eff > 2048 through the matrix-unit kernel), plain
  decode through D2 (T_eff > 2048) or today's single-pass kernel, MTP verify blocks k 2..5 through the matrix-unit kernel
  (joint-tolerance, not exact).
- ``m72_exact``: as ``m72_v1`` except verify blocks k 2..5 with lengths[0] > 2048 go through D2's exact multi-query kernel
  (each query bit-identical to the singleton at its own T_eff); shorter blocks take today's per-query loop.

T_eff is the key length the caller passes (``kv_sequence_length(keys_state)``), never ``cache.offset``. Out-of-scope calls
(``tq4_fast.supports`` false) run today's path and count ``fallback:<reason>``; the native last layer counts
``native:{prefill, verify, decode}`` and runs its shipped paths (``fused_v1`` prefill, ``joint_v1`` verify). The mode
absent stamps nothing: every path is today's. Shape mirrors ``mtp_verify_scan``.
"""

from __future__ import annotations

import math
import os
import sys
import time

import mlx.core as mx

from .models.base import (
    kv_sequence_length,
    slice_kv_sequence,
    tq4_scaled_dot_product_attention,
)

MODES = ("m72_v1", "m72_exact")
ENV = "MLX_VLM_TQ4_ATTENTION"
FP16_QUERY_BOUND = 2.0**14
CODEBOOK_BOUND = 1.0
SELF_TEST_BUDGET_S = 60.0


class Tq4AttentionError(RuntimeError):
    """The requested TQ4 attention mode cannot be served by this configuration/model/device."""


def _fail(message: str):
    line = f"tq4-attention: {message}"
    print(line, file=sys.stderr, flush=True)
    raise Tq4AttentionError(line)


def resolve_mode(value: str | None) -> str | None:
    """``None`` for absent / empty / ``off`` (today's behaviour); the mode name otherwise."""
    if value in (None, "", "off"):
        return None
    if value in MODES:
        return value
    raise ValueError(f"unknown tq4 attention mode {value!r}; expected one of {MODES}")


def require_environment(
    mode,
    *,
    kv_bits,
    kv_quant_scheme,
    attention_policy,
    mtp_verify_scan,
    kv_quant_mode=None,
) -> None:
    """Refuse unless kv_bits == 4, scheme turboquant, attention_policy fused_v1, mtp_verify_scan joint_v1, the MSE key
    codec and TQ_DECODE_2PASS_LEGACY off (production's parse). ``kv_quant_mode`` None: the value the served cache is
    built with (``effective_kv_quant_mode``). ``mode`` None is a no-op."""
    if mode is None:
        return
    if mode not in MODES:
        _fail(f"unknown mode {mode!r}; expected one of {MODES}")
    try:
        bits_ok = kv_bits is not None and float(kv_bits) == 4.0
    except (TypeError, ValueError):
        bits_ok = False
    if not bits_ok:
        _fail(f"{mode} requires kv_bits 4 (got {kv_bits!r}); refusing")
    if kv_quant_scheme != "turboquant":
        _fail(
            f"{mode} requires kv_quant_scheme turboquant (got {kv_quant_scheme!r}); refusing"
        )
    if attention_policy != "fused_v1":
        _fail(
            f"{mode} requires attention_policy fused_v1 (got {attention_policy!r}); refusing"
        )
    if mtp_verify_scan != "joint_v1":
        _fail(
            f"{mode} requires mtp_verify_scan joint_v1 (got {mtp_verify_scan!r}); refusing"
        )
    effective = effective_kv_quant_mode() if kv_quant_mode is None else kv_quant_mode
    if not key_codec_is_mse(effective):
        _fail(f"{mode} requires the MSE key codec (got {effective!r}); refusing")
    if legacy_decode_enabled():
        _fail(
            "TQ_DECODE_2PASS_LEGACY enables the legacy decode (D2's bit-identity contract "
            "is the non-legacy decode); refusing"
        )


def legacy_decode_enabled() -> bool:
    """``TQ_DECODE_2PASS_LEGACY`` as ``TurboQuantKVCache.__init__`` parses it (no served cache passes the argument)."""
    return os.environ.get("TQ_DECODE_2PASS_LEGACY", "0").lower() in ("1", "true", "yes")


def effective_kv_quant_mode() -> str:
    """The codec mode a served TurboQuant cache is built with: the constructor's own resolution (no served cache passes
    ``kv_quant_mode``; it reads ``TQ_KV_QUANT_MODE``, which ``--kv-quant-mode`` writes).
    """
    from . import turboquant as tq

    return tq.TurboQuantKVCache(bits=4)._kv_quant_mode


def key_codec_is_mse(kv_quant_mode) -> bool:
    """Whether ``_build_codec`` builds the MSE key codec at 4 bits for this mode."""
    from . import turboquant as tq

    if not isinstance(kv_quant_mode, str):
        return False
    probe = mx.zeros((1, 1, 1, 256), dtype=mx.float16)
    codec = tq._build_codec(probe, 4, mode=kv_quant_mode.lower(), seed=0)
    return type(codec) is tq._TurboQuantMSECodec


def _qualified_class():
    from .models.qwen3_5.language import Qwen3_5Attention

    return Qwen3_5Attention


def _attention_modules(model):
    qualified = _qualified_class()
    target = getattr(model, "language_model", model)
    return target, [(n, m) for n, m in target.named_modules() if type(m) is qualified]


def _leaf_caches(caches):
    if caches is None:
        return []
    from .speculative.cache_state import iter_leaf_caches

    return list(iter_leaf_caches(caches))


def check_fp16_range(model, caches=None) -> dict:
    """R1: per full-attention layer sqrt(D) * max|q_norm.weight| <= 2^14 (rotated queries are cast to fp16) and
    max|codebook| <= 1 (scratch values are codebook x the fp16 norm). Raises Tq4AttentionError naming the layer / value.
    ``caches``: the TurboQuant caches' own codebooks are checked; absent, the MSE codebook at 4 bits per dimension for each layer's D.
    """
    from . import turboquant as tq

    _, modules = _attention_modules(model)
    if not modules:
        _fail("no qwen3_5 full-attention modules to range-check")
    layers = {}
    for name, module in modules:
        D = int(module.head_dim)
        gamma = float(mx.max(mx.abs(module.q_norm.weight.astype(mx.float32))).item())
        bound = math.sqrt(D) * gamma
        layers[name] = {"head_dim": D, "max_abs_q_norm_weight": gamma, "q_bound": bound}
        if not bound <= FP16_QUERY_BOUND:
            _fail(
                f"R1 query range: layer {name!r} sqrt(D) * max|q_norm.weight| = {bound:.6g} "
                f"> 2^14; refusing"
            )
    books = []
    leaves = [
        c for c in _leaf_caches(caches) if isinstance(c, tq._TurboQuantAttentionMixin)
    ]
    if leaves:
        for i, c in enumerate(leaves):
            for side, codec in (("key", c.key_codec), ("value", c.value_codec)):
                if codec is None or not hasattr(codec, "codebook"):
                    continue
                books.append(
                    (
                        f"cache {i} {side}",
                        float(mx.max(mx.abs(codec.codebook.astype(mx.float32))).item()),
                    )
                )
    else:
        for name, module in modules:
            cb = tq._codebook(int(module.head_dim), 4)
            books.append(
                (
                    f"{name} (MSE codebook, 4 bits per dimension)",
                    float(mx.max(mx.abs(cb)).item()),
                )
            )
    for where, value in books:
        if not value <= CODEBOOK_BOUND:
            _fail(
                f"R1 codebook range: {where} max|codebook| = {value:.6g} > 1; refusing"
            )
    return {
        "layers": layers,
        "max_q_bound": max(v["q_bound"] for v in layers.values()),
        "max_codebook": max(v for _, v in books) if books else None,
        "q_limit": FP16_QUERY_BOUND,
        "codebook_limit": CODEBOOK_BOUND,
    }


def _is_native(cache) -> bool:
    """Native (unquantized, single-sequence) cache: the last layer's ``KVCache``."""
    from . import tq4_fast

    if isinstance(cache, tq4_fast.tq._TurboQuantAttentionMixin):
        return False
    return not tq4_fast._is_batch_cache(cache) and not hasattr(cache, "bits")


class Tq4Mode:
    """The stamped per-module hook: ``attend`` (``base.scaled_dot_product_attention``) and ``verify`` (the qwen3_5
    speculative verifier). Each returns the output, or None for today's path. Counters: ``tq4_fast.route_counts``.
    """

    def __init__(self, name: str):
        if name not in MODES:
            raise ValueError(f"unknown tq4 attention mode {name!r}")
        from . import tq4_fast

        self.name = name
        self.exact = name == "m72_exact"
        self._fast = tq4_fast

    # ---------------------------------------------------------------- prefill / decode (base.py)
    def attend(self, queries, keys, values, *, cache, scale, mask, sinks=None):
        if cache is None:
            return None
        f = self._fast
        L = queries.shape[-2]
        if _is_native(cache):
            f.count("native:prefill" if L > 1 else "native:decode")
            return None
        ok, reason = f.supports(cache, queries, mask, sinks)
        if not ok:
            f.count(f"fallback:{reason}")
            return None
        T = int(kv_sequence_length(keys))
        if L == 1:
            out = f.decode_attention(cache, queries, scale, T)
            f.count("d2" if T > f.SINGLE_PASS_MAX else "single_pass")
            return out
        if mask is None:  # a non-causal multi-query call is not a prefill chunk
            f.count("fallback:mask")
            return None
        if 2 <= L <= f.MAX_VERIFY and T > f.SINGLE_PASS_MAX:
            lengths = list(range(T - L + 1, T + 1))
            out = f.verify_attention(cache, queries, scale, lengths, exact=False)
            f.count("mma_prefill")
            return out
        out = f.prefill_attention(cache, queries, scale, T)
        f.count("p2s")
        return out

    # ---------------------------------------------------------------- verify (speculative_verifier.py)
    def verify(self, queries, keys, values, *, cache, scale, mask):
        """A verify block (length k >= 2). None: today's code runs (joint_v1 for the native layer, else the per-query
        loop, whose singleton calls route through ``attend`` with their own T_eff)."""
        if cache is None:
            return None
        f = self._fast
        if _is_native(cache):
            f.count("native:verify")
            return None
        ok, reason = f.supports(cache, queries, mask, None)
        if not ok:
            f.count(f"fallback:{reason}")
            return None
        k = queries.shape[2]
        T = int(kv_sequence_length(keys))
        lengths = [T - k + i + 1 for i in range(k)]
        return self.verify_lengths(cache, queries, scale, lengths)

    def verify_lengths(self, cache, queries, scale, lengths):
        """Route one verify block on explicit lengths; None = the per-query loop (counted)."""
        f = self._fast
        lengths = [int(x) for x in lengths]
        if len(lengths) > f.MAX_VERIFY:
            f.count("per_query:k_gt_5")
            return None
        if not f.consecutive(lengths):
            f.count("per_query:gapped")
            return None
        if self.exact:
            if lengths[0] <= f.SINGLE_PASS_MAX:
                f.count("per_query:short")
                return None
            out = f.verify_attention(cache, queries, scale, lengths, exact=True)
            f.count("d2_verify")
            return out
        out = f.verify_attention(cache, queries, scale, lengths, exact=False)
        f.count("mma")
        return out

    def verify_block(self, cache, queries, keys, values, scale, lengths):
        """``verify_lengths`` with today's per-query loop as the fallback (each query a singleton through ``attend``
        over keys sliced to its own length): always an output."""
        out = self.verify_lengths(cache, queries, scale, lengths)
        if out is not None:
            return out
        return mx.concatenate(
            [
                tq4_scaled_dot_product_attention(
                    queries[:, :, j : j + 1, :],
                    slice_kv_sequence(keys, int(L)),
                    slice_kv_sequence(values, int(L)),
                    cache=cache,
                    scale=scale,
                    mask=None,
                    tq4=self,
                )
                for j, L in enumerate(lengths)
            ],
            axis=2,
        )

    def counters(self) -> dict:
        return self._fast.route_counts()


def apply_to_model(model, mode) -> int:
    """Stamp the mode on ``model``, its language model and every qwen3_5 full-attention module (the attention call and
    the verifier read the module). ``mode``: a name in MODES, a Tq4Mode, or None (unstamp). Returns the module count.
    """
    target, modules = _attention_modules(model)
    config = getattr(model, "config", None) or getattr(target, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type != "qwen3_5" or not modules:
        _fail(
            f"no qualified attention call sites (model_type={model_type!r}); "
            "only the qwen3_5 family is supported"
        )
    hook = Tq4Mode(mode) if isinstance(mode, str) else mode
    for _, module in modules:
        module.tq4_attention = hook
    model.tq4_attention = hook
    target.tq4_attention = hook
    return len(modules)


def require_policy_stamps(model) -> int:
    """The native last layer is served by the shipped native16 paths: every full-attention module must carry the
    ``fused_v1`` attention policy and the ``joint_v1`` verify scan. Refuses otherwise; returns the module count.
    """
    _, modules = _attention_modules(model)
    for name, module in modules:
        ap = getattr(module, "attention_policy", None)
        vp = getattr(module, "mtp_verify_policy", None)
        if getattr(ap, "name", None) != "fused_v1":
            _fail(f"module {name!r} is not stamped with attention_policy fused_v1")
        if getattr(vp, "name", None) != "joint_v1":
            _fail(f"module {name!r} is not stamped with mtp_verify_scan joint_v1")
    return len(modules)


def self_test_model(model, mode) -> dict:
    """``tq4_fast.self_test`` (fragment mapping + synthetic caches vs references) under a watchdog deadline, plus the
    model's kernel scope: at least one full-attention module must have the in-scope shape (24 / 4 / 256).
    """
    from . import tq4_fast
    from .mtp_verify_scan import deadline

    name = mode.name if isinstance(mode, Tq4Mode) else mode
    if name not in MODES:
        _fail(f"unknown mode {name!r}")
    _, modules = _attention_modules(model)
    in_scope = [
        n
        for n, m in modules
        if (m.num_attention_heads, m.num_key_value_heads, m.head_dim)
        == (tq4_fast.N_Q, tq4_fast.N_KV, tq4_fast.HEAD_DIM)
    ]
    if not in_scope:
        _fail(
            "no full-attention module has the kernels' shape "
            f"(n_q={tq4_fast.N_Q}, n_kv={tq4_fast.N_KV}, D={tq4_fast.HEAD_DIM}); refusing"
        )
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        _fail(f"default device is {mx.default_device()}, not the Metal GPU; refusing")
    started = time.perf_counter()
    with deadline(SELF_TEST_BUDGET_S, what="tq4-attention self-test"):
        try:
            result = tq4_fast.self_test()
        except Exception as exc:  # noqa: BLE001 - any raise is a load failure
            _fail(f"self-test failed: {type(exc).__name__}: {exc}")
    result = dict(result)
    result.update(
        mode=name,
        modules_in_scope=len(in_scope),
        elapsed_s=time.perf_counter() - started,
    )
    return result
