"""TurboQuant 4 (TQ4) fast attention kernels for the M72 production mode (fork-only, M72 1.3).

Three kernels promoted from the dev prototypes (``dev/`` stays untouched as provenance):

- ``dequant_rotated`` / ``prefill_attention`` (p2s, ``dev/m72_scratch_prefill.py``): one Metal launch dequantizes a
  layer's keys and values for tokens ``[0, T)`` into two capacity-sized fp16 scratch arrays in the codecs' rotated space;
  MLX's fused SDPA attends with queries rotated by the key codec (fp16) and ``mask="causal"`` (queries aligned to the last
  L keys); the output is rotated back by the value codec in fp32.
- ``verify_attention`` (matrix-unit verify, ``dev/m72_mma_verify.py``, TQ4 loader only): one launch serves a k-token verify
  block (query j attends to ``[0, lengths[j])``) with simdgroup 8x8 matrix multiply-accumulate over KV tiles, 512 tokens
  per block (``VERIFY_BLOCK_TOKENS``), combined by the production pass 2. Joint-tolerance, not exact.
- ``decode_attention`` / ``verify_attention(exact=True)`` (D2, ``dev/m72_mq_decode.py``): the production 2-pass decode's
  pass 1 reading the cache's full-cap state in place, per query bit-identical to today's singleton at the query's own
  T_eff (same per-(query, head, block) arithmetic order, same block count: ``production_num_blocks``).

Every kernel reads the cache's full-capacity state arrays (``cache.keys`` / ``cache.values``; per-head stride = capacity)
plus an explicit T_eff, never the ``state`` property's strided slice (the slice forces a per-call layout copy).

R2 (empty-block sentinel): pass-1 partials of a (row, block) pair with no visible token carry ``max = -1e30, sum = 0,
acc = 0``; pass 2's combine starts at ``-INFINITY`` and relies on that finite sentinel (``exp(-1e30 - m) = 0``) to stay
NaN-free. Scores below -1e30 need inputs outside the R1 fp16 bounds (``tq4_attention.check_fp16_range``).

Route counters are per process (``count`` / ``route_counts``), never reset; ``tq4_attention`` increments them.
Batch 1 only; preconditions are asserted, including ``type(cache) is TurboQuantKVCache`` at every kernel entry point.

Test instruments, not production API (A1 sweeps and negative controls; production never calls or passes them):
``mma_attention``, ``mq_decode_attention``, the ``block_tokens`` keyword and the ``_kv_head_shift`` keyword of
``prefill_attention`` / ``verify_attention`` / ``mma_attention``. Production calls ``prefill_attention``,
``decode_attention`` and ``verify_attention`` with their defaults.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache

import mlx.core as mx

from . import turboquant as tq

# The only shapes in scope: the full-attention layers of Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed.
N_Q, N_KV, HEAD_DIM = 24, 4, 256
SINGLE_PASS_MAX = 2048  # today's single-pass decode serves T <= 2048
MAX_VERIFY = 5  # verify blocks k = 2..5 take the fast kernels

# ----------------------------------------------------------------------------- route counters
_COUNTS: dict[str, int] = {}


def count(name: str) -> None:
    """Advance one route counter (per process; never reset)."""
    _COUNTS[name] = _COUNTS.get(name, 0) + 1


def route_counts() -> dict[str, int]:
    """A copy of the per-process route counters, by counter name."""
    return dict(_COUNTS)


# ----------------------------------------------------------------------------- scope predicate
def _is_batch_cache(cache) -> bool:
    if isinstance(cache, tq.BatchTurboQuantKVCache):
        return True
    from .models import cache as kv_cache

    batch_classes = tuple(
        getattr(kv_cache, n)
        for n in ("BatchKVCache", "BatchRotatingKVCache", "BatchQuantizedKVCache")
        if hasattr(kv_cache, n)
    )
    return bool(batch_classes) and isinstance(cache, batch_classes)


def supports(cache, queries, mask=None, sinks=None) -> tuple[bool, str | None]:
    """(in scope, fallback reason). True iff the cache is exactly ``TurboQuantKVCache`` with both codecs
    ``_TurboQuantMSECodec`` at 4 bits per dimension and power-of-two RHT rotation, batch 1 queries of shape
    (1, 24, L, 256) in bf16, mask None or "causal", no sinks, the non-legacy 2-pass decode and no block override.
    """
    if _is_batch_cache(cache):
        return False, "batch_cache"
    if type(cache) is not tq.TurboQuantKVCache:
        return False, "cache_class"
    kc, vc = cache.key_codec, cache.value_codec
    if not (
        isinstance(kc, tq._TurboQuantMSECodec)
        and isinstance(vc, tq._TurboQuantMSECodec)
    ):
        return False, "codec"
    if int(kc.bits) != 4 or int(vc.bits) != 4:
        return False, "bits"
    if not (kc.use_rht and vc.use_rht):
        return False, "rotation"
    if not (
        isinstance(cache.keys, tq.TurboQuantMSEState)
        and isinstance(cache.values, tq.TurboQuantMSEState)
    ):
        return False, "codec"
    if queries.ndim != 4 or queries.shape[0] != 1:
        return False, "batch"
    if (
        queries.shape[1] != N_Q
        or queries.shape[3] != HEAD_DIM
        or cache.keys.norms.shape[1] != N_KV
        or int(kc.dim) != HEAD_DIM
        or int(vc.dim) != HEAD_DIM
    ):
        return False, "shape"
    if queries.dtype != mx.bfloat16:
        return False, "dtype"
    if not (mask is None or (isinstance(mask, str) and mask == "causal")):
        return False, "mask"
    if sinks is not None:
        return False, "sinks"
    if getattr(cache, "_decode_2pass_use_legacy", False):
        return False, "legacy"
    if getattr(cache, "_decode_2pass_num_blocks", None) is not None:
        return False, "block_override"
    return True, None


def _require_cache_class(cache) -> None:
    assert type(cache) is tq.TurboQuantKVCache, (
        f"tq4_fast kernels take exactly TurboQuantKVCache (got {type(cache).__name__}); "
        "batch / subclass caches are out of scope (``supports``)"
    )


def _check_cache(cache, T):
    _require_cache_class(cache)
    keys, values = cache.keys, cache.values
    assert isinstance(keys, tq.TurboQuantMSEState) and isinstance(
        values, tq.TurboQuantMSEState
    ), "MSE codec state only"
    assert isinstance(cache.key_codec, tq._TurboQuantMSECodec) and isinstance(
        cache.value_codec, tq._TurboQuantMSECodec
    ), "MSE codec only"
    assert (
        int(cache.key_codec.bits) == 4 and int(cache.value_codec.bits) == 4
    ), "4 bits per dimension only"
    assert keys.norms.shape[0] == 1, "batch 1 only"
    D = int(cache.key_codec.dim)
    assert keys.indices.shape[-1] * 8 == D and values.indices.shape[-1] * 8 == D
    assert keys.norms.shape == values.norms.shape
    assert keys.norms.dtype == mx.float16 and values.norms.dtype == mx.float16
    assert keys.indices.dtype == mx.uint32 and values.indices.dtype == mx.uint32
    assert 1 <= T <= cache.offset, f"T = {T} outside [1, offset = {cache.offset}]"
    return keys, values, D


# ============================================================================= p2s: dequantize + fused SDPA
_DQ_THREADGROUP = 256  # threads: 8 token rows of 32 words
_DTYPE_NAMES = {mx.float16: "half", mx.bfloat16: "bfloat"}


def _dq_source(out_t: str) -> str:
    return f"""
        typedef {out_t} OT;
        threadgroup float cbk[16];
        threadgroup float cbv[16];
        uint tid = thread_position_in_threadgroup.x;
        if (tid < 16) {{
            cbk[tid] = key_codebook[tid];
            cbv[tid] = val_codebook[tid];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint gx = thread_position_in_grid.x;  // token * PW + word; grid x = T_pad * PW (T_pad <= CapOut)
        uint h = thread_position_in_grid.y;
        uint t = gx / PW;
        uint wd = gx % PW;
        uint T = (uint)tlen[0];
        uint cap = key_norms_shape[2];  // full-cap state: its capacity is the per-head stride
        ulong o = ((ulong)h * CapOut + t) * (PW * 8) + wd * 8;  // scratch: per-head stride CapOut
        if (t < T) {{
            ulong row = (ulong)h * cap + t;
            float kn = static_cast<float>(key_norms[row]);
            float vn = static_cast<float>(val_norms[row]);
            uint kw = key_packed[row * PW + wd];
            uint vw = val_packed[row * PW + wd];
            for (int n = 0; n < 8; n++) {{
                k_rot[o + n] = static_cast<OT>(cbk[(kw >> (4 * n)) & 15u] * kn);
                v_rot[o + n] = static_cast<OT>(cbv[(vw >> (4 * n)) & 15u] * vn);
            }}
        }} else {{
            // Rows [T, T_pad): zeros. MLX's fused SDPA reads masked rows past the sliced key length; a
            // recycled scratch buffer may hold NaN / Inf there, and 0 x NaN would poison the output.
            for (int n = 0; n < 8; n++) {{
                k_rot[o + n] = OT(0);
                v_rot[o + n] = OT(0);
            }}
        }}
    """


@lru_cache(maxsize=None)
def _dq_kernel(out_t: str):
    return mx.fast.metal_kernel(
        name=f"tq4_fast_dequant_rotated_{out_t}",
        input_names=[
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
            "tlen",
        ],
        output_names=["k_rot", "v_rot"],
        source=_dq_source(out_t),
    )


SCRATCH_PAD = 128  # rows: the scratch capacity and its zeroed tail round up to this


def _round_up(x: int, m: int) -> int:
    return -(-int(x) // m) * m


def scratch_capacity(cap: int) -> int:
    """The scratch's per-head rows for a state of capacity ``cap`` (one size per cache: buffer reuse)."""
    return _round_up(cap, SCRATCH_PAD)


def dequant_rotated(cache, T, *, dtype=mx.float16) -> tuple[mx.array, mx.array]:
    """(K_rot, V_rot), each (1, n_kv, scratch_capacity(capacity), D) ``dtype``: tokens [0, T) are
    X[0, h, t, d] = dtype(float(codebook[idx(h, t, d)]) * float(norm(h, t))), fp32 product, one rounding;
    rows [T, round_up(T, 128)) are zero; rows past those are not written."""
    _require_cache_class(cache)
    keys, values, D = _check_cache(cache, int(T))
    assert (
        dtype in _DTYPE_NAMES
    ), f"dtype {dtype} unsupported: float16 (default) or bfloat16 only"
    n_kv, cap = keys.norms.shape[1], keys.norms.shape[2]
    pw = keys.indices.shape[-1]
    cap_out = scratch_capacity(cap)
    t_pad = min(_round_up(T, SCRATCH_PAD), cap_out)
    kb, vb = cache.key_codec.codebook, cache.value_codec.codebook
    assert kb.dtype == mx.float32 and vb.dtype == mx.float32 and kb.size == 16
    tlen = mx.array([int(T)], dtype=mx.int32)
    k_rot, v_rot = _dq_kernel(_DTYPE_NAMES[dtype])(
        inputs=[keys.norms, keys.indices, kb, values.norms, values.indices, vb, tlen],
        template=[("PW", pw), ("CapOut", cap_out)],
        grid=(t_pad * pw, n_kv, 1),
        threadgroup=(_DQ_THREADGROUP, 1, 1),
        output_shapes=[(1, n_kv, cap_out, D), (1, n_kv, cap_out, D)],
        output_dtypes=[dtype, dtype],
    )
    return k_rot, v_rot


def _perm(x, n_kv, shift):
    if shift % n_kv == 0:
        return x
    idx = mx.array([(h + shift) % n_kv for h in range(n_kv)], dtype=mx.int32)
    return mx.take(x, idx, axis=1)


def prefill_attention(cache, queries, scale, T, *, _kv_head_shift=0) -> mx.array:
    """p2s: queries (1, n_q, L, D) bf16 at positions [T - L, T) -> (1, n_q, L, D) ``queries.dtype``.
    ``_kv_head_shift`` rolls the KV heads SDPA sees (negative control only; 0 = identity).
    """
    _require_cache_class(cache)
    T = int(T)
    B, n_q, L, D = queries.shape
    assert B == 1, "batch 1 only"
    assert 1 <= L <= T, f"L = {L} outside [1, T = {T}]"
    k_rot, v_rot = dequant_rotated(cache, T, dtype=mx.float16)
    assert k_rot.shape[-1] == D
    n_kv = k_rot.shape[1]
    k = _perm(k_rot, n_kv, _kv_head_shift)[:, :, :T, :]
    v = _perm(v_rot, n_kv, _kv_head_shift)[:, :, :T, :]
    qr = cache.key_codec.prepare_queries(queries).astype(mx.float16)
    o = mx.fast.scaled_dot_product_attention(
        qr, k, v, scale=scale, mask="causal", **_fused_kwargs(L, n_q // n_kv)
    )
    return cache.value_codec._rotate_inverse(o.astype(mx.float32)).astype(queries.dtype)


def _fused_kwargs(L: int, gqa: int) -> dict:
    """MLX has a fused kernel for every L > 8 (full kernel) and for L * gqa <= 32 (vector kernel); a chunk in
    between (L 6..8 at GQA 6) is left to MLX's own choice — forcing it raises."""
    return {"force_fused": True} if (L > 8 or L * gqa <= 32) else {}


# ============================================================================= matrix-unit verify (TQ4 loader)
TILE_TOKENS = 16  # Tt: K + V tiles in threadgroup memory
VERIFY_BLOCK_TOKENS = 512  # tokens per block (SWEEP.md; A1 re-established on this plan)
SMEM_PAD = 8  # elements of padding per threadgroup row (bank spread for the transposed K loads)


def plan_blocks(max_length: int, block_tokens=VERIFY_BLOCK_TOKENS) -> tuple[int, int]:
    """(Blocks, tokens per block): every block holds >= 1 whole tile and starts below ``max_length``."""
    assert max_length >= 1
    tpb = max(TILE_TOKENS, -(-int(block_tokens) // TILE_TOKENS) * TILE_TOKENS)
    return -(-max_length // tpb), tpb


def _mma_loader():
    return [
        "        // TQ4 loader: indices at 4 bits per dimension, LSB-first (dim d at bits [4d, 4d + 4) of the row's words) -> unit values.",
        "        for (uint w = tid; w < Tt * PW; w += NT) {",
        "            uint tt = w / PW;",
        "            uint wd = w % PW;",
        "            uint t = ts + tt;",
        "            bool in = t < Lmax;",
        "            uint kw = in ? key_packed[((ulong)kvh * cap + t) * PW + wd] : 0u;",
        "            uint vw = in ? val_packed[((ulong)kvh * cap + t) * PW + wd] : 0u;",
        "            threadgroup QT* kd = Ks + tt * SK + wd * 8;",
        "            threadgroup PT* vd = Vs + tt * SK + wd * 8;",
        "            for (int n = 0; n < 8; n++) {",
        "                kd[n] = in ? cbk[(kw >> (4 * n)) & 15u] : QT(0);",
        "                vd[n] = in ? cbv[(vw >> (4 * n)) & 15u] : PT(0);",
        "            }",
        "        }",
        "        if (tid < Tt) {",
        "            uint t = ts + tid;",
        "            bool in = t < Lmax;",
        "            kfac[tid] = in ? static_cast<float>(key_norms[(ulong)kvh * cap + t]) : 0.0f;",
        "            vfac[tid] = in ? static_cast<float>(val_norms[(ulong)kvh * cap + t]) : 0.0f;",
        "        }",
    ]


@lru_cache(maxsize=None)
def _mma_kernel(rows: int, n_queries: int):
    """Pass-1 kernel for (rows per KV head, k). Template: Dim, Tt, TPB, Blocks, Shift, PW."""
    ns = (rows + 7) // 8
    L = [
        "        constexpr int D = Dim;",
        f"        constexpr int M = {rows};",
        f"        constexpr int K = {n_queries};",
        f"        constexpr uint NT = {ns * 32};",
        f"        constexpr int SK = D + {SMEM_PAD};",
        "        constexpr int NC = Tt / 8;",
        "        constexpr int ND = D / 8;",
        "        constexpr float NEG = -1e30f;  // R2: production pass 1's empty-block sentinel",
        "        typedef half QT;",
        "        typedef half PT;",
        "        threadgroup QT Ks[Tt * SK];",
        "        threadgroup PT Vs[Tt * SK];",
        "        threadgroup float kfac[Tt];",
        "        threadgroup float vfac[Tt];",
        "        uint blk = threadgroup_position_in_grid.x;",
        "        uint h = threadgroup_position_in_grid.y;",
        "        uint sg = simdgroup_index_in_threadgroup;",
        "        uint lane = thread_index_in_simdgroup;",
        "        uint tid = thread_position_in_threadgroup.x;",
        "        uint cap = key_norms_shape[2];  // full-cap buffer: its capacity is the per-head stride",
        "        uint kvh = (h + Shift) % key_norms_shape[1];",
        "        int qd = lane / 4;",
        "        int fm = (qd & 4) + ((lane / 2) % 4);",
        "        int fn = (qd & 2) * 2 + (lane % 2) * 2;",
        "        int r = sg * 8 + fm;",
        "        bool row_ok = r < M;",
        "        uint len_r = row_ok ? (uint)lengths[r % K] : 0u;",
        "        uint Lmax = (uint)lengths[K - 1];",
        "        uint t0 = blk * TPB;",
        "        uint t1 = min(t0 + TPB, Lmax);",
        "        threadgroup QT cbk[16];",
        "        threadgroup PT cbv[16];",
        "        if (tid < 16) {",
        "            cbk[tid] = static_cast<QT>(key_codebook[tid]);",
        "            cbv[tid] = static_cast<PT>(val_codebook[tid]);",
        "        }",
        "        // Q fragments in registers (rows >= M are zero and never written).",
        "        simdgroup_matrix<QT, 8, 8> qf[ND];",
        '        _Pragma("clang loop unroll(full)")',
        "        for (int dk = 0; dk < ND; dk++) {",
        "            for (int e = 0; e < 2; e++) {",
        "                float x = row_ok ? static_cast<float>(queries[((ulong)h * M + r) * D + dk * 8 + fn + e]) : 0.0f;",
        "                qf[dk].thread_elements()[e] = static_cast<QT>(x);",
        "            }",
        "        }",
        "        simdgroup_matrix<float, 8, 8> of[ND];",
        '        _Pragma("clang loop unroll(full)")',
        "        for (int dn = 0; dn < ND; dn++) of[dn] = simdgroup_matrix<float, 8, 8>(0);",
        "        float m_run = NEG;",
        "        float l_run = 0.0f;",
        "        for (uint ts = t0; ts < t1; ts += Tt) {",
        "            threadgroup_barrier(mem_flags::mem_threadgroup);",
    ]
    L += ["    " + s for s in _mma_loader()]
    L += [
        "            threadgroup_barrier(mem_flags::mem_threadgroup);",
        "            simdgroup_matrix<float, 8, 8> sf[NC];",
        '            _Pragma("clang loop unroll(full)")',
        "            for (int c = 0; c < NC; c++) {",
        "                sf[c] = simdgroup_matrix<float, 8, 8>(0);",
        '                _Pragma("clang loop unroll(full)")',
        "                for (int dk = 0; dk < ND; dk++) {",
        "                    simdgroup_matrix<QT, 8, 8> kt;",
        "                    simdgroup_load(kt, Ks + c * 8 * SK + dk * 8, SK, ulong2(0, 0), true);",
        "                    simdgroup_multiply_accumulate(sf[c], qf[dk], kt, sf[c]);",
        "                }",
        "            }",
        "            float sc[NC][2];",
        "            bool ok[NC][2];",
        "            float mt = NEG;",
        "            for (int c = 0; c < NC; c++) {",
        "                for (int e = 0; e < 2; e++) {",
        "                    int tt = c * 8 + fn + e;",
        "                    ok[c][e] = ts + tt < len_r;",
        "                    sc[c][e] = ok[c][e] ? sf[c].thread_elements()[e] * kfac[tt] : NEG;",
        "                    mt = max(mt, sc[c][e]);",
        "                }",
        "            }",
        "            mt = max(mt, simd_shuffle_xor(mt, 1));",
        "            mt = max(mt, simd_shuffle_xor(mt, 8));",
        "            float m_new = max(m_run, mt);",
        "            float fac = fast::exp(m_run - m_new);  // NEG - finite -> 0; NEG - NEG -> 1 (O, l still 0)",
        "            float rs = 0.0f;",
        "            simdgroup_matrix<PT, 8, 8> pf[NC];",
        "            for (int c = 0; c < NC; c++) {",
        "                for (int e = 0; e < 2; e++) {",
        "                    int tt = c * 8 + fn + e;",
        "                    float p = ok[c][e] ? fast::exp(sc[c][e] - m_new) : 0.0f;",
        "                    rs += p;",
        "                    pf[c].thread_elements()[e] = static_cast<PT>(p * vfac[tt]);",
        "                }",
        "            }",
        "            rs += simd_shuffle_xor(rs, 1);",
        "            rs += simd_shuffle_xor(rs, 8);",
        "            l_run = l_run * fac + rs;",
        "            m_run = m_new;",
        '            _Pragma("clang loop unroll(full)")',
        "            for (int dn = 0; dn < ND; dn++) {",
        "                of[dn].thread_elements()[0] *= fac;",
        "                of[dn].thread_elements()[1] *= fac;",
        '                _Pragma("clang loop unroll(full)")',
        "                for (int c = 0; c < NC; c++) {",
        "                    simdgroup_matrix<PT, 8, 8> vf;",
        "                    simdgroup_load(vf, Vs + c * 8 * SK + dn * 8, SK);",
        "                    simdgroup_multiply_accumulate(of[dn], pf[c], vf, of[dn]);",
        "                }",
        "            }",
        "        }",
        "        if (row_ok) {",
        "            bool empty = m_run == NEG;  // no visible token of this row in this block",
        "            ulong base = ((ulong)h * M + r) * Blocks + blk;",
        '            _Pragma("clang loop unroll(full)")',
        "            for (int dn = 0; dn < ND; dn++) {",
        "                out_acc[base * D + dn * 8 + fn] = empty ? 0.0f : of[dn].thread_elements()[0];",
        "                out_acc[base * D + dn * 8 + fn + 1] = empty ? 0.0f : of[dn].thread_elements()[1];",
        "            }",
        "            if (fn == 0) {",
        "                out_sums[base] = empty ? 0.0f : l_run;",
        "                out_maxs[base] = empty ? NEG : m_run;",
        "            }",
        "        }",
    ]
    return mx.fast.metal_kernel(
        name=f"tq4_fast_mma_verify_m{rows}_k{n_queries}",
        input_names=[
            "queries",
            "lengths",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
        ],
        output_names=["out_acc", "out_sums", "out_maxs"],
        source="\n".join(L),
    )


def _pass2(acc, sums, maxs, n_rows, D, nb):
    return tq._fused_mse_decode_2pass_2_kernel()(
        inputs=[acc, sums, maxs],
        template=[("Dim", D), ("Blocks", nb)],
        grid=(n_rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(n_rows, D)],
        output_dtypes=[mx.float32],
    )[0]


def _mma_verify(cache, queries, scale, lengths, block_tokens, shift) -> mx.array:
    keys, values = cache.keys, cache.values  # the full-cap preallocated states
    B, n_q, k, D = queries.shape
    n_kv = keys.norms.shape[1]
    R = n_q // n_kv
    rows = R * k
    # Same prep as the singleton (elementwise scale, row-wise rotation), grouped (B, n_kv, R, k, D): row = i * k + j.
    grouped = (queries * scale).reshape(B, n_kv, R, k, D)
    q_rows = cache.key_codec.prepare_queries(grouped).reshape(n_kv * rows, D)
    nb, tpb = plan_blocks(int(lengths[-1]), block_tokens)
    lens = mx.array([int(x) for x in lengths], dtype=mx.int32)
    ns = (rows + 7) // 8
    acc, sums, maxs = _mma_kernel(rows, k)(
        inputs=[
            q_rows,
            lens,
            keys.norms,
            keys.indices,
            cache.key_codec.codebook,
            values.norms,
            values.indices,
            cache.value_codec.codebook,
        ],
        template=[
            ("Dim", D),
            ("Tt", TILE_TOKENS),
            ("TPB", tpb),
            ("Blocks", nb),
            ("Shift", int(shift) % n_kv),
            ("PW", int(keys.indices.shape[-1])),
        ],
        grid=(ns * 32 * nb, n_kv, 1),
        threadgroup=(ns * 32, 1, 1),
        output_shapes=[(n_kv * rows * nb, D), (n_kv * rows * nb,), (n_kv * rows * nb,)],
        output_dtypes=[mx.float32] * 3,
    )
    out = _pass2(acc, sums, maxs, n_kv * rows, D, nb)
    out = cache.value_codec._rotate_inverse(out.reshape(B, n_kv, R, k, D))
    return out.reshape(B, n_q, k, D).astype(queries.dtype)


# ============================================================================= D2: copy-free exact decode
def production_num_blocks(total_tokens: int, n_repeats: int) -> int:
    """The shipped dispatch's block count (``TurboQuantKVCache.decode_attention``, non-legacy 2-pass branch)."""
    if total_tokens <= 8192:
        nb = 64
    elif total_tokens <= 32768:
        nb = 128
    elif total_tokens <= 65536:
        nb = 256
    else:
        nb = 512
    return nb * (2 if n_repeats % 2 == 0 else 1)


@lru_cache(maxsize=None)
def _d2_pass1_kernel(
    key_bits: int, val_bits: int, dim: int, heads_per_group: int, n_queries: int
):
    G, NQ = heads_per_group, n_queries
    elems = dim // 32
    k_off = "k_bit_off" if (elems * key_bits) % 8 else ""
    v_off = "v_bit_off" if (elems * val_bits) % 8 else ""
    key_exprs = tq._gen_unrolled_extract(key_bits, elems, "key_codebook", k_off)
    val_exprs = [
        e.replace("kb[", "vb[")
        for e in tq._gen_unrolled_extract(val_bits, elems, "val_codebook", v_off)
    ]
    L = [
        "        constexpr int BD = 32;",
        "        constexpr int qk_per_thread = Dim / BD;",
        "        constexpr int v_per_thread = Dim / BD;",
        f"        constexpr int G = {G};",
        "        typedef float U;",
        "        uint gid = threadgroup_position_in_grid.x;",
        "        auto simd_lid = thread_index_in_simdgroup;",
        "        // The full-cap buffer; its capacity is the per-head stride. Never read past lengths[].",
        "        auto cap = key_norms_shape[2];",
        "        auto kv_heads = key_norms_shape[1];",
        "        uint BQH = key_norms_shape[0] * kv_heads * RepeatCount;",
        "        uint groups = RepeatCount / G;",
        "        uint block_idx = gid % Blocks;",
        "        uint rest = gid / Blocks;",
        "        uint grp = rest % groups;",
        "        uint rest2 = rest / groups;",
        "        uint kv_head_idx = rest2 % kv_heads;",
        "        uint batch_idx = rest2 / kv_heads;",
        "        auto bh = batch_idx * kv_heads + kv_head_idx;",
        "        uint qbase = (batch_idx * kv_heads + kv_head_idx) * RepeatCount + grp * G;",
        "        auto k_nm = key_norms + bh * cap;",
        "        auto k_pk = key_packed + bh * cap * KPackedWidth;",
        "        auto v_nm = val_norms + bh * cap;",
        "        auto v_pk = val_packed + bh * cap * VPackedWidth;",
    ]
    for j in range(NQ):
        L.append(f"        uint T_{j} = (uint)lengths[{j}];")
        for r in range(G):
            L += [
                f"        thread U q_{j}_{r}[qk_per_thread];",
                f"        {{ auto qr = queries + ({j} * BQH + qbase + {r}) * Dim + simd_lid * qk_per_thread;",
                f"          for (int i = 0; i < qk_per_thread; i++) q_{j}_{r}[i] = static_cast<U>(qr[i]); }}",
                f"        thread U o_{j}_{r}[v_per_thread] = {{}};",
                f"        U max_{j}_{r} = -INFINITY;",
                f"        U sum_{j}_{r} = 0;",
            ]
    L += [
        f"        int k_bit_start = simd_lid * qk_per_thread * {key_bits};",
        f"        int v_bit_start = simd_lid * v_per_thread * {val_bits};",
        "        int k_byte_base = k_bit_start >> 3;",
        "        int v_byte_base = v_bit_start >> 3;",
    ]
    if k_off:
        L.append("        int k_bit_off = k_bit_start & 7;")
    if v_off:
        L.append("        int v_bit_off = v_bit_start & 7;")
    L += [
        "        // One class: every query shares Blocks and tpb (dispatcher-guaranteed); lengths ascending.",
        "        uint tpb = (T_0 + Blocks - 1) / Blocks;",
        "        uint t0 = block_idx * tpb;",
        f"        uint t1 = min(t0 + tpb, T_{NQ - 1});",
        "        for (uint t = t0; t < t1; t++) {",
        "            U kn = static_cast<U>(k_nm[t]);",
        "            U vn = static_cast<U>(v_nm[t]);",
        "            auto kb = (const device uint8_t*)(k_pk + t * KPackedWidth) + k_byte_base;",
        "            auto vb = (const device uint8_t*)(v_pk + t * VPackedWidth) + v_byte_base;",
        "            U kd[qk_per_thread];",
        "            U vd[v_per_thread];",
    ]
    L += [f"            kd[{i}] = {e};" for i, e in enumerate(key_exprs)]
    L += [f"            vd[{i}] = {e};" for i, e in enumerate(val_exprs)]
    for j in range(NQ):
        L.append(
            f"            if (t < T_{j}) {{"
            if j < NQ - 1
            else "            {  // t < t1 <= T_last"
        )
        for r in range(G):
            s = f"{j}_{r}"
            L += [
                f"            U score_{s} = 0;",
                f"            for (int i = 0; i < qk_per_thread; i++) score_{s} += q_{s}[i] * kd[i];",
                f"            score_{s} = simd_sum(score_{s}) * kn;",
                f"            U nm_{s} = max(max_{s}, score_{s});",
                f"            U fac_{s} = fast::exp(max_{s} - nm_{s});",
                f"            U es_{s} = fast::exp(score_{s} - nm_{s});",
                f"            max_{s} = nm_{s};",
                f"            sum_{s} = sum_{s} * fac_{s} + es_{s};",
                f"            for (int i = 0; i < v_per_thread; i++) o_{s}[i] = o_{s}[i] * fac_{s} + es_{s} * vd[i] * vn;",
            ]
        L.append("            }")
    L.append("        }")
    for j in range(NQ):
        L.append(f"        bool empty_{j} = min(t0 + tpb, T_{j}) <= t0;")
        for r in range(G):
            s = f"{j}_{r}"
            L += [
                f"        {{ uint qh = {j} * BQH + qbase + {r};",
                "          if (simd_lid == 0) {",
                f"              out_sums[qh * Blocks + block_idx] = sum_{s};",
                f"              out_maxs[qh * Blocks + block_idx] = empty_{j} ? -1e30f : max_{s};",
                "          }",
                "          for (int i = 0; i < v_per_thread; i++)",
                f"              out_acc[(qh * Blocks + block_idx) * Dim + simd_lid * v_per_thread + i] = static_cast<U>(o_{s}[i]);",
                "        }",
            ]
    return mx.fast.metal_kernel(
        name=f"tq4_fast_d2_pass1_k{key_bits}_v{val_bits}_d{dim}_g{G}_q{NQ}",
        input_names=[
            "queries",
            "lengths",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
        ],
        output_names=["out_acc", "out_sums", "out_maxs"],
        source="\n".join(L),
    )


def _d2_classes(lengths, n_repeats):
    """Consecutive runs of queries sharing (Blocks, tpb) — the singleton's partition key."""
    out = []
    for j, T in enumerate(lengths):
        nb = production_num_blocks(T, n_repeats)
        key = (nb, -(-T // nb))
        if out and out[-1][0] == key:
            out[-1][1].append(j)
        else:
            out.append((key, [j]))
    return out


def _d2(cache, queries, scale, lengths) -> mx.array:
    keys, values = cache.keys, cache.values  # full-cap preallocated states
    B, n_q, k, D = queries.shape
    n_kv = keys.norms.shape[1]
    R = n_q // n_kv
    G = 2 if R % 2 == 0 else 1
    kb, vb = int(cache.key_codec.bits), int(cache.value_codec.bits)
    # Same prep as the singleton (elementwise scale, row-wise Hadamard), batched over the k queries.
    grouped = (queries * scale).reshape(B, n_kv, R, k, D)
    q_rot = cache.key_codec.prepare_queries(grouped)
    BQH = B * n_q
    outs = []
    for (nb, _tpb), idx in _d2_classes(lengths, R):
        nq = len(idx)
        q_cls = mx.transpose(
            q_rot[:, :, :, idx[0] : idx[-1] + 1, :], (3, 0, 1, 2, 4)
        ).reshape(nq * BQH, D)
        lens = mx.array([int(lengths[j]) for j in idx], dtype=mx.int32)
        acc, sums, maxs = _d2_pass1_kernel(kb, vb, D, G, nq)(
            inputs=[
                q_cls,
                lens,
                keys.norms,
                keys.indices,
                cache.key_codec.codebook,
                values.norms,
                values.indices,
                cache.value_codec.codebook,
            ],
            template=[
                ("Dim", D),
                ("RepeatCount", R),
                ("Blocks", nb),
                ("KPackedWidth", keys.indices.shape[-1]),
                ("VPackedWidth", values.indices.shape[-1]),
            ],
            grid=(B * n_kv * (R // G) * nb * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(nq * BQH * nb, D), (nq * BQH * nb,), (nq * BQH * nb,)],
            output_dtypes=[mx.float32] * 3,
        )
        out = _pass2(acc, sums, maxs, nq * BQH, D, nb)
        out = cache.value_codec._rotate_inverse(out.reshape(nq, B, n_kv, R, D))
        outs.append(mx.transpose(out.reshape(nq, B, n_q, D), (1, 2, 0, 3)))
    out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
    return out.astype(queries.dtype)


def decode_attention(cache, queries, scale, T) -> mx.array:
    """queries (1, n_q, 1, D) attending to tokens [0, T): D2 on the full-cap state in place when T > 2048 (bit-identical
    to today's singleton at T), else today's single-pass kernel on the state sliced to T (today's call).
    """
    _require_cache_class(cache)
    T = int(T)
    B, n_q, L, D = queries.shape
    assert B == 1 and L == 1, "batch 1, one query"
    _check_cache(cache, T)
    if T > SINGLE_PASS_MAX:
        return _d2(cache, queries, scale, [T])
    return cache.decode_attention(
        queries,
        keys_state=tq._slice_state(cache.keys, T),
        values_state=tq._slice_state(cache.values, T),
        scale=scale,
        mask=None,
    )


def consecutive(lengths) -> bool:
    return all(int(b) == int(a) + 1 for a, b in zip(lengths, lengths[1:]))


def verify_attention(
    cache,
    queries,
    scale,
    lengths,
    *,
    exact=False,
    block_tokens=VERIFY_BLOCK_TOKENS,
    _kv_head_shift=0,
) -> mx.array:
    """A verify block: queries (1, n_q, k, D), k = 2..5, query j attends to tokens [0, lengths[j]); lengths consecutive.
    ``exact`` -> D2 multi-query (each query bit-identical to the singleton at its own T_eff; lengths[0] > 2048), else the
    matrix-unit kernel (joint-tolerance). ``block_tokens`` / ``_kv_head_shift`` are instrument knobs (matrix-unit only).
    """
    _require_cache_class(cache)
    B, n_q, k, D = queries.shape
    lengths = [int(x) for x in lengths]
    assert B == 1, "batch 1 only"
    assert 2 <= k <= MAX_VERIFY and len(lengths) == k, f"k = {k} outside 2..5"
    assert consecutive(lengths), f"lengths {lengths} not consecutive"
    assert lengths[0] >= 1
    _check_cache(cache, lengths[-1])
    if exact:
        assert (
            lengths[0] > SINGLE_PASS_MAX
        ), "exact verify needs lengths[0] > 2048 (shorter blocks take the per-query loop)"
        return _d2(cache, queries, scale, lengths)
    return _mma_verify(cache, queries, scale, lengths, block_tokens, _kv_head_shift)


def mma_attention(
    cache,
    queries,
    scale,
    lengths,
    *,
    block_tokens=VERIFY_BLOCK_TOKENS,
    _kv_head_shift=0,
):
    """The matrix-unit kernel on any ascending lengths (k = 1..5, gapped allowed): the instrument form of
    ``verify_attention(exact=False)`` (A1 sweeps and negative controls)."""
    _require_cache_class(cache)
    B, n_q, k, D = queries.shape
    lengths = [int(x) for x in lengths]
    assert B == 1 and 1 <= k <= MAX_VERIFY and len(lengths) == k
    assert lengths == sorted(lengths) and lengths[0] >= 1
    _check_cache(cache, lengths[-1])
    return _mma_verify(cache, queries, scale, lengths, block_tokens, _kv_head_shift)


def mq_decode_attention(cache, queries, scale, lengths) -> mx.array:
    """D2 on any ascending lengths > 2048 (k = 1..5): the instrument form of ``verify_attention(exact=True)``."""
    _require_cache_class(cache)
    lengths = [int(x) for x in lengths]
    assert queries.shape[0] == 1 and queries.shape[2] == len(lengths)
    assert lengths == sorted(lengths) and lengths[0] > SINGLE_PASS_MAX
    _check_cache(cache, lengths[-1])
    return _d2(cache, queries, scale, lengths)


# ============================================================================= self-test
def fragment_mapping_ok() -> bool:
    """True when the 8x8 fragment's thread elements follow the mapping the matrix-unit kernel relies on (this GPU)."""
    src = """
        uint lane = thread_index_in_simdgroup;
        simdgroup_matrix<float, 8, 8> m;
        simdgroup_load(m, inp, 8);
        out[lane * 2] = m.thread_elements()[0];
        out[lane * 2 + 1] = m.thread_elements()[1];
    """
    kern = mx.fast.metal_kernel(
        name="tq4_fast_fragment_map",
        input_names=["inp"],
        output_names=["out"],
        source=src,
    )
    got = kern(
        inputs=[mx.arange(64, dtype=mx.float32)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(64,)],
        output_dtypes=[mx.float32],
    )[0].tolist()
    for lane in range(32):
        q = lane // 4
        fm = (q & 4) + (lane // 2) % 4
        fn = (q & 2) * 2 + (lane % 2) * 2
        if [int(got[2 * lane]), int(got[2 * lane + 1])] != [
            fm * 8 + fn,
            fm * 8 + fn + 1,
        ]:
            return False
    return True


class Tq4SelfTestError(RuntimeError):
    """A kernel self-test cell failed."""


SELF_TEST_SEED = 72
SELF_TEST_T = 4100
SELF_TEST_CAP = 4352


def _bits16(x):
    return mx.contiguous(x).view(mx.uint16)


def _rel(a, b) -> float:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float((mx.max(mx.abs(a - b)) / mx.maximum(mx.max(mx.abs(b)), 1e-30)).item())


def _finite(x) -> bool:
    return bool(mx.all(mx.isfinite(x.astype(mx.float32))).item())


def _singleton(cache, q, scale, T):
    """Today's TQ4 singleton decode on the state sliced to T (the copying call base.py makes)."""
    return cache.decode_attention(
        q,
        keys_state=tq._slice_state(cache.keys, T),
        values_state=tq._slice_state(cache.values, T),
        scale=scale,
        mask=None,
    )


def self_test() -> dict:
    """Small synthetic TQ4 cache (4,100 tokens in a 4,352-token buffer) against references: fragment mapping; D2 decode
    and D2 multi-query bit-identical to today's singleton; matrix-unit verify (incl. a fully masked block) and p2s within
    2^-6 of today's per-query / production path, finite. Local RNG keys only (the global stream is untouched).
    """
    started = time.perf_counter()
    if not fragment_mapping_ok():
        raise Tq4SelfTestError(
            "fragment element mapping differs on this GPU (matrix-unit verify)"
        )
    ks = mx.random.split(mx.random.key(SELF_TEST_SEED), 4)
    T, cap = SELF_TEST_T, SELF_TEST_CAP
    K = mx.random.normal((1, N_KV, T, HEAD_DIM), key=ks[0]).astype(mx.bfloat16)
    V = mx.random.normal((1, N_KV, T, HEAD_DIM), key=ks[1]).astype(mx.bfloat16)
    cache = tq.TurboQuantKVCache(bits=4, max_kv_size=cap, prealloc_tokens=cap)
    cache.update_and_fetch(K, V)
    mx.eval(cache.keys, cache.values)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    q5 = mx.random.normal((1, N_Q, 5, HEAD_DIM), key=ks[2]).astype(mx.bfloat16)
    cells = 0
    # D2 decode at T_eff above / straddling the single-pass bound, incl. a state sliced below the offset
    for t_eff in (2049, T - 7, T):
        q = q5[:, :, :1, :]
        got, want = decode_attention(cache, q, scale, t_eff), _singleton(
            cache, q, scale, t_eff
        )
        mx.eval(got, want)
        if not bool(mx.all(_bits16(got) == _bits16(want)).item()):
            raise Tq4SelfTestError(
                f"D2 decode not bit-identical to the singleton at T={t_eff}"
            )
        cells += 1
    # D2 multi-query (exact verify), k = 3
    lengths = [T - 2, T - 1, T]
    got = verify_attention(cache, q5[:, :, :3, :], scale, lengths, exact=True)
    want = mx.concatenate(
        [
            _singleton(cache, q5[:, :, j : j + 1, :], scale, L)
            for j, L in enumerate(lengths)
        ],
        axis=2,
    )
    mx.eval(got, want)
    if not bool(mx.all(_bits16(got) == _bits16(want)).item()):
        raise Tq4SelfTestError("D2 multi-query not bit-identical to the singletons")
    cells += 1
    # matrix-unit verify, k = 3 at the cache end, and a fully masked block (lengths 512 / 513 -> block 1 empty for row 0)
    for lens in ([T - 2, T - 1, T], [512, 513]):
        k = len(lens)
        got = verify_attention(cache, q5[:, :, :k, :], scale, lens)
        want = mx.concatenate(
            [
                _singleton(cache, q5[:, :, j : j + 1, :], scale, L)
                for j, L in enumerate(lens)
            ],
            axis=2,
        )
        mx.eval(got, want)
        if not _finite(got) or _rel(got, want) > 2**-6:
            raise Tq4SelfTestError(
                f"matrix-unit verify off the per-query path at lengths {lens}"
            )
        cells += 1
    # p2s: a 37-query chunk at the cache end vs today's production prefill
    L = 37
    qp = mx.random.normal((1, N_Q, L, HEAD_DIM), key=ks[3]).astype(mx.bfloat16)
    got = prefill_attention(cache, qp, scale, T)
    want = cache.quantized_attention(
        qp,
        keys_state=tq._slice_state(cache.keys, T),
        values_state=tq._slice_state(cache.values, T),
        scale=scale,
        mask="causal",
    )
    mx.eval(got, want)
    if not _finite(got) or _rel(got, want) > 2**-6:
        raise Tq4SelfTestError("p2s prefill off the production prefill")
    cells += 1
    return {"cells": cells, "elapsed_s": time.perf_counter() - started}
