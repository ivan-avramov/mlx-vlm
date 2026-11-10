"""M72 P2' stage 1 prototype: fused TQ4 dequantize into a per-layer scratch + MLX fused SDPA (dev only, never
imported by ``mlx_vlm/``; C314).

One Metal launch dequantizes a layer's TurboQuant MSE keys and values (4 bits per dimension) for tokens ``[0, T)`` into two cap-sized
scratch arrays in the codecs' rotated space (unit codebook value x norm; no inverse rotation). MLX's own fused SDPA then
attends with queries rotated once by the key codec, and the output is rotated back once by the value codec (exact:
q.k = (qR).(kR) and sum_t w_t v_t = (sum_t w_t v_t R) R^T for each codec's orthogonal R). Same arguments to SDPA as the
probe's P1 (``rot_fused()``); only the dequantize is fused.

Kernel: the cache's full-cap preallocated state is read in place (``norms`` (1, n_kv, cap) fp16, ``indices``
(1, n_kv, cap, 32) uint32; dim d at bits [4d, 4d + 4) of a token's words, LSB first; per-head stride = capacity).
Thread mapping: one thread per (KV head, token, packed word); a simdgroup = one token row (32 words = 128 contiguous
bytes read per operand, 512 contiguous output bytes written per operand); grid x = T * 32, so tokens >= T are never
touched and T never enters the kernel source (one compiled kernel for every T). Codebook: the two fp32 codebooks staged
in threadgroup memory as fp32 and read as fp32 at the multiply; the product norm x centroid in fp32, one rounding to the
output dtype — the same arithmetic as the generic-op reference, hence bit-identical.

Memory contract: each call returns fresh cap-sized outputs (one size for every chunk, so MLX's buffer cache recycles the
previous layer's / chunk's freed scratch). The caller must evaluate once per full-attention layer during TQ4 prefill;
one lazy multi-layer graph keeps several scratches alive at once.
"""

from functools import lru_cache

import mlx.core as mx

from mlx_vlm import turboquant as tq

THREADGROUP = 256  # threads: 8 token rows of 32 words
THREAD_MAPPING = (
    "one thread per (KV head, token, packed uint32 word = 8 dims); a simdgroup = one token row "
    "(coalesced 128-byte reads, 512-byte fp16 writes per operand); grid (T * 32, n_kv), threadgroup 256"
)
CODEBOOK_PLACEMENT = "both fp32 codebooks staged in threadgroup memory as fp32 (16 + 16 floats), read as fp32 at the multiply"
_DTYPE_NAMES = {mx.float16: "half", mx.bfloat16: "bfloat"}


def _source(out_t: str) -> str:
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
        uint gx = thread_position_in_grid.x;  // token * PW + word; only t < T exist (grid x = T * PW)
        uint h = thread_position_in_grid.y;
        uint t = gx / PW;
        uint wd = gx % PW;
        uint cap = key_norms_shape[2];  // full-cap state: its capacity is the per-head stride
        ulong row = (ulong)h * cap + t;
        float kn = static_cast<float>(key_norms[row]);
        float vn = static_cast<float>(val_norms[row]);
        uint kw = key_packed[row * PW + wd];
        uint vw = val_packed[row * PW + wd];
        ulong o = row * (PW * 8) + wd * 8;
        for (int n = 0; n < 8; n++) {{
            k_rot[o + n] = static_cast<OT>(cbk[(kw >> (4 * n)) & 15u] * kn);
            v_rot[o + n] = static_cast<OT>(cbv[(vw >> (4 * n)) & 15u] * vn);
        }}
    """


@lru_cache(maxsize=None)
def _kernel(out_t: str):
    return mx.fast.metal_kernel(
        name=f"m72_scratch_dequant_rotated_{out_t}",
        input_names=[
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
        ],
        output_names=["k_rot", "v_rot"],
        source=_source(out_t),
    )


def _check_cache(cache, T):
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


def dequant_rotated(cache, T, *, dtype=mx.float16) -> tuple[mx.array, mx.array]:
    """(K_rot, V_rot), each (1, n_kv, cap, D) ``dtype``; only tokens [0, T) written:
    X[0, h, t, d] = dtype(float(codebook[idx(h, t, d)]) * float(norm(h, t))), fp32 product, one rounding.
    """
    keys, values, D = _check_cache(cache, int(T))
    assert (
        dtype in _DTYPE_NAMES
    ), f"dtype {dtype} unsupported: float16 (default) or bfloat16 only"
    out_t = _DTYPE_NAMES[dtype]
    n_kv, cap = keys.norms.shape[1], keys.norms.shape[2]
    pw = keys.indices.shape[-1]
    kb, vb = cache.key_codec.codebook, cache.value_codec.codebook
    assert kb.dtype == mx.float32 and vb.dtype == mx.float32 and kb.size == 16
    k_rot, v_rot = _kernel(out_t)(
        inputs=[keys.norms, keys.indices, kb, values.norms, values.indices, vb],
        template=[("PW", pw)],
        grid=(int(T) * pw, n_kv, 1),
        threadgroup=(THREADGROUP, 1, 1),
        output_shapes=[(1, n_kv, cap, D), (1, n_kv, cap, D)],
        output_dtypes=[dtype, dtype],
    )
    return k_rot, v_rot


def _perm(x, n_kv, shift):
    if shift % n_kv == 0:
        return x
    idx = mx.array([(h + shift) % n_kv for h in range(n_kv)], dtype=mx.int32)
    return mx.take(x, idx, axis=1)


def scratch_prefill_tq4(
    cache, queries, scale, *, dtype=mx.float16, _kv_head_shift=0
) -> mx.array:
    """queries (1, n_q, L, D) bf16 = the chunk at positions [T - L, T), T = cache.offset -> (1, n_q, L, D)
    ``queries.dtype``. ``_kv_head_shift`` rolls the KV heads seen by SDPA (negative control only; 0 = identity).
    """
    T = int(cache.offset)
    B, n_q, L, D = queries.shape
    assert B == 1, "batch 1 only"
    assert 1 <= L <= T, f"L = {L} outside [1, T = {T}]"
    k_rot, v_rot = dequant_rotated(cache, T, dtype=dtype)
    assert k_rot.shape[-1] == D
    n_kv = k_rot.shape[1]
    k = _perm(k_rot, n_kv, _kv_head_shift)[:, :, :T, :]
    v = _perm(v_rot, n_kv, _kv_head_shift)[:, :, :T, :]
    qr = cache.key_codec.prepare_queries(queries).astype(dtype)
    o = mx.fast.scaled_dot_product_attention(
        qr, k, v, scale=scale, mask="causal", force_fused=True
    )
    return cache.value_codec._rotate_inverse(o.astype(mx.float32)).astype(queries.dtype)


def reference_dequant_rotated(state, T, codec, dtype) -> mx.array:
    """The generic-op form (the probe's P1 ``deq_rot()``) on [0, T): (1, n_kv, T, D) ``dtype``."""
    st = tq._slice_state(state, int(T))
    idx = tq._unpack_lowbit(st.indices, int(codec.bits), int(codec.dim)).astype(
        mx.int32
    )
    return (mx.take(codec.codebook, idx, axis=0) * st.norms[..., None]).astype(dtype)


def reference_prefill(cache, queries, scale, T, dtype) -> mx.array:
    """The probe's P1 ``rot_fused()`` with explicit T and scratch dtype."""
    k = reference_dequant_rotated(cache.keys, T, cache.key_codec, dtype)
    v = reference_dequant_rotated(cache.values, T, cache.value_codec, dtype)
    qr = cache.key_codec.prepare_queries(queries).astype(dtype)
    o = mx.fast.scaled_dot_product_attention(
        qr, k, v, scale=scale, mask="causal", force_fused=True
    )
    return cache.value_codec._rotate_inverse(o.astype(mx.float32)).astype(queries.dtype)
