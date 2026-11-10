"""M72 step 1.2 matrix-unit verify kernel prototype (dev only, never imported by ``mlx_vlm/``; C312 / C313).

One launch computes a whole k-token MTP verify block (query j attends to tokens ``[0, lengths[j])``) with simdgroup 8x8
matrix multiply-accumulate over KV tiles loaded once into threadgroup memory, reading the cache's full-cap preallocated
buffers in place (per-head stride = the buffer's capacity). Two tile loaders share one kernel body:

- TurboQuant 4 (MSE codec): unpacks each token's packed indices (4 bits per dimension) to the codebook's unit values in the codec's rotated space; the key
  norm scales S, the value norm is folded into P before P.V; the output is rotated back with the value codec.
- native16: copies the bf16 K / V rows; the softmax scale is applied to S in fp32.

Layout: one threadgroup per (KV head, token block); the rows of a KV head are its R query heads x k queries (row
``r = i * k + j``), padded to a multiple of 8, one simdgroup per 8-row tile. Per tile of ``TILE_TOKENS`` tokens: S = Q.K^T
and O += P.V with fp32 accumulators; online softmax per row in fp32. Each block writes per-row partials in the layout of
``turboquant._fused_mse_decode_2pass_2_kernel`` (empty (row, block) pairs: max = -1e30, sum = 0, acc = 0), which combines
them unchanged. Accuracy standard: ``joint_v1``'s tolerance (C312), not bit-identity to the singleton.

Fragment element mapping (verified on this GPU by ``fragment_mapping_ok``): lane l of an 8x8 fragment holds row
``fm = (q & 4) + (l / 2) % 4`` and columns ``fn, fn + 1`` with ``fn = (q & 2) * 2 + (l % 2) * 2``, ``q = l / 4``; the
lanes sharing a row differ in lane bits 0 and 3 (row reductions: ``simd_shuffle_xor`` 1 and 8).
"""

from functools import lru_cache

import mlx.core as mx

from mlx_vlm import turboquant as tq

TILE_TOKENS = 16  # Tt: K + V tiles in threadgroup memory (<= 32 KB with the fp32 V tile of the native16 arm)
TARGET_BLOCKS = (
    128  # token blocks per KV head (4 KV heads -> 512 threadgroups at long T)
)
MIN_BLOCK_TOKENS = (
    128  # keeps pass-2 partial traffic small next to the KV bytes at short T
)
SMEM_PAD = 8  # elements of padding per threadgroup row (bank spread for the transposed K loads)

# (QK operand type, P.V operand type) per arm. TQ4: the fp32 codebook centroids are rounded to half (about 2^-11
# relative), as are the rotated queries and P; not exact. Accuracy is shown by A1 against the fp32 reference, not assumed.
# native16 keeps bf16 K as bfloat (exact copy) and runs P.V in fp32 (P unrounded, V exact).
FRAGMENT_TYPES = {"tq4": ("half", "half"), "native16": ("bfloat", "float")}


def plan_blocks(max_length: int, block_tokens=None) -> tuple[int, int]:
    """(Blocks, tokens per block): every block holds >= 1 whole tile and starts below ``max_length``."""
    assert max_length >= 1
    if block_tokens is None:
        block_tokens = max(MIN_BLOCK_TOKENS, -(-max_length // TARGET_BLOCKS))
    tpb = max(TILE_TOKENS, -(-int(block_tokens) // TILE_TOKENS) * TILE_TOKENS)
    return -(-max_length // tpb), tpb


def _loader_tq4():
    return [
        "        // TQ4 loader: 4-bit indices LSB-first (dim d at bits [4d, 4d + 4) of the row's words) -> unit values.",
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


def _loader_native16():
    return [
        "        // native16 loader: bf16 rows copied (V widened to the P.V operand type).",
        "        for (uint w = tid; w < Tt * (D / 8); w += NT) {",
        "            uint tt = w / (D / 8);",
        "            uint c8 = w % (D / 8);",
        "            uint t = ts + tt;",
        "            bool in = t < Lmax;",
        "            ulong g = ((ulong)kvh * cap + t) * D + c8 * 8;",
        "            threadgroup QT* kd = Ks + tt * SK + c8 * 8;",
        "            threadgroup PT* vd = Vs + tt * SK + c8 * 8;",
        "            for (int n = 0; n < 8; n++) {",
        "                kd[n] = in ? static_cast<QT>(static_cast<float>(keys[g + n])) : QT(0);",
        "                vd[n] = in ? static_cast<PT>(static_cast<float>(values[g + n])) : PT(0);",
        "            }",
        "        }",
        "        if (tid < Tt) {",
        "            kfac[tid] = scale[0];",
        "            vfac[tid] = 1.0f;",
        "        }",
    ]


@lru_cache(maxsize=None)
def mma_kernel(arm: str, rows: int, n_queries: int):
    """Pass-1 kernel for one arm and (rows per KV head, k). Template: Dim, Tt, TPB, Blocks, Shift (+ PW for TQ4)."""
    qt, pt = FRAGMENT_TYPES[arm]
    ns = (rows + 7) // 8
    if arm == "tq4":
        cap_expr = "key_norms_shape[2]"
        nkv_expr = "key_norms_shape[1]"
        loader = _loader_tq4()
        inputs = [
            "queries",
            "lengths",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
        ]
    else:
        cap_expr = "keys_shape[2]"
        nkv_expr = "keys_shape[1]"
        loader = _loader_native16()
        inputs = ["queries", "lengths", "scale", "keys", "values"]
    L = [
        "        constexpr int D = Dim;",
        f"        constexpr int M = {rows};",
        f"        constexpr int K = {n_queries};",
        f"        constexpr uint NT = {ns * 32};",
        f"        constexpr int SK = D + {SMEM_PAD};",
        "        constexpr int NC = Tt / 8;",
        "        constexpr int ND = D / 8;",
        "        constexpr float NEG = -1e30f;",
        f"        typedef {qt} QT;",
        f"        typedef {pt} PT;",
        "        threadgroup QT Ks[Tt * SK];",
        "        threadgroup PT Vs[Tt * SK];",
        "        threadgroup float kfac[Tt];",
        "        threadgroup float vfac[Tt];",
        "        uint blk = threadgroup_position_in_grid.x;",
        "        uint h = threadgroup_position_in_grid.y;",
        "        uint sg = simdgroup_index_in_threadgroup;",
        "        uint lane = thread_index_in_simdgroup;",
        "        uint tid = thread_position_in_threadgroup.x;",
        f"        uint cap = {cap_expr};  // full-cap buffer: its capacity is the per-head stride",
        f"        uint kvh = (h + Shift) % {nkv_expr};",
        "        int qd = lane / 4;",
        "        int fm = (qd & 4) + ((lane / 2) % 4);",
        "        int fn = (qd & 2) * 2 + (lane % 2) * 2;",
        "        int r = sg * 8 + fm;",
        "        bool row_ok = r < M;",
        "        uint len_r = row_ok ? (uint)lengths[r % K] : 0u;",
        "        uint Lmax = (uint)lengths[K - 1];",
        "        uint t0 = blk * TPB;",
        "        uint t1 = min(t0 + TPB, Lmax);",
    ]
    if arm == "tq4":
        L += [
            "        threadgroup QT cbk[16];",
            "        threadgroup PT cbv[16];",
            "        if (tid < 16) {",
            "            cbk[tid] = static_cast<QT>(key_codebook[tid]);",
            "            cbv[tid] = static_cast<PT>(val_codebook[tid]);",
            "        }",
        ]
    L += [
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
    L += ["    " + s for s in loader]
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
        name=f"m72_mma_verify_{arm}_m{rows}_k{n_queries}",
        input_names=inputs,
        output_names=["out_acc", "out_sums", "out_maxs"],
        source="\n".join(L),
    )


def _check(queries, lengths, available):
    B, n_q, k, D = queries.shape
    assert B == 1, "batch 1 only"
    assert 1 <= k <= 5 and len(lengths) == k
    assert list(lengths) == sorted(lengths), "lengths must be ascending"
    assert lengths[0] >= 1, "lengths[0] >= 1"
    assert lengths[-1] <= available, f"lengths[-1] = {lengths[-1]} > {available}"
    assert D % 8 == 0
    return B, n_q, k, D


def _run(arm, q_rows, kv_inputs, lengths, n_kv, rows, k, D, block_tokens, shift):
    nb, tpb = plan_blocks(int(lengths[-1]), block_tokens)
    lens = mx.array([int(x) for x in lengths], dtype=mx.int32)
    ns = (rows + 7) // 8
    template = [
        ("Dim", D),
        ("Tt", TILE_TOKENS),
        ("TPB", tpb),
        ("Blocks", nb),
        ("Shift", int(shift) % n_kv),
    ]
    if arm == "tq4":
        template.append(("PW", int(kv_inputs[1].shape[-1])))
    acc, sums, maxs = mma_kernel(arm, rows, k)(
        inputs=[q_rows, lens, *kv_inputs],
        template=template,
        grid=(ns * 32 * nb, n_kv, 1),
        threadgroup=(ns * 32, 1, 1),
        output_shapes=[(n_kv * rows * nb, D), (n_kv * rows * nb,), (n_kv * rows * nb,)],
        output_dtypes=[mx.float32] * 3,
    )
    return tq._fused_mse_decode_2pass_2_kernel()(
        inputs=[acc, sums, maxs],
        template=[("Dim", D), ("Blocks", nb)],
        grid=(n_kv * rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(n_kv * rows, D)],
        output_dtypes=[mx.float32],
    )[0]


def mma_verify_tq4(
    cache, queries, scale, lengths, *, block_tokens=None, _kv_head_shift=0
) -> mx.array:
    """TurboQuant MSE 4-bit cache: queries (1, n_q, k, D) -> (1, n_q, k, D) in ``queries.dtype``, unrotated space."""
    # The full-cap preallocated TurboQuantMSEState buffers.
    keys, values = cache.keys, cache.values
    B, n_q, k, D = _check(queries, lengths, cache.offset)
    assert (
        int(cache.key_codec.bits) == 4 and int(cache.value_codec.bits) == 4
    ), "4-bit MSE codec only"
    assert keys.indices.shape[-1] * 32 == 4 * D
    n_kv = keys.norms.shape[1]
    R = n_q // n_kv
    rows = R * k
    # Same prep as the singleton (elementwise scale, row-wise rotation), grouped (B, n_kv, R, k, D): row = i * k + j.
    grouped = (queries * scale).reshape(B, n_kv, R, k, D)
    q_rows = cache.key_codec.prepare_queries(grouped).reshape(n_kv * rows, D)
    kv = [
        keys.norms,
        keys.indices,
        cache.key_codec.codebook,
        values.norms,
        values.indices,
        cache.value_codec.codebook,
    ]
    out = _run(
        "tq4", q_rows, kv, lengths, n_kv, rows, k, D, block_tokens, _kv_head_shift
    )
    out = cache.value_codec._rotate_inverse(out.reshape(B, n_kv, R, k, D))
    return out.reshape(B, n_q, k, D).astype(queries.dtype)


def mma_verify_native16(
    keys, values, queries, scale, lengths, *, block_tokens=None, _kv_head_shift=0
) -> mx.array:
    """bf16 full-cap K / V (1, n_kv, cap, D): queries (1, n_q, k, D) -> (1, n_q, k, D) in ``queries.dtype``."""
    B, n_q, k, D = _check(queries, lengths, keys.shape[2])
    assert keys.shape == values.shape
    assert keys.dtype == mx.bfloat16 and values.dtype == mx.bfloat16
    n_kv = keys.shape[1]
    R = n_q // n_kv
    rows = R * k
    # (1, n_kv, R, k, D) order: row = i * k + j; the scale is applied to S in the kernel.
    q_rows = queries.reshape(n_kv * rows, D)
    sc = mx.array([float(scale)], dtype=mx.float32)
    kv = [sc, keys, values]
    out = _run(
        "native16", q_rows, kv, lengths, n_kv, rows, k, D, block_tokens, _kv_head_shift
    )
    return out.reshape(B, n_q, k, D).astype(queries.dtype)


def fragment_mapping_ok() -> bool:
    """True when the 8x8 fragment's thread elements follow the mapping documented above (checked on this GPU)."""
    src = """
        uint lane = thread_index_in_simdgroup;
        simdgroup_matrix<float, 8, 8> m;
        simdgroup_load(m, inp, 8);
        out[lane * 2] = m.thread_elements()[0];
        out[lane * 2 + 1] = m.thread_elements()[1];
    """
    kern = mx.fast.metal_kernel(
        name="m72_mma_fragment_map",
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
        pair = [int(got[2 * lane]), int(got[2 * lane + 1])]
        if pair != [fm * 8 + fn, fm * 8 + fn + 1]:
            return False
    return True
