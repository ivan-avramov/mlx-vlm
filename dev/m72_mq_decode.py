"""M72 step 1.2 D1 + D2 prototype (dev only, not wired into production): a stride-aware multi-query TurboQuant MSE decode.

One pass-1 launch serves a k-token MTP verify block (query j attends to the first ``lengths[j]`` tokens) and reads the
cache's full-cap preallocated buffer directly (head stride = the buffer's capacity), so the sub-cap layout copy that
``ensure_row_contiguous`` makes of a time-sliced state never happens.

Exactness contract: per query, bit-identical to the shipped singleton path (``decode_attention`` on the state sliced to
``lengths[j]`` through ``_fused_mse_decode_2pass_1_kernel`` + ``_fused_mse_decode_2pass_2_kernel``). The singleton splits
its T tokens into ``Blocks`` contiguous ranges of ``tpb = ceil(T / Blocks)`` tokens, so this kernel keeps each query's own
partition: queries are grouped into classes of equal (Blocks, tpb) (in a verify block the lengths differ by < k, so almost
always one class), inside a class the ranges share their start and differ only where ``t >= lengths[j]``, which a uniform
per-token guard skips. Per (query, head, block) the accumulation is the singleton's expression sequence verbatim; pass 2 is
the production kernel unchanged.
"""

from functools import lru_cache

import mlx.core as mx

from mlx_vlm import turboquant as tq


def production_num_blocks(total_tokens: int, n_repeats: int) -> int:
    """The shipped dispatch's block count (``decode_attention``, non-legacy 2-pass branch)."""
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
def mq_pass1_kernel(key_bits: int, val_bits: int, dim: int, heads_per_group: int, n_queries: int):
    G, NQ = heads_per_group, n_queries
    elems = dim // 32
    k_off = "k_bit_off" if (elems * key_bits) % 8 else ""
    v_off = "v_bit_off" if (elems * val_bits) % 8 else ""
    key_exprs = tq._gen_unrolled_extract(key_bits, elems, "key_codebook", k_off)
    val_exprs = [e.replace("kb[", "vb[") for e in tq._gen_unrolled_extract(val_bits, elems, "val_codebook", v_off)]
    L = [
        "        constexpr int BD = 32;",
        "        constexpr int qk_per_thread = Dim / BD;",
        "        constexpr int v_per_thread = Dim / BD;",
        f"        constexpr int G = {G};",
        "        typedef float U;",
        "        uint gid = threadgroup_position_in_grid.x;",
        "        auto simd_lid = thread_index_in_simdgroup;",
        "        // D2: the full-cap buffer; its capacity is the per-head stride. Never read past lengths[].",
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
        L.append(f"            if (t < T_{j}) {{" if j < NQ - 1 else "            {  // t < t1 <= T_last")
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
        name=f"m72_tq_mq_decode_pass1_k{key_bits}_v{val_bits}_d{dim}_g{G}_q{NQ}",
        input_names=["queries", "lengths", "key_norms", "key_packed", "key_codebook", "val_norms", "val_packed",
                     "val_codebook"],
        output_names=["out_acc", "out_sums", "out_maxs"],
        source="\n".join(L),
    )


def _classes(lengths, n_repeats):
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


def mq_decode_attention(cache, queries, scale, lengths, heads_per_group=None):
    """queries (B, n_q, k, D); lengths: k ascending token counts (verify: prefix + j + 1). Returns (B, n_q, k, D)."""
    keys, values = cache.keys, cache.values  # full-cap preallocated states (D2)
    B, n_q, k, D = queries.shape
    assert len(lengths) == k and list(lengths) == sorted(lengths)
    assert lengths[0] > 2048, "T <= 2048 uses the single-pass kernel; not covered by the prototype"
    assert lengths[-1] <= cache.offset
    n_kv = keys.norms.shape[1]
    R = n_q // n_kv
    G = heads_per_group or (2 if R % 2 == 0 else 1)
    kb, vb = int(cache.key_codec.bits), int(cache.value_codec.bits)
    # Same prep as the singleton (elementwise scale, row-wise Hadamard), batched over the k queries.
    grouped = (queries * scale).reshape(B, n_kv, R, k, D)
    q_rot = cache.key_codec.prepare_queries(grouped)
    pass2 = tq._fused_mse_decode_2pass_2_kernel()
    outs = []
    for (nb, _tpb), idx in _classes(lengths, R):
        nq = len(idx)
        q_cls = mx.transpose(q_rot[:, :, :, idx[0]:idx[-1] + 1, :], (3, 0, 1, 2, 4)).reshape(nq * B * n_q, D)
        lens = mx.array([lengths[j] for j in idx], dtype=mx.int32)
        BQH = B * n_q
        acc, sums, maxs = mq_pass1_kernel(kb, vb, D, G, nq)(
            inputs=[q_cls, lens, keys.norms, keys.indices, cache.key_codec.codebook, values.norms, values.indices,
                    cache.value_codec.codebook],
            template=[("Dim", D), ("RepeatCount", R), ("Blocks", nb), ("KPackedWidth", keys.indices.shape[-1]),
                      ("VPackedWidth", values.indices.shape[-1])],
            grid=(B * n_kv * (R // G) * nb * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(nq * BQH * nb, D), (nq * BQH * nb,), (nq * BQH * nb,)],
            output_dtypes=[mx.float32] * 3,
        )
        out = pass2(inputs=[acc, sums, maxs], template=[("Dim", D), ("Blocks", nb)], grid=(nq * BQH * 1024, 1, 1),
                    threadgroup=(1024, 1, 1), output_shapes=[(nq * BQH, D)], output_dtypes=[mx.float32])[0]
        out = cache.value_codec._rotate_inverse(out.reshape(nq, B, n_kv, R, D))
        outs.append(mx.transpose(out.reshape(nq, B, n_q, D), (1, 2, 0, 3)))
    out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
    return out.astype(queries.dtype)
