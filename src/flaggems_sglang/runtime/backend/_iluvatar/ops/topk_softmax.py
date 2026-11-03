# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused MoE top-k gating (sgl_kernel.topk_softmax semantics) in Triton.

Per token row: fp32 softmax over the expert axis, then the top-k experts are
written into the preallocated topk_weights (float32) and topk_ids (int32)
buffers, in descending-weight order.

Ordering fidelity
-----------------
The reference is torch.softmax(...) followed by torch.topk(..., dim=-1).  On
this target torch.topk dispatches (measured with the profiler) to

    at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>
    at::native::bitonicSortKVInPlace<2, -1, 16, 16, float, long,
                                     GTOp<float, true>, unsigned int>

so Power2SortSize = block_dim_x * items_per_thread = 16 * 2 = 32: the selected
elements are sorted inside a 32-slot bitonic network whose remaining slots are
invalid (key 0.0, valid=false).  Because that network is not stable, the
relative order of exactly-equal probabilities -- common whenever gating_output
is fp16/bf16 -- is a fixed permutation, and the integer ids only match if the
same arrangement goes into the same network.

Measured arrangement of gatherTopK for a row with kth = K-th largest value:

    [experts with p > kth, ascending expert index]
 ++ [experts with p == kth, ascending expert index]

This kernel reproduces that pipeline exactly:
  1. softmax in fp32 (max-subtracted, exactly torch's expression);
  2. rank each expert with the key  (p > kth ? 0 : 1, expert index)  using two
     masked cumulative scans, which yields that arrangement's positions;
  3. scatter the selected entries into a 32-slot tile keyed by that rank;
  4. run the ATen bitonic network (size loop, then the final merge);
  5. store slots 0..K-1, optionally renormalized by their fp32 sum.
"""

import triton
import triton.language as tl


@triton.jit
def _cmp_stage(
    keys,
    vals,
    valid,
    NK: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr,
):
    """One compare-exchange level of at::native::bitonicSort.

    Pairs are (i, i + S) with bit S of i clear.  `H` is size/2 of the level and
    supplies the direction `(threadIdx.x & (size/2)) != 0`, where threadIdx.x is
    the ATen thread owning the pair; H == 0 selects the final merge, whose
    direction is always false.  Invalid slots (key 0.0) sort to the end.
    """
    G: tl.constexpr = NK // (2 * S)
    ka, kb = tl.split(
        tl.permute(tl.reshape(keys, (BM, G, 2, S)), (0, 1, 3, 2))
    )
    va, vb = tl.split(
        tl.permute(tl.reshape(vals, (BM, G, 2, S)), (0, 1, 3, 2))
    )
    wa, wb = tl.split(
        tl.permute(tl.reshape(valid, (BM, G, 2, S)), (0, 1, 3, 2))
    )

    swp = ((ka > kb) & (wa != 0)) | (wb == 0)
    if H == 0:
        ex = ~swp
    else:
        tx = (
            tl.arange(0, G)[None, :, None] * S + tl.arange(0, S)[None, None, :]
        )
        ex = swp == ((tx & H) != 0)

    na = tl.where(ex, kb, ka)
    nb = tl.where(ex, ka, kb)
    nva = tl.where(ex, vb, va)
    nvb = tl.where(ex, va, vb)
    nwa = tl.where(ex, wb, wa)
    nwb = tl.where(ex, wa, wb)

    keys = tl.reshape(tl.permute(tl.join(na, nb), (0, 1, 3, 2)), (BM, NK))
    vals = tl.reshape(tl.permute(tl.join(nva, nvb), (0, 1, 3, 2)), (BM, NK))
    valid = tl.reshape(tl.permute(tl.join(nwa, nwb), (0, 1, 3, 2)), (BM, NK))
    return keys, vals, valid


@triton.jit
def _topk_softmax_kernel(
    gating_ptr,
    weights_ptr,
    ids_ptr,
    M,
    E,
    stride_gm,
    stride_ge,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    K: tl.constexpr,
    LOGN: tl.constexpr,
    NK: tl.constexpr,
    RENORM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_e = tl.arange(0, BLOCK_E)
    m_mask = offs_m < M
    e_mask = offs_e < E

    x = tl.load(
        gating_ptr + offs_m[:, None] * stride_gm + offs_e[None, :] * stride_ge,
        mask=m_mask[:, None] & e_mask[None, :],
        other=float("-inf"),
    ).to(tl.float32)

    mx = tl.max(x, 1)
    e = tl.where(e_mask[None, :], tl.exp(x - mx[:, None]), 0.0)
    p = e / tl.sum(e, 1)[:, None]
    p = tl.where(e_mask[None, :], p, float("-inf"))

    # kth value = the K-th largest probability.
    e_or_k = tl.where(e_mask[None, :], offs_e[None, :], E)
    p_only = tl.where(e_mask[None, :], p, float("-inf"))
    for _t in tl.static_range(K - 1):
        top_i = tl.min(
            tl.where(p_only == tl.max(p_only, 1)[:, None], e_or_k, E), 1
        )
        p_only = tl.where(
            offs_e[None, :] == top_i[:, None], float("-inf"), p_only
        )
    kth_i = tl.min(
        tl.where(p_only == tl.max(p_only, 1)[:, None], e_or_k, E), 1
    )
    kth = tl.max(
        tl.where(offs_e[None, :] == kth_i[:, None], p, float("-inf")), 1
    )

    # Arrangement rank: strictly-greater lanes first, then equal lanes, each in
    # ascending expert-index order.
    gmask = (p > kth[:, None]).to(tl.int32)
    emask = ((p == kth[:, None]) & e_mask[None, :]).to(tl.int32)
    n_gt = tl.sum(gmask, 1)
    rank_g = tl.cumsum(gmask, 1) - 1
    rank_e = tl.cumsum(emask, 1) - 1 + n_gt[:, None]
    rank = tl.where(gmask != 0, rank_g, rank_e)

    offs_k = tl.arange(0, NK)
    assembled = tl.where(emask != 0, rank_e, rank)
    take = ((gmask != 0) | (emask != 0)) & (assembled < K) & (assembled < NK)
    # [BM, BE, NK] lane-to-slot match; the selected expert count never exceeds K
    # per row (one lane per row can match a given slot), so a plain sum is exact.
    hit = take[:, :, None] & (assembled[:, :, None] == offs_k[None, None, :])
    sel_w = tl.sum(tl.where(hit, p[:, :, None], 0.0), 1)
    sel_i = tl.sum(tl.where(hit, offs_e[None, :, None], 0), 1)

    kk = tl.arange(0, NK)
    valid = (kk < K).to(tl.int32)[None, :] + tl.zeros(
        (BLOCK_M, NK), dtype=tl.int32
    )

    for lvl in tl.static_range(1, LOGN):
        for s in tl.static_range(lvl - 1, -1, -1):
            sel_w, sel_i, valid = _cmp_stage(
                sel_w, sel_i, valid, NK, 1 << s, 1 << (lvl - 1), BLOCK_M
            )
    for s in tl.static_range(LOGN - 1, -1, -1):
        sel_w, sel_i, valid = _cmp_stage(
            sel_w, sel_i, valid, NK, 1 << s, 0, BLOCK_M
        )

    if RENORM:
        wsum = tl.sum(tl.where(valid != 0, sel_w, 0.0), 1)
        sel_w = sel_w / wsum[:, None]

    st_mask = m_mask[:, None] & (kk[None, :] < K)
    tl.store(
        weights_ptr + offs_m[:, None] * stride_wm + kk[None, :] * stride_wk,
        sel_w,
        mask=st_mask,
    )
    tl.store(
        ids_ptr + offs_m[:, None] * stride_im + kk[None, :] * stride_ik,
        sel_i,
        mask=st_mask,
    )


def topk_softmax(topk_weights, topk_ids, gating_output, renormalize=False):
    M = int(gating_output.shape[0])
    E = int(gating_output.shape[1])
    K = int(topk_ids.shape[-1])
    if M == 0 or K == 0:
        return topk_weights, topk_ids
    NK = max(32, triton.next_power_of_2(K))
    LOGN = NK.bit_length() - 1
    BLOCK_E = triton.next_power_of_2(max(E, 1))
    BLOCK_M = max(1, min(16, 2048 // BLOCK_E))
    if M < BLOCK_M:
        BLOCK_M = max(1, min(BLOCK_M, triton.next_power_of_2(M)))
    num_warps = 4 if BLOCK_M * BLOCK_E >= 1024 else 2
    grid = (triton.cdiv(M, BLOCK_M),)
    _topk_softmax_kernel[grid](
        gating_output,
        topk_weights,
        topk_ids,
        M,
        E,
        gating_output.stride(0),
        gating_output.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        K=K,
        LOGN=LOGN,
        NK=NK,
        RENORM=bool(renormalize),
        BLOCK_M=BLOCK_M,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps,
    )
    return topk_weights, topk_ids


__all__ = ["topk_softmax"]
