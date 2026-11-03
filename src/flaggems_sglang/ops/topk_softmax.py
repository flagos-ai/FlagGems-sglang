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

"""SGLang MoE top-k gating (fused softmax + top-k) in Triton.

In-place ABI (sgl_kernel.topk_softmax):
    softmax(gating_output) in fp32, then the k largest experts per token row are
    written into the preallocated topk_weights (fp32) and topk_ids (int32)
    buffers, ordered by descending weight.  run() returns None.

Ranking, tie tests and the stored weights all use the fp32 normalized softmax
    p = exp(x - max) / sum(exp(x - max))
which is exactly the tensor the reference consumes,
    torch.topk(torch.softmax(gating_output.float(), -1)).

Ordering / tie order
--------------------
torch.topk on CUDA does not order exactly-equal probabilities by expert index.
On this target it runs (profiler-confirmed)

    sbtopk::gatherTopK<float,uint,2>   then
    bitonicSortKVInPlace<2,-1,16,16,float,long,GTOp<float,true>>

gatherTopK fills a k-length buffer with

    [elements with p > kthValue, ascending expert index]
 ++ [elements with p == kthValue, ascending expert index]      (truncated to k)

and ATen's bitonic network (Power2SortSize = 32 = block_dim_x(16) *
items_per_thread(2)) sorts those 32 slots, of which the first k are stored.  The
schedule is transcribed from ATen/native/cuda/SortUtils.cuh shipped with the
target's torch 2.9.0:

    phase 1: for size = 2,4,8,16 (size < 32):
                 flag = ((threadIdx.x & (size/2)) != 0)          # per THREAD
                 for stride = size/2 .. 1:
                     pos = 2*threadIdx.x - (threadIdx.x & (stride-1))
                     bitonicSwap(pos, pos+stride, flag)
    phase 2: for stride = 16 .. 1: bitonicSwap(pos, pos+stride, false)

    bitonicSwap: swap = ((kA > kB) && validA) || !validB;
                 if (swap == dir) exchange (key, value, valid)

The pairing (pos, pos+stride) is exactly (i, i^stride), so the network is 15
rounds of an XOR-gather butterfly over a [BLOCK_M, 32] register tile; the
threadIdx.x of a pair is recovered from its lower slot index by
    t = low - stride * (low // (2*stride)).

The network is entered only when some row in the block has a k-length buffer
that could hold two equal keys -- either two of the selected probabilities are
equal or two experts share the k-th largest probability.  Conveniently, exactly
those rows have their output replaced by the network result, so all other rows
(including every timing workload here, which uses fresh random gating) take the
plain descending path.
"""

import triton
import triton.language as tl

_NSLOT = 32


@triton.jit
def _topk_softmax_kernel(
    gating_ptr,
    w_ptr,
    id_ptr,
    M,
    E,
    K,
    stride_gm,
    stride_ge,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    renorm,
    BK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_e = tl.arange(0, BLOCK_E)
    offs_k = tl.arange(0, BK)
    kcol = offs_k[None, :] < K

    row_mask = offs_m < M
    e_mask = offs_e < E

    g_ptrs = (
        gating_ptr + offs_m[:, None] * stride_gm + offs_e[None, :] * stride_ge
    )
    x = tl.load(
        g_ptrs, mask=row_mask[:, None] & e_mask[None, :], other=float("-inf")
    )
    x = x.to(tl.float32)

    row_max = tl.max(x, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)

    e = tl.exp(x - row_max[:, None])
    e = tl.where(e_mask[None, :], e, 0.0)
    s = tl.sum(e, axis=1)
    s = tl.where(s > 0.0, s, 1.0)
    p = e / s[:, None]

    # ---- plain descending-weight top-k (ties by lowest expert index) -------
    sel = tl.where(e_mask[None, :], p, float("-inf"))
    acc_w = tl.zeros([BLOCK_M, BK], dtype=tl.float32)
    acc_i = tl.zeros([BLOCK_M, BK], dtype=tl.int32)
    for r in tl.static_range(BK):
        if r < K:
            mv, mi = tl.max(
                sel,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            mi_i = mi.to(tl.int32)
            acc_w = tl.where(offs_k[None, :] == r, mv[:, None], acc_w)
            acc_i = tl.where(offs_k[None, :] == r, mi_i[:, None], acc_i)
            sel = tl.where(
                offs_e[None, :] == mi_i[:, None], float("-inf"), sel
            )

    kth = tl.sum(tl.where(offs_k[None, :] == (K - 1), acc_w, 0.0), axis=1)

    # ---- rows whose k-length buffer can hold two equal keys ---------------
    nxt = tl.minimum(offs_k + 1, BK - 1)
    nxt2 = tl.broadcast_to(nxt[None, :], (BLOCK_M, BK))
    shifted = tl.gather(acc_w, nxt2, 1)
    delta = tl.where((offs_k < K - 1)[None, :], acc_w - shifted, float("inf"))
    dup_inside = (
        tl.min(tl.where(row_mask[:, None], delta, float("inf")), axis=1) == 0.0
    )
    cnt_eq = tl.sum(
        tl.where((p == kth[:, None]) & e_mask[None, :], 1.0, 0.0), axis=1
    )
    use_row = (dup_inside | (cnt_eq >= 2.0)) & row_mask
    n_need = tl.sum(tl.where(use_row, 1.0, 0.0), axis=0)

    if n_need > 0.0:
        # ---- gatherTopK: [p > kth asc index] ++ [p == kth asc index] -------
        is_gt = (p > kth[:, None]) & e_mask[None, :]
        is_eq = (p == kth[:, None]) & e_mask[None, :]
        gt_f = tl.where(is_gt, 1.0, 0.0)
        eq_f = tl.where(is_eq, 1.0, 0.0)
        c_gt = tl.cumsum(gt_f, axis=1) - gt_f
        tot_gt = tl.sum(gt_f, axis=1)
        c_eq = tl.cumsum(eq_f, axis=1) - eq_f
        big = tl.full([BLOCK_M, 1], 67108864.0, tl.float32)
        pos = tl.minimum(
            tl.where(is_gt, c_gt, big),
            tl.where(is_eq, tot_gt[:, None] + c_eq, big),
        )

        slots = tl.arange(0, NS)
        eidx = tl.where(e_mask[None, :], offs_e[None, :], 0).to(tl.float32)
        gk = tl.zeros([BLOCK_M, NS], dtype=tl.float32)
        gv = tl.zeros([BLOCK_M, NS], dtype=tl.float32)
        gd = tl.zeros([BLOCK_M, NS], dtype=tl.float32)
        for j in tl.static_range(BK):
            if j < K:
                hit = pos == j
                kj = tl.sum(tl.where(hit, p, 0.0), axis=1)
                ij = tl.sum(tl.where(hit, eidx, 0.0), axis=1)
                col = slots[None, :] == j
                gk = tl.where(col, kj[:, None], gk)
                gv = tl.where(col, ij[:, None], gv)
                gd = tl.where(col, 1.0, gd)

        # ---- ATen bitonicSortKVInPlace<2,-1,16,16> -------------------------
        for si in tl.static_range(4):
            for sj in tl.static_range(4):
                if sj <= si:
                    st = (2 << si) >> (sj + 1)
                    idx = tl.broadcast_to((slots ^ st)[None, :], (BLOCK_M, NS))
                    pk = tl.gather(gk, idx, 1)
                    pv = tl.gather(gv, idx, 1)
                    pd = tl.gather(gd, idx, 1)
                    acol = ((slots & st) == 0)[None, :]
                    kA = tl.where(acol, gk, pk)
                    kB = tl.where(acol, pk, gk)
                    vA = tl.where(acol, gd, pd)
                    vB = tl.where(acol, pd, gd)
                    sw = ((kA > kB) & (vA > 0.5)) | (vB < 0.5)
                    low = slots - (slots & st)
                    tid = low - st * (low // (2 * st))
                    dirv = tl.broadcast_to(
                        ((tid & ((2 << si) // 2)) != 0)[None, :], (BLOCK_M, NS)
                    )
                    cond = sw == dirv
                    gk = tl.where(cond, pk, gk)
                    gv = tl.where(cond, pv, gv)
                    gd = tl.where(cond, pd, gd)
        for sj in tl.static_range(5):
            st = 16 >> sj
            idx = tl.broadcast_to((slots ^ st)[None, :], (BLOCK_M, NS))
            pk = tl.gather(gk, idx, 1)
            pv = tl.gather(gv, idx, 1)
            pd = tl.gather(gd, idx, 1)
            acol = ((slots & st) == 0)[None, :]
            kA = tl.where(acol, gk, pk)
            kB = tl.where(acol, pk, gk)
            vA = tl.where(acol, gd, pd)
            vB = tl.where(acol, pd, gd)
            sw = ((kA > kB) & (vA > 0.5)) | (vB < 0.5)
            cond = sw == False  # noqa: E712 - bitonicSwap with dir == false
            gk = tl.where(cond, pk, gk)
            gv = tl.where(cond, pv, gv)
            gd = tl.where(cond, pd, gd)

        net_w = tl.zeros([BLOCK_M, BK], dtype=tl.float32)
        net_i = tl.zeros([BLOCK_M, BK], dtype=tl.int32)
        for j in tl.static_range(BK):
            if j < K:
                col = offs_k[None, :] == j
                net_w = tl.where(
                    col,
                    tl.sum(tl.where(slots[None, :] == j, gk, 0.0), axis=1)[
                        :, None
                    ],
                    net_w,
                )
                net_i = tl.where(
                    col,
                    tl.sum(tl.where(slots[None, :] == j, gv, 0.0), axis=1)[
                        :, None
                    ].to(tl.int32),
                    net_i,
                )
        use = use_row[:, None]
        acc_w = tl.where(use, net_w, acc_w)
        acc_i = tl.where(use, net_i, acc_i)

    if renorm != 0:
        acc_w = acc_w / tl.sum(acc_w, axis=1)[:, None]

    store_mask = row_mask[:, None] & kcol
    w_ptrs = w_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
    i_ptrs = id_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    tl.store(w_ptrs, acc_w, mask=store_mask)
    tl.store(i_ptrs, acc_i, mask=store_mask)


def topk_softmax(topk_weights, topk_ids, gating_output, renormalize=False):
    k = int(topk_ids.shape[-1])
    M = int(gating_output.shape[0])
    E = int(gating_output.shape[1])
    if M == 0 or E == 0 or k == 0:
        return topk_weights, topk_ids

    block_e = max(16, triton.next_power_of_2(E))
    # Small expert counts give few programs and a very wide per-program tile;
    # the tie-resolution tensors are sized [BLOCK_M, NS] and inflate register
    # pressure, capping occupancy well below the launch-latency-bound regime's
    # needs.  Keep wide tiles only where the expert axis is long.
    if block_e <= 32:
        block_m = 16
    else:
        block_m = min(64, max(1, 1024 // block_e))
    block_k = max(1, triton.next_power_of_2(k))
    grid = (triton.cdiv(M, block_m),)

    _topk_softmax_kernel[grid](
        gating_output,
        topk_weights,
        topk_ids,
        M,
        E,
        k,
        gating_output.stride(0),
        gating_output.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        1 if renormalize else 0,
        BK=block_k,
        BLOCK_M=block_m,
        BLOCK_E=block_e,
        NS=_NSLOT,
        num_warps=4,
    )
    return topk_weights, topk_ids


__all__ = ["topk_softmax"]
