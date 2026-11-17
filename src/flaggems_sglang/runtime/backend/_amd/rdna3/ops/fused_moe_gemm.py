# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""fused_moe_gemm for RDNA3 (gfx1100).

The generic kernel computes one (token, slot) pair per program with fp32
FMA, so the weights of an expert are read again for every pair routed to it.
For bf16 and fp16 the pairs are grouped by expert here and each group is
computed with tl.dot; other inputs use the generic kernel. Each K block is
its own dot and the blocks are summed in fp32: chaining them through the
WMMA accumulator rounds up to 6% of fp16 outputs away from the fp32 result.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def _fused_moe_gemm_single_token_kernel(
    A_ptr,
    B_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    Out_ptr,
    T,
    E,
    N,
    K,
    top_k,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_out_t,
    stride_out_j,
    stride_out_n,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_slot = tl.program_id(1)
    pid_n = tl.program_id(2)

    if pid_tok >= T or pid_slot >= top_k:
        return

    expert_id = tl.load(topk_ids_ptr + pid_tok * top_k + pid_slot).to(tl.int64)
    routing_weight = tl.load(topk_weights_ptr + pid_tok * top_k + pid_slot).to(
        tl.float32
    )

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        a_ptrs = (
            A_ptr
            + pid_tok.to(tl.int64) * stride_am
            + offs_k.to(tl.int64) * stride_ak
        )
        a = tl.load(a_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        b_ptrs = (
            B_ptr
            + expert_id * stride_be
            + offs_n[None, :].to(tl.int64) * stride_bn
            + offs_k[:, None].to(tl.int64) * stride_bk
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(a[:, None] * b, axis=0)

    scaled = acc * routing_weight

    out_ptrs = (
        Out_ptr
        + pid_tok.to(tl.int64) * stride_out_t
        + pid_slot.to(tl.int64) * stride_out_j
        + offs_n.to(tl.int64) * stride_out_n
    )
    tl.store(out_ptrs, scaled, mask=n_mask)


@triton.jit
def _moe_count_kernel(
    topk_ids_ptr,
    hist_ptr,
    P,
    E,
    BLOCK_E: tl.constexpr,
    CHUNK: tl.constexpr,
):
    c = tl.program_id(0)
    offs = c * CHUNK + tl.arange(0, CHUNK)
    ids = tl.load(topk_ids_ptr + offs, mask=offs < P, other=-1)
    valid = (ids >= 0) & (ids < E)
    ids = tl.where(valid, ids, 0)
    hist = tl.histogram(ids, BLOCK_E, mask=valid)
    tl.store(hist_ptr + c * BLOCK_E + tl.arange(0, BLOCK_E), hist)


@triton.jit
def _moe_align_kernel(
    topk_ids_ptr,
    hist_ptr,
    sorted_ids_ptr,
    block_expert_ptr,
    P,
    E,
    num_chunks,
    num_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    CHUNK: tl.constexpr,
    SUB: tl.constexpr,
    HIST_ROWS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    # sorted_ids holds the pairs of each expert padded with P to a multiple
    # of BLOCK_M, block_expert the expert of each BLOCK_M block (>= E past
    # the last expert). Pairs with an expert id outside [0, E) are left out.
    # Program c < num_chunks places the pairs of chunk c, the others write
    # block_expert and the padding for BLOCK_B blocks each; every entry is
    # written once.
    c = tl.program_id(0)
    e_offs = tl.arange(0, BLOCK_E)
    counts = tl.zeros([BLOCK_E], dtype=tl.int32)
    before = tl.zeros([BLOCK_E], dtype=tl.int32)
    for r0 in range(0, num_chunks, HIST_ROWS):
        r = r0 + tl.arange(0, HIST_ROWS)
        hist = tl.load(
            hist_ptr + r[:, None] * BLOCK_E + e_offs[None, :],
            mask=(r < num_chunks)[:, None],
            other=0,
        )
        counts += tl.sum(hist, axis=0)
        before += tl.sum(tl.where((r < c)[:, None], hist, 0), axis=0)
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    pad_end = tl.cumsum(padded, axis=0)
    pad_start = pad_end - padded

    if c >= num_chunks:
        b = (c - num_chunks) * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = b < num_blocks
        done = pad_end[None, :] <= b[:, None] * BLOCK_M
        expert = tl.sum(done.to(tl.int32), axis=1)
        tl.store(block_expert_ptr + b, expert, mask=b_mask)
        e_c = tl.minimum(expert, BLOCK_E - 1)
        fill_start = tl.gather(pad_start + counts, e_c, 0)
        slot = b[:, None] * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
        used = b_mask & (expert < E)
        is_pad = used[:, None] & (slot >= fill_start[:, None])
        pad_id = tl.full([BLOCK_B, BLOCK_M], 0, tl.int32) + P
        tl.store(sorted_ids_ptr + slot, pad_id, mask=is_pad)
    else:
        run = pad_start + before
        local = tl.arange(0, SUB)
        for s in range(0, tl.minimum(P - c * CHUNK, CHUNK), SUB):
            offs = c * CHUNK + s + local
            ids = tl.load(topk_ids_ptr + offs, mask=offs < P, other=-1)
            valid = (ids >= 0) & (ids < E)
            ids = tl.where(valid, ids, 0)
            same = (ids[:, None] == ids[None, :]) & valid[None, :]
            earlier = same & (local[None, :] < local[:, None])
            slot = tl.gather(run, ids, 0) + tl.sum(earlier.to(tl.int32), 1)
            tl.store(sorted_ids_ptr + slot, offs, mask=valid)
            run += tl.histogram(ids, BLOCK_E, mask=valid)


@triton.jit
def _fused_moe_grouped_gemm_kernel(
    A_ptr,
    B_ptr,
    topk_weights_ptr,
    sorted_ids_ptr,
    block_expert_ptr,
    Out_ptr,
    P,
    E,
    N,
    K,
    top_k,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_out_pair,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_STORE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.load(block_expert_ptr + pid_m)
    if expert >= E:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_ids_ptr + offs_m)
    valid = pair < P
    token = pair // top_k
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        A_ptr
        + token[:, None].to(tl.int64) * stride_am
        + offs_k[None, :].to(tl.int64) * stride_ak
    )
    b_ptrs = (
        B_ptr
        + expert.to(tl.int64) * stride_be
        + offs_n[None, :].to(tl.int64) * stride_bn
        + offs_k[:, None].to(tl.int64) * stride_bk
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    ones = tl.full((BLOCK_M, BLOCK_N), 1.0, tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = k_start + offs_k < K
        a = tl.load(a_ptrs, mask=valid[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        part = tl.dot(a, b, out_dtype=tl.float32)
        # tl.fma instead of + keeps Triton from folding the sum into the
        # accumulator of the dot.
        acc = tl.fma(part, ones, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    routing_weight = tl.load(topk_weights_ptr + pair, mask=valid, other=0.0)
    scaled = acc * routing_weight.to(tl.float32)[:, None]
    if SPLIT_STORE:
        # Without this round trip the store keeps the WMMA layout and writes
        # two bytes a lane.
        low, high = tl.split(tl.reshape(scaled, (BLOCK_M, BLOCK_N // 2, 2)))
        scaled = tl.reshape(tl.join(low, high), (BLOCK_M, BLOCK_N))
    out_ptrs = (
        Out_ptr
        + pair[:, None].to(tl.int64) * stride_out_pair
        + offs_n[None, :]
    )
    tl.store(
        out_ptrs,
        scaled.to(Out_ptr.dtype.element_ty),
        mask=valid[:, None] & n_mask[None, :],
    )


def _grouped_config(P, E, N, K, over_2gb):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)."""
    rows_per_expert = P / E
    big = rows_per_expert > 64 and N * K > 1 << 20
    if over_2gb:
        # Tensors over 2 GB get 64-bit addressing instead of buffer loads,
        # and the tiles with BLOCK_K or BLOCK_N = 256 then spill registers.
        if big:
            return 32, 128, 64, 4, 2
        if rows_per_expert <= 16:
            return 16, 32, 128, 2, 3
        return 16, 64, 128, 4, 2
    if N * K < 1 << 20:
        # Small experts, e.g. up 256 x 2048 and down 2048 x 128.
        if K <= 512:
            return 16, 256, 128, 8, 1
        if rows_per_expert <= 2:
            return 32, 32, 256, 8, 1
        if rows_per_expert <= 32:
            return 32, 64, 256, 8, 2
        return 64, 64, 128, 4, 2
    if K <= 512:
        if rows_per_expert <= 16:
            return 16, 64, 128, 4, 3
        return 64, 256, 64, 8, 2
    if rows_per_expert <= 16:
        # Every program reads its rows of A again, so wider N tiles pay off
        # once A is over about 4 MB, or once K is short and there are enough
        # programs (about one row block per expert in use).
        if P * K > 1 << 21 or (K <= 1536 and min(P, E) * N >= 1 << 17):
            return 16, 64, 256, 4, 2
        return 16, 16, 256, 2, 3
    if not big:
        return 16, 64, 256, 4, 2
    if rows_per_expert <= 128 and K > 1024:
        return 32, 128, 128, 4, 2
    return 64, 256, 64, 8, 2


def _split_store(K):
    # Up to this K the stores bound the kernel and the round trip made it up
    # to 2.2x faster; from K = 1536 on its shuffles cost 2-11%.
    return K <= 1024


def _moe_align(topk_ids_c, E, block_m, chunk=256, sub=64, num_warps=4):
    P = topk_ids_c.numel()
    device = topk_ids_c.device
    block_e = triton.next_power_of_2(E)
    num_chunks = triton.cdiv(P, chunk)
    num_blocks = triton.cdiv(P + E * (block_m - 1), block_m)
    hist = torch.empty(num_chunks * block_e, dtype=torch.int32, device=device)
    sorted_ids = torch.empty(
        num_blocks * block_m, dtype=torch.int32, device=device
    )
    block_expert = torch.empty(num_blocks, dtype=torch.int32, device=device)
    _moe_count_kernel[(num_chunks,)](
        topk_ids_c, hist, P, E, BLOCK_E=block_e, CHUNK=chunk
    )
    block_b = max(1, min(4096 // block_e, 4096 // block_m))
    grid = (num_chunks + triton.cdiv(num_blocks, block_b),)
    _moe_align_kernel[grid](
        topk_ids_c,
        hist,
        sorted_ids,
        block_expert,
        P,
        E,
        num_chunks,
        num_blocks,
        BLOCK_M=block_m,
        BLOCK_E=block_e,
        CHUNK=chunk,
        SUB=sub,
        HIST_ROWS=max(1, 4096 // block_e),
        BLOCK_B=block_b,
        num_warps=num_warps,
    )
    return sorted_ids, block_expert, num_blocks


def _fused_moe_gemm_grouped(A, B, topk_weights_c, topk_ids_c, top_k, Out):
    T, K = A.shape
    E, N, _ = B.shape
    P = T * top_k
    over_2gb = any(
        t.untyped_storage().nbytes() > 2**31 - 1 for t in (A, B, Out)
    )
    block_m, block_n, block_k, num_warps, num_stages = _grouped_config(
        P, E, N, K, over_2gb
    )
    sorted_ids, block_expert, num_blocks = _moe_align(topk_ids_c, E, block_m)
    grid = (num_blocks, triton.cdiv(N, block_n))
    _fused_moe_grouped_gemm_kernel[grid](
        A,
        B,
        topk_weights_c,
        sorted_ids,
        block_expert,
        Out,
        P,
        E,
        N,
        K,
        top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        Out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_STORE=_split_store(K),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return Out


def fused_moe_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    assert A.ndim == 2, "A must be (T, K)"
    assert B.ndim == 3, "B must be (E, N, K)"
    assert topk_weights.ndim == 2, "topk_weights must be (T, top_k)"
    assert topk_ids.ndim == 2, "topk_ids must be (T, top_k)"

    T, K = A.shape
    E, N, K_b = B.shape
    assert K == K_b, f"Hidden dimension mismatch: A has K={K}, B has K={K_b}"
    assert topk_weights.shape == (T, top_k), "topk_weights shape mismatch"
    assert topk_ids.shape == (T, top_k), "topk_ids shape mismatch"

    # A/B 原 dtype 直传：kernel 内 load 后即升 fp32 累加，且 A/B 走显式 strides，
    # 无需 contiguous；topk_* 在 kernel 里是扁平索引，须连续（张量很小，代价可忽略）。
    topk_weights_c = topk_weights.contiguous()
    topk_ids_c = topk_ids.to(torch.int32).contiguous()

    # grid 覆盖每个 (token, slot, n-block) 且各写各的位置：empty 安全；
    # tl.store 落到 A.dtype 指针时自动降精度，与参考实现的舍入点一致。
    Out = torch.empty((T, top_k, N), dtype=A.dtype, device=A.device)

    # Grouping adds two small kernels (about 16 us), which pays off once the
    # pairs read more than about 8 MB of weights, or 16 MB for small experts.
    # For small experts with a short K the per-pair kernel keeps up until
    # about 8 pairs an expert.
    P = T * top_k
    small = N * K < 1 << 20
    if (
        P >= 2
        and (not small or K >= 1024 or P >= 8 * E)
        and P * N * K > (1 << 23 if small else 1 << 22)
        and A.dtype in (torch.bfloat16, torch.float16)
        and B.dtype == A.dtype
    ):
        return _fused_moe_gemm_grouped(
            A, B, topk_weights_c, topk_ids_c, top_k, Out
        )

    grid = lambda meta: (T, top_k, triton.cdiv(N, meta["BLOCK_N"]))

    _fused_moe_gemm_single_token_kernel[grid](
        A,
        B,
        topk_weights_c,
        topk_ids_c,
        Out,
        T,
        E,
        N,
        K,
        top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
    )

    return Out


__all__ = ["fused_moe_gemm"]
