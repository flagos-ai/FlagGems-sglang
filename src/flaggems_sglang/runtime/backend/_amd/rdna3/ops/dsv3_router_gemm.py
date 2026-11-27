# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""dsv3_router_gemm for RDNA3 (gfx1100).

Up to 16 tokens the op mostly streams the router weights, so 16-bit inputs
take the kernel that reads them fastest: the AMD per-(token, expert) kernel
with more warps for small expert counts and for 5 to 8 tokens, and a few
experts per program for 1 to 4 tokens. From 9 tokens on they run on the
matrix units: each program multiplies a block of token rows by a block of
experts over a slice of the hidden dimension, so each router weight is read
once per row block instead of once per token; small grids split the hidden
dimension and a second kernel adds the parts. Products are exact and sums
are fp32. Other dtypes keep the AMD kernels.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dsv3_router_gemm_small_batch_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    STRIDE_X_ROW: tl.constexpr,
    STRIDE_X_COL: tl.constexpr,
    STRIDE_WEIGHT_ROW: tl.constexpr,
    STRIDE_WEIGHT_COL: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0)
    cols = tl.program_id(1)
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_K,), tl.float32)
    for base in range(tl.cdiv(HIDDEN_SIZE, BLOCK_K)):
        k_block = base * BLOCK_K + k_offsets
        x = tl.load(
            x_ptr + rows * STRIDE_X_ROW + k_block * STRIDE_X_COL,
            k_block < HIDDEN_SIZE,
            other=0,
        ).to(tl.float32)
        weights = tl.load(
            weight_ptr
            + cols * STRIDE_WEIGHT_ROW
            + k_block * STRIDE_WEIGHT_COL,
            k_block < HIDDEN_SIZE,
            other=0,
            cache_modifier=".cg",
        ).to(tl.float32)
        accumulator += weights * x
    out = tl.sum(accumulator, 0)
    tl.store(out_ptr + rows * NUM_EXPERTS + cols, out)


@triton.jit
def _dsv3_router_gemm_large_batch_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    STRIDE_X_ROW: tl.constexpr,
    STRIDE_X_COL: tl.constexpr,
    STRIDE_WEIGHT_ROW: tl.constexpr,
    STRIDE_WEIGHT_COL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.program_id(0)
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    for base in range(tl.cdiv(HIDDEN_SIZE, BLOCK_K)):
        k_block = base * BLOCK_K + k_offsets
        x = tl.load(
            x_ptr
            + rows[:, None] * STRIDE_X_ROW
            + k_block[None, :] * STRIDE_X_COL,
            (rows[:, None] < NUM_ROWS) & (k_block[None, :] < HIDDEN_SIZE),
            other=0,
        ).to(tl.float32)
        weights = tl.load(
            weight_ptr
            + cols * STRIDE_WEIGHT_ROW
            + k_block * STRIDE_WEIGHT_COL,
            k_block < HIDDEN_SIZE,
            other=0,
            cache_modifier=".cg",
        ).to(tl.float32)
        accumulator += x * weights[None, :]
    out = tl.sum(accumulator, 1)
    tl.store(out_ptr + rows * NUM_EXPERTS + cols, out, rows < NUM_ROWS)


@triton.jit
def _router_dot_kernel(
    X,
    WT,
    P,
    M,
    SPLIT,
    E: tl.constexpr,
    K: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    BM: tl.constexpr,
    BE: tl.constexpr,
    BK: tl.constexpr,
):
    # Split part j covers hidden columns [j * chunk, (j + 1) * chunk) and
    # writes P[j]; with SPLIT == 1, P is the output.
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BE + tl.arange(0, BE)
    part = tl.program_id(2)
    rk = tl.arange(0, BK)
    chunk = tl.cdiv(tl.cdiv(K, SPLIT), BK) * BK
    acc = tl.zeros((BM, BE), tl.float32)
    ones = tl.full((BM, BE), 1.0, tl.float32)
    for k in range(part * chunk, tl.minimum(K, (part + 1) * chunk), BK):
        kk = k + rk
        x = tl.load(
            X + rm[:, None] * SX0 + kk[None, :] * SX1,
            (rm[:, None] < M) & (kk[None, :] < K),
            0,
        )
        w = tl.load(
            WT + re[None, :] * SW0 + kk[:, None] * SW1,
            (re[None, :] < E) & (kk[:, None] < K),
            0,
        )
        # A dot accumulating into acc truncates the fp32 addition here.
        acc = tl.fma(tl.dot(x, w), ones, acc)
    tl.store(
        P + part * M * E + rm[:, None] * E + re[None, :],
        acc,
        (rm[:, None] < M) & (re[None, :] < E),
    )


@triton.jit
def _router_gemv_kernel(
    X,
    WT,
    OUT,
    M,
    E: tl.constexpr,
    K: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    BM: tl.constexpr,
    BE: tl.constexpr,
    BK: tl.constexpr,
):
    rm = tl.arange(0, BM)
    re = tl.program_id(0) * BE + tl.arange(0, BE)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BE, BK), tl.float32)
    for k in range(0, K, BK):
        kk = k + rk
        x = tl.load(
            X + rm[:, None] * SX0 + kk[None, :] * SX1,
            (rm[:, None] < M) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        w = tl.load(
            WT + re[:, None] * SW0 + kk[None, :] * SW1,
            (re[:, None] < E) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        acc += x[:, None, :] * w[None, :, :]
    tl.store(
        OUT + rm[:, None] * E + re[None, :],
        tl.sum(acc, 2),
        (rm[:, None] < M) & (re[None, :] < E),
    )


@triton.jit
def _router_sum_kernel(P, OUT, SIZE, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for part in tl.static_range(SPLIT):
        acc += tl.load(P + part * SIZE + x, x < SIZE, 0)
    tl.store(OUT + x, acc, x < SIZE)


def _config(rows, experts):
    """Kernel for 16-bit inputs and its parameters, tuned on gfx1100.

    "scalar": num_warps of the AMD per-(token, expert) kernel.
    "gemv": experts per program, BK and num_warps.
    "dot": BM, BE, BK, num_warps, num_stages and hidden splits.
    """
    if rows <= 16:
        if experts < 256:
            return "scalar", 2
        if rows <= 4:
            return "gemv", (2, 1024, 4)
        if rows <= 8:
            return "scalar", 8
        if experts >= 384:
            return "dot", (16, 64, 128, 4, 1, 8)
        return "dot", (16, 16, 128, 1, 1, 8)
    if experts < 128:
        return "scalar", 4
    if rows <= 128:
        return "dot", (16, 32, 256, 2, 2, 4)
    if rows <= 1024:
        return "dot", (128, 32, 128, 4, 2, 4)
    return "dot", (128, 64, 128, 8, 1, 1)


def _scalar_router_gemm(
    hidden_states, router_weights, out_dtype, num_warps=None
):
    num_rows, hidden_size = hidden_states.shape
    num_experts = router_weights.shape[0]
    out = torch.empty(
        (num_rows, num_experts), dtype=out_dtype, device=hidden_states.device
    )
    if num_warps is None and num_rows == 16:
        _dsv3_router_gemm_large_batch_kernel[(num_experts,)](
            hidden_states,
            router_weights,
            out,
            num_rows,
            num_experts,
            hidden_size,
            hidden_states.stride(0),
            hidden_states.stride(1),
            router_weights.stride(0),
            router_weights.stride(1),
            triton.next_power_of_2(num_rows),
            1024,
            num_warps=8,
            enable_fp_fusion=False,
        )
    else:
        if num_warps is None:
            num_warps = 1 if num_rows <= 4 else 4
        _dsv3_router_gemm_small_batch_kernel[(num_rows, num_experts)](
            hidden_states,
            router_weights,
            out,
            num_experts,
            hidden_size,
            hidden_states.stride(0),
            hidden_states.stride(1),
            router_weights.stride(0),
            router_weights.stride(1),
            triton.next_power_of_2(hidden_size),
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    return out


def dsv3_router_gemm(hidden_states, router_weights, out_dtype):
    num_rows, hidden_size = hidden_states.shape
    num_experts = router_weights.shape[0]
    if (
        hidden_states.dtype not in (torch.float16, torch.bfloat16)
        or router_weights.dtype != hidden_states.dtype
        or not num_rows
    ):
        return _scalar_router_gemm(hidden_states, router_weights, out_dtype)
    kind, cfg = _config(num_rows, num_experts)
    if kind == "scalar":
        return _scalar_router_gemm(
            hidden_states, router_weights, out_dtype, cfg
        )
    out = torch.empty(
        (num_rows, num_experts), dtype=out_dtype, device=hidden_states.device
    )
    if kind == "gemv":
        be, bk, warps = cfg
        _router_gemv_kernel[(triton.cdiv(num_experts, be),)](
            hidden_states,
            router_weights,
            out,
            num_rows,
            num_experts,
            hidden_size,
            *hidden_states.stride(),
            *router_weights.stride(),
            triton.next_power_of_2(num_rows),
            be,
            bk,
            num_warps=warps,
        )
        return out
    bm, be, bk, warps, stages, split = cfg
    partial = (
        out
        if split == 1
        else torch.empty(
            (split, num_rows, num_experts),
            dtype=torch.float32,
            device=hidden_states.device,
        )
    )
    _router_dot_kernel[
        (triton.cdiv(num_rows, bm), triton.cdiv(num_experts, be), split)
    ](
        hidden_states,
        router_weights,
        partial,
        num_rows,
        split,
        num_experts,
        hidden_size,
        *hidden_states.stride(),
        *router_weights.stride(),
        bm,
        be,
        bk,
        num_warps=warps,
        num_stages=stages,
    )
    if split > 1:
        size = num_rows * num_experts
        _router_sum_kernel[(triton.cdiv(size, 1024),)](
            partial, out, size, split, 1024, num_warps=4
        )
    return out


__all__ = ["dsv3_router_gemm"]
