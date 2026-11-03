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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Fused affine-free LayerNorm over the head dimension of Q and K."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_head_layernorm_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    SEQ_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    Q_S0: tl.constexpr,
    Q_S1: tl.constexpr,
    Q_S2: tl.constexpr,
    Q_S3: tl.constexpr,
    K_S0: tl.constexpr,
    K_S1: tl.constexpr,
    K_S2: tl.constexpr,
    K_S3: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    WIDE: tl.constexpr,
):
    src_ptr = q_ptr
    out_ptr = q_out_ptr
    if tl.program_id(1) == 1:
        src_ptr = k_ptr
        out_ptr = k_out_ptr
    col = tl.arange(0, BLOCK_D)
    if WIDE:
        col = col.to(tl.int64)
    if D == BLOCK_D:
        safe_col = col
    else:
        safe_col = tl.minimum(col, D - 1)
    for group in range(
        tl.program_id(0), tl.cdiv(NUM_ROWS, BLOCK_ROWS), tl.num_programs(0)
    ):
        group_index = group
        if WIDE:
            group_index = group_index.to(tl.int64)
        row = group_index * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        if NUM_ROWS % BLOCK_ROWS == 0:
            safe_row = row
        else:
            safe_row = tl.minimum(row, NUM_ROWS - 1)
        if (
            Q_S3 == 1
            and Q_S2 == D
            and Q_S1 == NUM_HEADS * D
            and Q_S0 == SEQ_LEN * NUM_HEADS * D
            and K_S3 == 1
            and K_S2 == D
            and K_S1 == NUM_HEADS * D
            and K_S0 == SEQ_LEN * NUM_HEADS * D
        ):
            offset = safe_row[:, None] * D + safe_col[None, :]
        else:
            q_offset = (
                safe_row[:, None] // (SEQ_LEN * NUM_HEADS) * Q_S0
                + safe_row[:, None] // NUM_HEADS % SEQ_LEN * Q_S1
                + safe_row[:, None] % NUM_HEADS * Q_S2
                + safe_col[None, :] * Q_S3
            )
            if Q_S0 == K_S0 and Q_S1 == K_S1 and Q_S2 == K_S2 and Q_S3 == K_S3:
                offset = q_offset
            else:
                k_offset = (
                    safe_row[:, None] // (SEQ_LEN * NUM_HEADS) * K_S0
                    + safe_row[:, None] // NUM_HEADS % SEQ_LEN * K_S1
                    + safe_row[:, None] % NUM_HEADS * K_S2
                    + safe_col[None, :] * K_S3
                )
                offset = tl.where(tl.program_id(1) == 0, q_offset, k_offset)
        # Clamped load addresses are valid; only uniquely owned lanes store.
        x = tl.load(src_ptr + offset).to(tl.float32)
        x = tl.where(col[None, :] < D, x, 0.0)
        mean = tl.sum(x, 1) / D
        centered = tl.where(col[None, :] < D, x - mean[:, None], 0.0)
        variance = tl.sum(centered * centered, 1) / D
        result = centered * tl.rsqrt(variance[:, None] + EPS)
        tl.store(
            out_ptr + row[:, None] * D + col[None, :],
            result,
            (row[:, None] < NUM_ROWS) & (col[None, :] < D),
        )


@triton.jit
def _qk_head_layernorm_packed_kernel(
    q_ptr,
    k_ptr,
    out_ptr,
    D: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    WIDE: tl.constexpr,
):
    col = tl.arange(0, D)
    tile_row = tl.arange(0, BLOCK_ROWS)
    tensor_id = tl.program_id(1)
    src_ptr = q_ptr
    if tensor_id == 1:
        src_ptr = k_ptr
    if WIDE:
        col = col.to(tl.int64)
        tile_row = tile_row.to(tl.int64)
        tensor_id = tensor_id.to(tl.int64)
    # The wrapper guarantees whole tiles; no row clamp or tail mask is needed.
    tile_offset = tile_row[:, None] * D + col[None, :]
    out_row_ptr = out_ptr + tensor_id * NUM_ROWS * D
    for group in range(
        tl.program_id(0), tl.cdiv(NUM_ROWS, BLOCK_ROWS), tl.num_programs(0)
    ):
        group_index = group
        if WIDE:
            group_index = group_index.to(tl.int64)
        group_base = group_index * BLOCK_ROWS * D
        src_row_ptr = src_ptr + group_base
        x = tl.load(
            src_row_ptr + tile_offset, offset_state_policy="continuous"
        ).to(tl.float32)
        mean = tl.sum(x, 1) / D
        centered = x - mean[:, None]
        variance = tl.sum(centered * centered, 1) / D
        result = centered * tl.rsqrt(variance[:, None] + EPS)
        tl.store(
            out_row_ptr + group_base + tile_offset,
            result,
            offset_state_policy="continuous",
        )


def qk_head_layernorm(q, k, eps):
    """Return normalized Q/K; all inputs follow the task."""
    batch, seq, heads, dim = q.shape
    rows = batch * seq * heads
    block_dim = triton.next_power_of_2(dim)
    if rows:
        q_span = sum(
            (n - 1) * stride for n, stride in zip(q.shape, q.stride())
        )
        k_span = sum(
            (n - 1) * stride for n, stride in zip(k.shape, k.stride())
        )
        wide = (
            max(q_span, k_span, 2 * rows * dim + 64 * dim + block_dim) >= 2**31
        )
    else:
        wide = False
    if (
        rows
        and dim == block_dim
        and rows % 64 == 0
        and q.is_contiguous()
        and k.is_contiguous()
    ):
        outputs = torch.empty(
            (2, batch, seq, heads, dim), dtype=q.dtype, device=q.device
        )
        _qk_head_layernorm_packed_kernel[(min(rows // 64, 12), 2)](
            q,
            k,
            outputs,
            dim,
            rows,
            eps,
            64,
            wide,
            groups_per_cluster=64,
            isCloseUnrollControl=True,
            enable_fp_fusion=False,
        )
        return outputs[0], outputs[1]
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if rows:
        _qk_head_layernorm_kernel[(min(triton.cdiv(rows, 4), 12), 2)](
            q,
            k,
            q_out,
            k_out,
            seq,
            heads,
            dim,
            rows,
            *q.stride(),
            *k.stride(),
            eps,
            block_dim,
            4,
            wide,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q_out, k_out


__all__ = ["qk_head_layernorm"]
