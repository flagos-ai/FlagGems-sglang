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

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_full_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    slope_ptr,
    slot_ptr,
    out_ptr,
    H,
    D,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_vb,
    stride_vh,
    stride_cs,
    stride_ch,
    stride_ck,
    stride_ob,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    slot = tl.load(slot_ptr + b)
    if slot == -1:
        return

    decay = tl.exp(-tl.load(slope_ptr + h).to(tl.float32))

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < D

    q_base = q_ptr + b * stride_qb + h * stride_qh
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_vb + h * stride_vh
    state_base = cache_ptr + slot.to(tl.int64) * stride_cs + h * stride_ch

    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < D

    k_vec = tl.load(k_base + rows, mask=row_mask, other=0.0).to(tl.float32)
    v_vec = tl.load(v_base + cols, mask=col_mask, other=0.0).to(tl.float32)
    q_vec = tl.load(q_base + rows, mask=row_mask, other=0.0).to(tl.float32)

    state_ptrs = (
        state_base + rows[:, None].to(tl.int64) * stride_ck + cols[None, :]
    )
    state_mask = row_mask[:, None] & col_mask[None, :]
    old_state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)

    new_state = k_vec[:, None] * v_vec[None, :] + decay * old_state
    tl.store(
        state_ptrs,
        new_state.to(cache_ptr.dtype.element_ty),
        mask=state_mask,
    )

    acc = tl.sum(q_vec[:, None] * new_state, axis=0)
    tl.store(
        out_ptr + b * stride_ob + h * D + cols,
        acc.to(out_ptr.dtype.element_ty),
        mask=col_mask,
    )


@triton.jit
def _decode_loop_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    slope_ptr,
    slot_ptr,
    out_ptr,
    H,
    D,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_vb,
    stride_vh,
    stride_cs,
    stride_ch,
    stride_ck,
    stride_ob,
    ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOAD_CG: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_c = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    slot = tl.load(slot_ptr + b)
    if slot == -1:
        return

    decay = tl.exp(-tl.load(slope_ptr + h).to(tl.float32))

    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < D

    q_base = q_ptr + b * stride_qb + h * stride_qh
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_vb + h * stride_vh
    state_base = cache_ptr + slot.to(tl.int64) * stride_cs + h * stride_ch

    v_vec = tl.load(v_base + cols, mask=col_mask, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for i in range(0, D, ROWS):
        rows = i + tl.arange(0, ROWS)
        row_mask = rows < D
        k_vec = tl.load(k_base + rows, mask=row_mask, other=0.0).to(tl.float32)
        q_vec = tl.load(q_base + rows, mask=row_mask, other=0.0).to(tl.float32)
        state_ptrs = (
            state_base + rows[:, None].to(tl.int64) * stride_ck + cols[None, :]
        )
        state_mask = row_mask[:, None] & col_mask[None, :]
        if LOAD_CG:
            old_state = tl.load(
                state_ptrs, mask=state_mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
        else:
            old_state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(
                tl.float32
            )
        new_state = k_vec[:, None] * v_vec[None, :] + decay * old_state
        tl.store(
            state_ptrs,
            new_state.to(cache_ptr.dtype.element_ty),
            mask=state_mask,
        )
        acc += tl.sum(q_vec[:, None] * new_state, axis=0)

    tl.store(
        out_ptr + b * stride_ob + h * D + cols,
        acc.to(out_ptr.dtype.element_ty),
        mask=col_mask,
    )


def linear_decode_forward(
    q, k, v, kv_caches, slope_rate, slot_idx, BLOCK_SIZE
):
    b, h, _, d = q.shape

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    kv_caches = kv_caches.contiguous()
    slope_rate = slope_rate.contiguous()

    out = torch.empty((b, h * d), dtype=q.dtype, device=q.device)

    block_d = max(triton.next_power_of_2(d), 16)
    num_bh = b * h

    if num_bh <= 16:
        n_col = min(8, block_d // 16)
        _decode_full_kernel[(num_bh, n_col)](
            q,
            k,
            v,
            kv_caches,
            slope_rate,
            slot_idx,
            out,
            h,
            d,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            kv_caches.stride(0),
            kv_caches.stride(1),
            kv_caches.stride(2),
            out.stride(0),
            BLOCK_N=block_d // n_col,
            BLOCK_R=block_d,
            num_warps=2,
            num_stages=1,
        )
        return out, kv_caches

    if num_bh <= 64:
        rows, block_c, num_warps, load_cg = 8, 16, 1, False
    elif num_bh <= 256:
        rows, block_c, num_warps, load_cg = 4, 16, 1, True
    elif num_bh <= 512:
        rows, block_c, num_warps, load_cg = 2, 32, 1, True
    else:
        rows, block_c, num_warps, load_cg = 1, min(128, block_d), 2, True

    rows = min(rows, block_d)
    block_c = min(block_c, block_d)
    n_col = triton.cdiv(block_d, block_c)
    _decode_loop_kernel[(num_bh, n_col)](
        q,
        k,
        v,
        kv_caches,
        slope_rate,
        slot_idx,
        out,
        h,
        d,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        kv_caches.stride(0),
        kv_caches.stride(1),
        kv_caches.stride(2),
        out.stride(0),
        ROWS=rows,
        BLOCK_C=block_c,
        LOAD_CG=load_cg,
        num_warps=num_warps,
        num_stages=1,
    )
    return out, kv_caches


__all__ = ["linear_decode_forward"]
