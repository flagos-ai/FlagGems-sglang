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


@triton.jit(do_not_specialize=["batch_idx"])
def _linear_decode_state_kernel(
    k_ptr,
    v_ptr,
    cache_ptr,
    slope_ptr,
    slot_ptr,
    tmp_ptr,
    batch_idx,
    H: tl.constexpr,
    D: tl.constexpr,
    K_STRIDE_B: tl.constexpr,
    K_STRIDE_H: tl.constexpr,
    K_STRIDE_D: tl.constexpr,
    V_STRIDE_B: tl.constexpr,
    V_STRIDE_H: tl.constexpr,
    V_STRIDE_D: tl.constexpr,
    CACHE_STRIDE_S: tl.constexpr,
    CACHE_STRIDE_H: tl.constexpr,
    CACHE_STRIDE_R: tl.constexpr,
    CACHE_STRIDE_C: tl.constexpr,
    SLOPE_STRIDE: tl.constexpr,
    SLOT_IDX_STRIDE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.load(slot_ptr + batch_idx * SLOT_IDX_STRIDE).to(tl.int64)

    if slot >= 0:
        for tile in range(
            tl.program_id(0),
            NUM_TILES,
            tl.num_programs(0),
        ):
            off = tile * BLOCK + tl.arange(0, BLOCK)

            head = off // (D * D)
            row = (off // D) % D
            col = off % D

            mask = off < H * D * D

            k_val = tl.load(
                k_ptr
                + batch_idx * K_STRIDE_B
                + head * K_STRIDE_H
                + row * K_STRIDE_D,
                mask=mask,
                other=0,
            ).to(tl.float32)

            v_val = tl.load(
                v_ptr
                + batch_idx * V_STRIDE_B
                + head * V_STRIDE_H
                + col * V_STRIDE_D,
                mask=mask,
                other=0,
            ).to(tl.float32)

            slope = tl.load(
                slope_ptr + tl.minimum(head, H - 1) * SLOPE_STRIDE
            ).to(tl.float32)

            cache_ptr_cur = cache_ptr + (
                slot * CACHE_STRIDE_S
                + head.to(tl.int64) * CACHE_STRIDE_H
                + row.to(tl.int64) * CACHE_STRIDE_R
                + col.to(tl.int64) * CACHE_STRIDE_C
            )

            old_state = tl.load(
                cache_ptr_cur,
                mask=mask,
                other=0,
            ).to(tl.float32)

            new_state = k_val * v_val + tl.exp(-slope) * old_state

            tl.store(
                tmp_ptr + off,
                new_state,
                mask=mask,
            )

            tl.store(
                cache_ptr_cur,
                new_state,
                mask=mask,
            )


@triton.jit(do_not_specialize=["batch_idx"])
def _linear_decode_out_kernel(
    q_ptr,
    tmp_ptr,
    slot_ptr,
    out_ptr,
    batch_idx,
    H: tl.constexpr,
    D: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    Q_STRIDE_D: tl.constexpr,
    SLOT_IDX_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.load(slot_ptr + batch_idx * SLOT_IDX_STRIDE)

    if slot >= 0:
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

        safe_off = tl.minimum(off, H * D - 1)
        head = safe_off // D
        col = safe_off % D

        acc = tl.zeros((BLOCK,), dtype=tl.float32)

        for row in range(D):
            q_val = tl.load(
                q_ptr
                + batch_idx * Q_STRIDE_B
                + head * Q_STRIDE_H
                + row * Q_STRIDE_D
            ).to(tl.float32)

            state = tl.load(tmp_ptr + (head * D + row) * D + col)

            acc += q_val * state

        tl.store(
            out_ptr + batch_idx * H * D + off,
            acc,
            mask=off < H * D,
        )


@triton.jit
def _linear_decode_single_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    slope_ptr,
    slot_ptr,
    out_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    Q_STRIDE_D: tl.constexpr,
    K_STRIDE_H: tl.constexpr,
    K_STRIDE_D: tl.constexpr,
    V_STRIDE_H: tl.constexpr,
    V_STRIDE_D: tl.constexpr,
    CACHE_STRIDE_S: tl.constexpr,
    CACHE_STRIDE_H: tl.constexpr,
    CACHE_STRIDE_R: tl.constexpr,
    SLOPE_STRIDE: tl.constexpr,
    COL_TILES: tl.constexpr,
):
    program_id = tl.program_id(0)

    head = program_id // COL_TILES
    col = (program_id % COL_TILES) * 64 + tl.arange(0, 64)

    slot = tl.load(slot_ptr).to(tl.int64)

    if slot >= 0:
        v_val = tl.load(v_ptr + head * V_STRIDE_H + col * V_STRIDE_D).to(
            tl.float32
        )

        slope = tl.load(slope_ptr + head * SLOPE_STRIDE).to(tl.float32)

        decay = tl.exp(-slope)

        cache_base = cache_ptr + slot * CACHE_STRIDE_S + head * CACHE_STRIDE_H

        acc = tl.zeros((64,), dtype=tl.float32)

        for row in range(D):
            q_val = tl.load(q_ptr + head * Q_STRIDE_H + row * Q_STRIDE_D).to(
                tl.float32
            )

            k_val = tl.load(k_ptr + head * K_STRIDE_H + row * K_STRIDE_D).to(
                tl.float32
            )

            old_state = tl.load(cache_base + row * CACHE_STRIDE_R + col).to(
                tl.float32
            )

            new_state = k_val * v_val + decay * old_state

            acc += q_val * new_state

            tl.store(
                cache_base + row * CACHE_STRIDE_R + col,
                new_state,
            )

        tl.store(
            out_ptr + head * D + col,
            acc,
        )


@triton.jit(do_not_specialize=["batch_idx"])
def _linear_decode_fused_matrix_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    slope_ptr,
    slot_ptr,
    out_ptr,
    batch_idx,
    H: tl.constexpr,
    D: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    Q_STRIDE_D: tl.constexpr,
    K_STRIDE_B: tl.constexpr,
    K_STRIDE_H: tl.constexpr,
    K_STRIDE_D: tl.constexpr,
    V_STRIDE_B: tl.constexpr,
    V_STRIDE_H: tl.constexpr,
    V_STRIDE_D: tl.constexpr,
    CACHE_STRIDE_S: tl.constexpr,
    CACHE_STRIDE_H: tl.constexpr,
    CACHE_STRIDE_R: tl.constexpr,
    SLOPE_STRIDE: tl.constexpr,
    SLOT_IDX_STRIDE: tl.constexpr,
    COL_TILES: tl.constexpr,
):
    program_id = tl.program_id(0)

    head = program_id // COL_TILES

    col = (program_id % COL_TILES) * 64 + tl.arange(0, 64)

    row = tl.arange(0, D)

    slot = tl.load(slot_ptr + batch_idx * SLOT_IDX_STRIDE).to(tl.int64)

    safe_slot = tl.maximum(slot, 0)

    cache_offset = (
        safe_slot * CACHE_STRIDE_S
        + head.to(tl.int64) * CACHE_STRIDE_H
        + row[None, :].to(tl.int64) * CACHE_STRIDE_R
        + col[:, None].to(tl.int64)
    )

    old_state = tl.load(
        cache_ptr + cache_offset,
        mask=slot >= 0,
        other=0,
    ).to(tl.float32)

    q_val = tl.load(
        q_ptr
        + batch_idx * Q_STRIDE_B
        + head * Q_STRIDE_H
        + row[None, :] * Q_STRIDE_D
    ).to(tl.float32)

    k_val = tl.load(
        k_ptr
        + batch_idx * K_STRIDE_B
        + head * K_STRIDE_H
        + row[None, :] * K_STRIDE_D
    ).to(tl.float32)

    v_val = tl.load(
        v_ptr
        + batch_idx * V_STRIDE_B
        + head * V_STRIDE_H
        + col[:, None] * V_STRIDE_D
    ).to(tl.float32)

    slope = tl.load(slope_ptr + head * SLOPE_STRIDE).to(tl.float32)

    decay = tl.exp(-slope)

    new_state = k_val * v_val + decay * old_state

    value = tl.sum(q_val * new_state, 1)

    if slot >= 0:
        tl.store(
            cache_ptr + cache_offset,
            new_state,
        )

        tl.store(
            out_ptr
            + (batch_idx * H * D + head * D + col[:, None]).to(tl.int64),
            value[:, None],
        )


def linear_decode_forward(
    q,
    k,
    v,
    kv_caches,
    slope_rate,
    slot_idx,
    BLOCK_SIZE,
):
    bs, h, _, d = q.shape

    cache = kv_caches
    out = q.new_empty((bs, h * d))

    if bs == 0 or h == 0 or d == 0:
        return out, cache

    q_stride = q.stride()
    k_stride = k.stride()
    v_stride = v.stride()
    cache_stride = cache.stride()

    slope_stride = slope_rate.stride(0)
    slot_idx_stride = slot_idx.stride(0)

    num_tiles = triton.cdiv(
        h * d * d,
        512,
    )

    aligned = (
        d % 64 == 0
        and cache_stride[3] == 1
        and cache.data_ptr() % 128 == 0
        and all(
            (cache_stride[j] * cache.element_size()) % 128 == 0
            for j in range(3)
        )
    )

    # Fast path 1:
    # Fuse state update and output computation.
    if aligned and d <= 256 and d & (d - 1) == 0:
        for batch_idx in range(bs):
            _linear_decode_fused_matrix_kernel[(h * triton.cdiv(d, 64),)](
                q,
                k,
                v,
                cache,
                slope_rate,
                slot_idx,
                out,
                batch_idx,
                h,
                d,
                q_stride[0],
                q_stride[1],
                q_stride[3],
                k_stride[0],
                k_stride[1],
                k_stride[3],
                v_stride[0],
                v_stride[1],
                v_stride[3],
                cache_stride[0],
                cache_stride[1],
                cache_stride[2],
                slope_stride,
                slot_idx_stride,
                triton.cdiv(d, 64),
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
                isCloseUnrollControl=True,
            )

        return out, cache

    # Fast path 2:
    # Single-batch case with float32 cache.
    if bs == 1 and cache.dtype == torch.float32 and aligned:
        _linear_decode_single_kernel[(h * triton.cdiv(d, 64),)](
            q,
            k,
            v,
            cache,
            slope_rate,
            slot_idx,
            out,
            h,
            d,
            q_stride[1],
            q_stride[3],
            k_stride[1],
            k_stride[3],
            v_stride[1],
            v_stride[3],
            cache_stride[0],
            cache_stride[1],
            cache_stride[2],
            slope_stride,
            triton.cdiv(d, 64),
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            isCloseUnrollControl=True,
        )

        return out, cache

    # General path:
    # Update cache first, then compute output.
    tmp = torch.empty(
        (h, d, d),
        device=q.device,
        dtype=torch.float32,
    )

    for batch_idx in range(bs):
        _linear_decode_state_kernel[(num_tiles if aligned else 1,)](
            k,
            v,
            cache,
            slope_rate,
            slot_idx,
            tmp,
            batch_idx,
            h,
            d,
            k_stride[0],
            k_stride[1],
            k_stride[3],
            v_stride[0],
            v_stride[1],
            v_stride[3],
            cache_stride[0],
            cache_stride[1],
            cache_stride[2],
            cache_stride[3],
            slope_stride,
            slot_idx_stride,
            num_tiles,
            512,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            isCloseUnrollControl=True,
        )

        _linear_decode_out_kernel[(triton.cdiv(h * d, 128),)](
            q,
            tmp,
            slot_idx,
            out,
            batch_idx,
            h,
            d,
            q_stride[0],
            q_stride[1],
            q_stride[3],
            slot_idx_stride,
            128,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            isCloseUnrollControl=True,
        )

    return out, cache


__all__ = ["linear_decode_forward"]
