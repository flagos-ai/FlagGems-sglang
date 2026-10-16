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
def _linear_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    slope_ptr,
    slot_ptr,
    out_ptr,
    H,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One program per (batch, head); walks the state in BLOCK_K row."""
    pid_bh = tl.program_id(0)  # flat (batch, head) id
    pid_v = tl.program_id(1)  # value-dim block id (1 when BLOCK_V == D)

    b = pid_bh // H
    h = pid_bh % H

    # Padding row: neither the cache nor the output row is touched.
    slot = tl.load(slot_ptr + b)
    if slot < 0:
        return

    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    # Decay applies to the previous state only; the new outer product is not
    # decayed.
    decay = tl.exp(-tl.load(slope_ptr + h))

    # Inputs are contiguous [B, H, 1, D]; the token axis is size 1.
    base_qkv = (b * H + h) * D
    v = tl.load(v_ptr + base_qkv + offs_v).to(tl.float32)

    cache_base = cache_ptr + (slot * H + h) * D * D + offs_v[None, :]
    out_acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    for kb in tl.range(0, D // BLOCK_K):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(q_ptr + base_qkv + offs_k).to(tl.float32)
        k = tl.load(k_ptr + base_qkv + offs_k).to(tl.float32)
        cache_ptrs = cache_base + offs_k[:, None] * D
        cache = tl.load(cache_ptrs).to(tl.float32)
        state = k[:, None] * v[None, :] + decay * cache
        tl.store(cache_ptrs, state.to(cache_ptr.dtype.element_ty))
        out_acc += tl.sum(q[:, None] * state, axis=0)

    # out[b, h, offs_v] = sum_k q[k] * state[k, offs_v]; chunks of `out` are
    # disjoint across v-blocks, so each program stores its own slice.
    tl.store(
        out_ptr + b * (H * D) + h * D + offs_v,
        out_acc.to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _linear_decode_kernel_masked(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    slope_ptr,
    slot_ptr,
    out_ptr,
    H,
    D,
    D_PAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Generic fallback for non-power-of-two D: padded lanes are masked off.
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    slot = tl.load(slot_ptr + b)
    if slot < 0:
        return

    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < D

    decay = tl.exp(-tl.load(slope_ptr + h))

    base_qkv = (b * H + h) * D
    v = tl.load(v_ptr + base_qkv + offs_v, mask=mask_v, other=0.0).to(
        tl.float32
    )

    cache_base = cache_ptr + (slot * H + h) * D * D + offs_v[None, :]
    out_acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    for kb in tl.static_range(D_PAD // BLOCK_K):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D
        q = tl.load(q_ptr + base_qkv + offs_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        k = tl.load(k_ptr + base_qkv + offs_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        tile_mask = mask_k[:, None] & mask_v[None, :]
        cache_ptrs = cache_base + offs_k[:, None] * D
        cache = tl.load(cache_ptrs, mask=tile_mask, other=0.0).to(tl.float32)
        state = k[:, None] * v[None, :] + decay * cache
        tl.store(
            cache_ptrs, state.to(cache_ptr.dtype.element_ty), mask=tile_mask
        )
        out_acc += tl.sum(q[:, None] * state, axis=0)

    tl.store(
        out_ptr + b * (H * D) + h * D + offs_v,
        out_acc.to(out_ptr.dtype.element_ty),
        mask=mask_v,
    )


def _launch_config(block_d, num_bh):
    """(BLOCK_K, BLOCK_V, num_warps, num_stages) for the power-of-two."""
    if block_d <= 32:
        # Tiny tiles: more warps than rows of work only adds scheduling cost.
        return block_d, block_d, 1, 1
    if block_d <= 128:
        if num_bh <= 128:
            num_warps = 4 if num_bh <= 16 else 1
            return block_d, min(32, block_d), num_warps, 1
        # Bandwidth-bound: 1 warp, BLOCK_K=64, 2-stage pipeline (~2% faster
        # than 4 warps over the full [D, D] tile, which spills).
        return min(64, block_d), block_d, 1, 2
    # D > 128: chunk the k dimension to bound the register tile.
    return 64, 64, 4, 1


def linear_decode_forward(
    q, k, v, kv_caches, slope_rate, slot_idx, BLOCK_SIZE
):
    b, h, _, d = q.shape

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not kv_caches.is_contiguous():
        kv_caches = kv_caches.contiguous()

    out = torch.empty((b, h * d), dtype=q.dtype, device=q.device)

    block_d = triton.next_power_of_2(d)
    block_k, block_v, num_warps, num_stages = _launch_config(block_d, b * h)

    if d == block_d:
        # Fast path: D is a power of two, no masks needed anywhere.
        grid = (b * h, triton.cdiv(d, block_v))
        _linear_decode_kernel[grid](
            q,
            k,
            v,
            kv_caches,
            slope_rate,
            slot_idx,
            out,
            h,
            D=block_d,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        grid = (b * h, triton.cdiv(d, block_v))
        _linear_decode_kernel_masked[grid](
            q,
            k,
            v,
            kv_caches,
            slope_rate,
            slot_idx,
            out,
            h,
            d,
            D_PAD=block_d,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out, kv_caches


__all__ = ["linear_decode_forward"]
