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

_ROW_BLOCK_MIN_BH = 1024

# Largest head dim served by the whole-tile fast arm. [D, D] fp32 tiles at
# D=128 compile to gcu.dsm_memory_size = 133120 (< 224KB keeps the L600
# double-program-per-SP concurrency); larger D is left to the split arms.
_TILE_FAST_MAX_D = 128


@triton.jit
def _linear_decode_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    slope_ptr,
    slot_ptr,
    out_ptr,
    H,
    NCB,
    D: tl.constexpr,
    DP: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    cb = pid % NCB
    bh = pid // NCB
    b = bh // H
    h = bh % H

    slot = tl.load(slot_ptr + b)
    if slot == -1:
        return

    decay = tl.exp(-tl.load(slope_ptr + h).to(tl.float32))

    offs_k = tl.arange(0, DP)
    offs_n = cb * BN + tl.arange(0, BN)
    kmask = offs_k < D
    nmask = offs_n < D

    q = tl.load(q_ptr + bh * D + offs_k, mask=kmask, other=0.0).to(tl.float32)
    kk = tl.load(k_ptr + bh * D + offs_k, mask=kmask, other=0.0).to(tl.float32)
    vv = tl.load(v_ptr + bh * D + offs_n, mask=nmask, other=0.0).to(tl.float32)

    base = cache_ptr + slot * (H * D * D) + h * (D * D)
    ptrs = base + offs_k[:, None] * D + offs_n[None, :]
    mask2 = kmask[:, None] & nmask[None, :]

    cache = tl.load(ptrs, mask=mask2, other=0.0).to(tl.float32)
    state = kk[:, None] * vv[None, :] + decay * cache
    tl.store(ptrs, state.to(cache_ptr.dtype.element_ty), mask=mask2)

    out = tl.sum(q[:, None] * state, axis=0)
    tl.store(
        out_ptr + b * (H * D) + h * D + offs_n,
        out.to(out_ptr.dtype.element_ty),
        mask=nmask,
    )


@triton.jit
def _linear_decode_forward_rowblock_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    slope_ptr,
    slot_ptr,
    out_ptr,
    H,
    NRB,
    D: tl.constexpr,
    BR: tl.constexpr,
):
    pid = tl.program_id(0)
    rb = pid % NRB
    bh = pid // NRB
    b = bh // H
    h = bh % H

    slot = tl.load(slot_ptr + b)
    if slot == -1:
        return

    decay = tl.exp(-tl.load(slope_ptr + h).to(tl.float32))

    offs_r = rb * BR + tl.arange(0, BR)
    offs_n = tl.arange(0, D)

    q = tl.load(q_ptr + bh * D + offs_r).to(tl.float32)
    kk = tl.load(k_ptr + bh * D + offs_r).to(tl.float32)
    vv = tl.load(v_ptr + bh * D + offs_n).to(tl.float32)

    base = cache_ptr + slot * (H * D * D) + h * (D * D)
    ptrs = base + offs_r[:, None] * D + offs_n[None, :]

    cache = tl.load(ptrs).to(tl.float32)
    state = kk[:, None] * vv[None, :] + decay * cache
    tl.store(ptrs, state.to(cache_ptr.dtype.element_ty))

    out = tl.sum(q[:, None] * state, axis=0)
    tl.atomic_add(
        out_ptr + b * (H * D) + h * D + offs_n,
        out.to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _linear_decode_forward_tile_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    slope_ptr,
    slot_ptr,
    out_ptr,
    H,
    D: tl.constexpr,
):
    # Enflame GCU (L600) fast arm: one program per (b*h) covers the whole
    # [D, D] state tile with no masks (host dispatch guarantees D is a
    # power of two <= 128, so D % BN == 0 and DP == D trivially). Probe
    # ablation on (64, 8, 128) fp32 showed the generic masked column-split
    # kernel is dominated by (a) the runtime `offs_n < D` predication on
    # every staged tile load/store (removed by the whole-tile split) and
    # (b) the num_warps=4 cross-warp sync on the tile compute + axis-0
    # reduction (removed by num_warps=1): 211.3us -> 64.3us, do_bench
    # median; topsprof Avg 62.0us.
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H

    slot = tl.load(slot_ptr + b)
    if slot == -1:
        return

    decay = tl.exp(-tl.load(slope_ptr + h).to(tl.float32))

    offs = tl.arange(0, D)

    q = tl.load(q_ptr + bh * D + offs).to(tl.float32)
    kk = tl.load(k_ptr + bh * D + offs).to(tl.float32)
    vv = tl.load(v_ptr + bh * D + offs).to(tl.float32)

    base = cache_ptr + slot * (H * D * D) + h * (D * D)
    ptrs = base + offs[:, None] * D + offs[None, :]

    cache = tl.load(ptrs).to(tl.float32)
    state = kk[:, None] * vv[None, :] + decay * cache
    tl.store(ptrs, state.to(cache_ptr.dtype.element_ty))

    out = tl.sum(q[:, None] * state, axis=0)
    tl.store(
        out_ptr + b * (H * D) + h * D + offs,
        out.to(out_ptr.dtype.element_ty),
    )


def _launch_config(d, n_bh):
    dp = triton.next_power_of_2(max(d, 16))
    if dp <= 64:
        bn = dp
        num_warps = 4 if dp <= 32 else 2
        num_stages = 3 if dp <= 32 else 2
    elif n_bh <= 16:
        bn, num_warps, num_stages = 16, 4, 2
    else:
        bn, num_warps, num_stages = 64, 4, 2
    return min(bn, dp), num_warps, num_stages


def linear_decode_forward(
    q, k, v, kv_caches, slope_rate, slot_idx, BLOCK_SIZE
):
    b, h, _, d = q.shape
    n_bh = b * h

    dp = triton.next_power_of_2(max(d, 16))

    if dp == d and d <= _TILE_FAST_MAX_D:
        # Whole-tile fast arm (any n_bh): also replaces the rowblock/atomic
        # path for large batches (b=256, n_bh=2048: 23099.7us -> 237.5us).
        out = torch.empty(b, h * d, dtype=q.dtype, device=q.device)
        _linear_decode_forward_tile_kernel[(n_bh,)](
            q,
            k,
            v,
            kv_caches,
            slope_rate,
            slot_idx,
            out,
            h,
            D=d,
            num_warps=1,
            num_stages=2,
        )
        return out, kv_caches

    if n_bh >= _ROW_BLOCK_MIN_BH and dp == d and d % 8 == 0:
        out = torch.zeros(b, h * d, dtype=q.dtype, device=q.device)
        br, num_warps = 8, 2
        nrb = d // br
        _linear_decode_forward_rowblock_kernel[(n_bh * nrb,)](
            q,
            k,
            v,
            kv_caches,
            slope_rate,
            slot_idx,
            out,
            h,
            nrb,
            D=d,
            BR=br,
            num_warps=num_warps,
            num_stages=1,
        )
        return out, kv_caches

    out = torch.empty(b, h * d, dtype=q.dtype, device=q.device)

    bn, num_warps, num_stages = _launch_config(d, n_bh)

    ncb = triton.cdiv(d, bn)
    _linear_decode_forward_kernel[(n_bh * ncb,)](
        q,
        k,
        v,
        kv_caches,
        slope_rate,
        slot_idx,
        out,
        h,
        ncb,
        D=d,
        DP=dp,
        BN=bn,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, kv_caches


__all__ = ["linear_decode_forward"]
