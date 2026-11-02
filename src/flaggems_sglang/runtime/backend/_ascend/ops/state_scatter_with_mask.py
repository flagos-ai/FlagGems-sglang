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
def _state_scatter_fused_kernel(
    dst_ptr,  # [L, C, *state] contiguous, E = prod(state)
    src_ptr,  # [L, R, draft, *state] contiguous
    out_ptr,  # [L, C, *state] contiguous
    dst_idx_ptr,  # [R] int32 (or int64)
    step_idx_ptr,  # [R] int32 (or int64)
    R,
    E,
    C,
    ss_layer,
    ss_req,
    ss_draft,  # src layer/request/draft strides (elements)
    sd_layer,
    sd_cache,  # dst layer/cache strides
    so_layer,
    so_cache,  # out layer/cache strides
    RB: tl.constexpr,  # cache rows owned by one program
    BR_CHUNK: tl.constexpr,  # request-scan chunk (pow2)
    BLOCK: tl.constexpr,  # row tile width
):
    rows = tl.program_id(0) * RB + tl.arange(0, RB)  # cache rows owned
    layer_idx = tl.program_id(1)
    rows_ok = rows < C

    own = tl.full((RB,), -1, tl.int32)
    stp = tl.full((RB,), -1, tl.int32)
    for r0 in range(0, R, BR_CHUNK):
        r = r0 + tl.arange(0, BR_CHUNK)
        rm = r < R
        d = tl.load(dst_idx_ptr + r, mask=rm, other=-1)
        s = tl.load(step_idx_ptr + r, mask=rm, other=-1)
        mt = (rows[:, None] == d[None, :]) & (s[None, :] >= 0) & rm[None, :]
        own = tl.maximum(own, tl.max(tl.where(mt, r[None, :], -1), axis=1))
        stp = tl.maximum(stp, tl.max(tl.where(mt, s[None, :], -1), axis=1))
    live = (own >= 0) & rows_ok
    # Dead rows never dereference src (masked), so clamp their address to 0.
    own_s = tl.where(live, own, 0)
    stp_s = tl.where(live, stp, 0)

    # --- write the row tiles exactly once: gather live rows from src,
    #     copy untouched rows from dst, in a single masked store ---
    e = tl.arange(0, BLOCK)
    src_off = (
        layer_idx * ss_layer
        + own_s[:, None] * ss_req
        + stp_s[:, None] * ss_draft
        + e[None, :]
    )
    dst_off = layer_idx * sd_layer + rows[:, None] * sd_cache + e[None, :]
    out_off = layer_idx * so_layer + rows[:, None] * so_cache + e[None, :]
    for off in range(0, E, BLOCK):
        em = (off + e[None, :]) < E
        v = tl.load(
            src_ptr + src_off + off, mask=live[:, None] & em, other=0.0
        ) + tl.load(
            dst_ptr + dst_off + off,
            mask=(~live)[:, None] & rows_ok[:, None] & em,
            other=0.0,
        )
        tl.store(out_ptr + out_off + off, v, mask=em & rows_ok[:, None])


def state_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):
    if not dst.is_contiguous():
        dst = dst.contiguous()
    if not src.is_contiguous():
        src = src.contiguous()

    L, C = dst.shape[0], dst.shape[1]
    E = dst[0, 0].numel()
    R = src.shape[1]
    out = torch.empty_like(dst)
    if L == 0 or C == 0 or E == 0:
        return out

    RB = 8
    # Scan chunk: exact pow2 of R when small (single chunk, no wasted
    # lanes), capped at 256 so the [RB, BR_CHUNK] match tile fits in UB.
    br_chunk = min(256, max(1, triton.next_power_of_2(R)))
    _state_scatter_fused_kernel[(triton.cdiv(C, RB), L)](
        dst,
        src,
        out,
        dst_indices_raw,
        step_indices_raw,
        R,
        E,
        C,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        dst.stride(0),
        dst.stride(1),
        out.stride(0),
        out.stride(1),
        RB=RB,
        BR_CHUNK=br_chunk,
        BLOCK=1024,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["state_scatter_with_mask"]
