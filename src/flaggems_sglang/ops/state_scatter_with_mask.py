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

import triton
import triton.language as tl

_INT32_LIMIT = 2**31 - 1

_BLOCK_N_CAP = 2048


@triton.jit
def _state_scatter_kernel(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    src_sl,
    src_sr,
    src_ss,
    dst_sl,
    dst_sr,
    TOTAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NO_MASK: tl.constexpr,
    WIDE: tl.constexpr,
):
    req = tl.program_id(0)
    layer = tl.program_id(1)

    step = tl.load(step_idx_ptr + req)
    if step >= 0:
        slot = tl.load(dst_idx_ptr + req)
        n = tl.arange(0, BLOCK_N)
        if WIDE:
            n = n.to(tl.int64)
            dst_off = (
                layer.to(tl.int64) * dst_sl + slot.to(tl.int64) * dst_sr + n
            )
            src_off = (
                layer.to(tl.int64) * src_sl
                + req.to(tl.int64) * src_sr
                + step.to(tl.int64) * src_ss
                + n
            )
        else:
            dst_off = layer * dst_sl + slot * dst_sr + n
            src_off = layer * src_sl + req * src_sr + step * src_ss + n
        if NO_MASK:
            val = tl.load(src_ptr + src_off)
            tl.store(dst_ptr + dst_off, val)
        else:
            nmask = n < TOTAL
            val = tl.load(src_ptr + src_off, mask=nmask, other=0)
            tl.store(dst_ptr + dst_off, val, mask=nmask)


@triton.jit
def _state_scatter_kernel_tiled(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    src_sl,
    src_sr,
    src_ss,
    dst_sl,
    dst_sr,
    LAYERS: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NO_MASK: tl.constexpr,
    WIDE: tl.constexpr,
):
    req = tl.program_id(0)
    l0 = tl.program_id(1) * BLOCK_L
    n0 = tl.program_id(2) * BLOCK_N

    step = tl.load(step_idx_ptr + req)
    if step >= 0:
        slot = tl.load(dst_idx_ptr + req)
        layer_idx = l0 + tl.arange(0, BLOCK_L)[:, None]
        n = n0 + tl.arange(0, BLOCK_N)[None, :]
        if WIDE:
            layer_idx = layer_idx.to(tl.int64)
            n = n.to(tl.int64)
            dst_off = layer_idx * dst_sl + slot.to(tl.int64) * dst_sr + n
            src_off = (
                layer_idx * src_sl
                + req.to(tl.int64) * src_sr
                + step.to(tl.int64) * src_ss
                + n
            )
        else:
            dst_off = layer_idx * dst_sl + slot * dst_sr + n
            src_off = layer_idx * src_sl + req * src_sr + step * src_ss + n
        if NO_MASK:
            val = tl.load(src_ptr + src_off)
            tl.store(dst_ptr + dst_off, val)
        else:
            mask = (layer_idx < LAYERS) & (n < TOTAL)
            val = tl.load(src_ptr + src_off, mask=mask, other=0)
            tl.store(dst_ptr + dst_off, val, mask=mask)


def state_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):
    layers, sequences = src.shape[0], src.shape[1]
    total = (
        dst[0, 0].numel() if dst.dim() > 1 else dst.numel() // max(layers, 1)
    )

    if not dst.is_contiguous():
        dst = dst.contiguous()
    if not src.is_contiguous():
        src = src.contiguous()

    if sequences == 0 or layers == 0 or total == 0:
        return dst

    src_sl, src_sr, src_ss = src.stride(0), src.stride(1), src.stride(2)
    dst_sl, dst_sr = dst.stride(0), dst.stride(1)

    draft = src.shape[2]
    cache = dst.shape[1] if dst.dim() > 1 else 1
    src_span = (
        (layers - 1) * src_sl
        + (sequences - 1) * src_sr
        + (draft - 1) * src_ss
        + total
        - 1
    )
    dst_span = (layers - 1) * dst_sl + (cache - 1) * dst_sr + total - 1
    wide = max(src_span, dst_span) > _INT32_LIMIT

    if total <= _BLOCK_N_CAP:
        block_n = max(16, triton.next_power_of_2(total))
        if block_n >= 2048:
            num_warps = 8
        elif block_n >= 1024:
            num_warps = 4
        elif block_n >= 256:
            num_warps = 2
        else:
            num_warps = 1
        _state_scatter_kernel[(sequences, layers)](
            dst,
            src,
            dst_indices_raw,
            step_indices_raw,
            src_sl,
            src_sr,
            src_ss,
            dst_sl,
            dst_sr,
            TOTAL=total,
            BLOCK_N=block_n,
            NO_MASK=(total % block_n) == 0,
            WIDE=wide,
            num_warps=num_warps,
        )
        return dst

    block_l = 1 if layers <= 8 else 2
    block_n = _BLOCK_N_CAP
    _state_scatter_kernel_tiled[
        (sequences, triton.cdiv(layers, block_l), triton.cdiv(total, block_n))
    ](
        dst,
        src,
        dst_indices_raw,
        step_indices_raw,
        src_sl,
        src_sr,
        src_ss,
        dst_sl,
        dst_sr,
        LAYERS=layers,
        TOTAL=total,
        BLOCK_L=block_l,
        BLOCK_N=block_n,
        NO_MASK=(layers % block_l) == 0 and (total % block_n) == 0,
        WIDE=wide,
        num_warps=8,
    )
    return dst


__all__ = ["state_scatter_with_mask"]
