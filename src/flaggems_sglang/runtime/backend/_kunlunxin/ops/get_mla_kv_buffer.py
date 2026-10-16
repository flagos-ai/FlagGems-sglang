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

"""Operator: kvcache/get_mla_kv_buffer (pure Triton)."""

import torch
import triton
import triton.language as tl

_PACK_BLOCKS = (2048, 512, 128)
# Row count below which the fused single-launch kernel wins (below it the op
# is launch-bound; above it the two-stream two-launch form amortizes better).
_FUSE_MAX_ROWS = 256
# Block size / pack factor for the generic fallback.
_GEN_BLOCK = 1024


@triton.jit
def _gather_fused(
    kv,
    loce,
    out,
    total,
    RW: tl.constexpr,
    BLK: tl.constexpr,
    MASKED: tl.constexpr,
):
    """Whole row (NoPE + RoPE) in one launch into a single packed."""
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    if MASKED:
        m = off < total
        row = off // RW
        col = off - row * RW
        s = tl.load(loce + row, mask=m, other=0).to(tl.int32)
        tl.store(
            out + off, tl.load(kv + s * RW + col, mask=m, other=0), mask=m
        )
    else:
        row = off // RW
        col = off - row * RW
        s = tl.load(loce + row).to(tl.int32)
        tl.store(out + off, tl.load(kv + s * RW + col))


@triton.jit
def _gather_packed(
    kv,
    loce,
    out,
    total,
    HW: tl.constexpr,
    RW: tl.constexpr,
    COFF: tl.constexpr,
    BLK: tl.constexpr,
    MASKED: tl.constexpr,
):
    """One half of the split as a flat 1-D sweep over packed 64-bit."""
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    if MASKED:
        m = off < total
        row = off // HW
        col = off - row * HW
        s = tl.load(loce + row, mask=m, other=0).to(tl.int32)
        tl.store(
            out + off,
            tl.load(kv + s * RW + COFF + col, mask=m, other=0),
            mask=m,
        )
    else:
        row = off // HW
        col = off - row * HW
        s = tl.load(loce + row).to(tl.int32)
        tl.store(out + off, tl.load(kv + s * RW + COFF + col))


@triton.jit
def _gather_flat(kv, loce, out, total, HW, RW, COFF, BLK: tl.constexpr):
    """Generic fallback: one half, element-wide, 1-D, any."""
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = off < total
    row = off // HW
    col = off - row * HW
    s = tl.load(loce + row, mask=m, other=0).to(tl.int32)
    tl.store(
        out + off, tl.load(kv + s * RW + COFF + col, mask=m, other=0), mask=m
    )


def _pick_block(half_units, rope_units):
    """Largest candidate block dividing both halves' lane counts, else None.

    Returning a block means the launch is exact and the tail mask can be
    dropped; ``None`` means the caller falls back to a masked launch.
    """
    for blk in _PACK_BLOCKS:
        if half_units % blk == 0 and rope_units % blk == 0:
            return blk
    return None


def get_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    """Read ``kv_buffer[loc[i]]`` and split it into its NoPE / RoPE halves."""
    nope_dim = cache_k_nope.shape[-1]
    row_dim = kv_buffer.shape[-1]
    rope_dim = row_dim - nope_dim
    n_rows = loc.numel()

    nope = torch.empty(
        (n_rows, nope_dim),
        dtype=cache_k_nope.dtype,
        device=cache_k_nope.device,
    )
    rope = torch.empty(
        (n_rows, rope_dim),
        dtype=cache_k_rope.dtype,
        device=cache_k_rope.device,
    )
    if n_rows == 0:
        return nope, rope

    loc_l = loc.long().reshape(-1).contiguous()

    pack = 4  # bf16/fp16 elements per uint64 lane
    kv_ok = (
        kv_buffer.is_contiguous()
        and kv_buffer.dim() == 2
        and kv_buffer.dtype == cache_k_nope.dtype
        and cache_k_nope.dtype == cache_k_rope.dtype
        and cache_k_nope.dtype.itemsize == 2
        and nope.dim() == 2
        and rope.dim() == 2
        and nope_dim % pack == 0
        and rope_dim % pack == 0
        and n_rows * row_dim < 2**31
    )
    if kv_ok:
        kvp = kv_buffer.view(torch.uint64).reshape(-1)
        nope_p = nope.view(torch.uint64).reshape(-1)
        rope_p = rope.view(torch.uint64).reshape(-1)
        hw_n = nope_dim // pack  # NoPE half width, packed lanes
        hw_r = rope_dim // pack  # RoPE half width, packed lanes
        rw = row_dim // pack  # full row width, packed lanes
        tn = n_rows * hw_n  # total NoPE lanes
        tr = n_rows * hw_r  # total RoPE lanes
        tall = n_rows * rw  # total whole-row lanes

        if n_rows <= _FUSE_MAX_ROWS:
            # Fused single launch over whole rows into one packed buffer;
            # the NoPE / RoPE halves are plain slices of it.
            blk = None
            for b in _PACK_BLOCKS:
                if tall % b == 0:
                    blk = b
                    break
            full = torch.empty(
                (n_rows, row_dim),
                dtype=cache_k_nope.dtype,
                device=cache_k_nope.device,
            )
            full_p = full.view(torch.uint64).reshape(-1)
            if blk is not None:
                _gather_fused[(tall // blk,)](
                    kvp,
                    loc_l,
                    full_p,
                    tall,
                    RW=rw,
                    BLK=blk,
                    MASKED=False,
                    num_warps=4,
                )
            else:
                _gather_fused[(triton.cdiv(tall, 256),)](
                    kvp,
                    loc_l,
                    full_p,
                    tall,
                    RW=rw,
                    BLK=256,
                    MASKED=True,
                    num_warps=4,
                )
            return full[:, :nope_dim], full[:, nope_dim:]

        # Two launches, one per half (amortizes better at scale).
        blk = _pick_block(tn, tr)
        if blk is not None:
            _gather_packed[(tn // blk,)](
                kvp,
                loc_l,
                nope_p,
                tn,
                HW=hw_n,
                RW=rw,
                COFF=0,
                BLK=blk,
                MASKED=False,
                num_warps=4,
            )
            _gather_packed[(tr // blk,)](
                kvp,
                loc_l,
                rope_p,
                tr,
                HW=hw_r,
                RW=rw,
                COFF=hw_n,
                BLK=blk,
                MASKED=False,
                num_warps=4,
            )
        else:
            _gather_packed[(triton.cdiv(tn, 256),)](
                kvp,
                loc_l,
                nope_p,
                tn,
                HW=hw_n,
                RW=rw,
                COFF=0,
                BLK=256,
                MASKED=True,
                num_warps=4,
            )
            _gather_packed[(triton.cdiv(tr, 256),)](
                kvp,
                loc_l,
                rope_p,
                tr,
                HW=hw_r,
                RW=rw,
                COFF=hw_n,
                BLK=256,
                MASKED=True,
                num_warps=4,
            )
        return nope, rope

    # Generic fallback: element-wide, dtype- and width-agnostic, flat 1-D.
    kvf = kv_buffer.reshape(-1)
    _gather_flat[(triton.cdiv(nope.numel(), _GEN_BLOCK),)](
        kvf,
        loc_l,
        nope.reshape(-1),
        nope.numel(),
        nope_dim,
        row_dim,
        0,
        BLK=_GEN_BLOCK,
        num_warps=4,
    )
    if rope_dim > 0:
        _gather_flat[(triton.cdiv(rope.numel(), _GEN_BLOCK),)](
            kvf,
            loc_l,
            rope.reshape(-1),
            rope.numel(),
            rope_dim,
            row_dim,
            nope_dim,
            BLK=_GEN_BLOCK,
            num_warps=4,
        )
    return nope, rope


__all__ = ["get_mla_kv_buffer"]
