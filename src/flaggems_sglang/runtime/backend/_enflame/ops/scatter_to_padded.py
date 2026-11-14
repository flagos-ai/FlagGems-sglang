# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import torch
import triton
import triton.language as tl

# GCU launch limit: grid.x must stay within 65535.
_GRID_X_MAX = 65535
# One program per SP on the 48-SP part: a per-row program costs ~55 ns of
# machine dispatch time (measured), so row-block programs persist and walk
# blocks with a grid stride instead.
_PERSISTENT_GRID = 48
# Row-batch kernels keep U * BLOCK <= 16384 live lanes, matching the
# on-chip staging budget used by the other GCU row-gather ops.
_MAX_ROW_BLOCK = 4096


@triton.jit
def _scatter(
    X,
    I,
    Y,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    SI: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    PROGRAMS: tl.constexpr,
    STEPS: tl.constexpr,
):
    # Generic per-row fallback (any strides, N == 0, wide rows).  GCU
    # grid.x hardware limit is 65535, but `rows` can exceed it
    # (batch=64, seqlen=1024 -> 65536).  Cap the launch at PROGRAMS
    # programs and let each one walk `STEPS` rows with a stride of
    # PROGRAMS, so every row is still covered by exactly one program.
    pid = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, B)
    for step in range(STEPS):
        row = pid + step * PROGRAMS
        if row < R:
            idx = tl.load(I + row * SI).to(tl.int64)
            value = tl.full((B,), 0, Y.dtype.element_ty)
            if N > 0:
                if idx >= 0:
                    if S2 == 1 and S1 == D:
                        addr = idx * S0 + cols
                    else:
                        addr = idx * S0 + cols // D * S1 + cols % D * S2
                    if B == H * D:
                        value = tl.load(X + addr, cache_modifier=".cg")
                    else:
                        value = tl.load(
                            X + addr, mask=cols < H * D, other=0, cache_modifier=".cg"
                        )
            if B == H * D:
                tl.store(Y + row * (H * D) + cols, value, cache_modifier=".cs")
            else:
                tl.store(
                    Y + row * (H * D) + cols,
                    value,
                    mask=cols < H * D,
                    cache_modifier=".cs",
                )


@triton.jit
def _scatter_rows_u8(
    X,
    I,
    Y,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    SI: tl.constexpr,
    B: tl.constexpr,
):
    # Fast path only (contiguous out_unpad rows and N > 0); the generic
    # per-row kernel covers every other layout.  Persistent grid-stride
    # over 8-row blocks with a textually unrolled three-segment body
    # (idx scalars -> row loads -> stores): the GCU compiler does not
    # reorder interleaved per-row load/store chains, so the batching
    # must be spelled out (fused_pack_qkv / get_mla_kv_buffer
    # precedent).  Padding rows (idx < 0) mask their load off entirely:
    # the previous version clamped them to row 0 and read it just to
    # discard the value, which wasted L2/DRAM traffic on padding-heavy
    # shapes (58% padding in the large bench shape, measured -57%);
    # other=0 leaves the value already-zero so the store needs no
    # where-fix and the clamped row is never dereferenced.  Tail rows
    # clamp to R-1 and re-write the same value, so partial blocks need
    # no masks or branches.
    cols = tl.arange(0, B)
    mask = cols < H * D
    for blk in range(tl.program_id(0), (R + 7) // 8, tl.num_programs(0)):
        base = blk.to(tl.int64) * 8
        c0 = tl.minimum(base + 0, R - 1)
        i0 = tl.load(I + c0 * SI)
        s0 = tl.maximum(i0, 0).to(tl.int64)
        c1 = tl.minimum(base + 1, R - 1)
        i1 = tl.load(I + c1 * SI)
        s1 = tl.maximum(i1, 0).to(tl.int64)
        c2 = tl.minimum(base + 2, R - 1)
        i2 = tl.load(I + c2 * SI)
        s2 = tl.maximum(i2, 0).to(tl.int64)
        c3 = tl.minimum(base + 3, R - 1)
        i3 = tl.load(I + c3 * SI)
        s3 = tl.maximum(i3, 0).to(tl.int64)
        c4 = tl.minimum(base + 4, R - 1)
        i4 = tl.load(I + c4 * SI)
        s4 = tl.maximum(i4, 0).to(tl.int64)
        c5 = tl.minimum(base + 5, R - 1)
        i5 = tl.load(I + c5 * SI)
        s5 = tl.maximum(i5, 0).to(tl.int64)
        c6 = tl.minimum(base + 6, R - 1)
        i6 = tl.load(I + c6 * SI)
        s6 = tl.maximum(i6, 0).to(tl.int64)
        c7 = tl.minimum(base + 7, R - 1)
        i7 = tl.load(I + c7 * SI)
        s7 = tl.maximum(i7, 0).to(tl.int64)
        v0 = tl.load(X + s0 * S0 + cols, mask=(i0 >= 0) & mask, other=0)
        v1 = tl.load(X + s1 * S0 + cols, mask=(i1 >= 0) & mask, other=0)
        v2 = tl.load(X + s2 * S0 + cols, mask=(i2 >= 0) & mask, other=0)
        v3 = tl.load(X + s3 * S0 + cols, mask=(i3 >= 0) & mask, other=0)
        v4 = tl.load(X + s4 * S0 + cols, mask=(i4 >= 0) & mask, other=0)
        v5 = tl.load(X + s5 * S0 + cols, mask=(i5 >= 0) & mask, other=0)
        v6 = tl.load(X + s6 * S0 + cols, mask=(i6 >= 0) & mask, other=0)
        v7 = tl.load(X + s7 * S0 + cols, mask=(i7 >= 0) & mask, other=0)
        tl.store(Y + c0 * (H * D) + cols, v0, mask=mask)
        tl.store(Y + c1 * (H * D) + cols, v1, mask=mask)
        tl.store(Y + c2 * (H * D) + cols, v2, mask=mask)
        tl.store(Y + c3 * (H * D) + cols, v3, mask=mask)
        tl.store(Y + c4 * (H * D) + cols, v4, mask=mask)
        tl.store(Y + c5 * (H * D) + cols, v5, mask=mask)
        tl.store(Y + c6 * (H * D) + cols, v6, mask=mask)
        tl.store(Y + c7 * (H * D) + cols, v7, mask=mask)


@triton.jit
def _scatter_rows_u4(
    X,
    I,
    Y,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    SI: tl.constexpr,
    B: tl.constexpr,
):
    # See _scatter_rows_u8: 4-row blocks for wide rows (BLOCK in
    # (2048, 4096], same 16384 live-lane budget), padding rows skip
    # their load (see _scatter_rows_u8).
    cols = tl.arange(0, B)
    mask = cols < H * D
    for blk in range(tl.program_id(0), (R + 3) // 4, tl.num_programs(0)):
        base = blk.to(tl.int64) * 4
        c0 = tl.minimum(base + 0, R - 1)
        i0 = tl.load(I + c0 * SI)
        s0 = tl.maximum(i0, 0).to(tl.int64)
        c1 = tl.minimum(base + 1, R - 1)
        i1 = tl.load(I + c1 * SI)
        s1 = tl.maximum(i1, 0).to(tl.int64)
        c2 = tl.minimum(base + 2, R - 1)
        i2 = tl.load(I + c2 * SI)
        s2 = tl.maximum(i2, 0).to(tl.int64)
        c3 = tl.minimum(base + 3, R - 1)
        i3 = tl.load(I + c3 * SI)
        s3 = tl.maximum(i3, 0).to(tl.int64)
        v0 = tl.load(X + s0 * S0 + cols, mask=(i0 >= 0) & mask, other=0)
        v1 = tl.load(X + s1 * S0 + cols, mask=(i1 >= 0) & mask, other=0)
        v2 = tl.load(X + s2 * S0 + cols, mask=(i2 >= 0) & mask, other=0)
        v3 = tl.load(X + s3 * S0 + cols, mask=(i3 >= 0) & mask, other=0)
        tl.store(Y + c0 * (H * D) + cols, v0, mask=mask)
        tl.store(Y + c1 * (H * D) + cols, v1, mask=mask)
        tl.store(Y + c2 * (H * D) + cols, v2, mask=mask)
        tl.store(Y + c3 * (H * D) + cols, v3, mask=mask)


@triton.jit
def _scatter_rows_u16(
    X,
    I,
    Y,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    SI: tl.constexpr,
    B: tl.constexpr,
):
    # See _scatter_rows_u8: 16-row blocks for narrow rows (BLOCK <= 1024,
    # same 16384 live-lane budget), padding rows skip their load (see
    # _scatter_rows_u8).
    cols = tl.arange(0, B)
    mask = cols < H * D
    for blk in range(tl.program_id(0), (R + 15) // 16, tl.num_programs(0)):
        base = blk.to(tl.int64) * 16
        c0 = tl.minimum(base + 0, R - 1)
        i0 = tl.load(I + c0 * SI)
        s0 = tl.maximum(i0, 0).to(tl.int64)
        c1 = tl.minimum(base + 1, R - 1)
        i1 = tl.load(I + c1 * SI)
        s1 = tl.maximum(i1, 0).to(tl.int64)
        c2 = tl.minimum(base + 2, R - 1)
        i2 = tl.load(I + c2 * SI)
        s2 = tl.maximum(i2, 0).to(tl.int64)
        c3 = tl.minimum(base + 3, R - 1)
        i3 = tl.load(I + c3 * SI)
        s3 = tl.maximum(i3, 0).to(tl.int64)
        c4 = tl.minimum(base + 4, R - 1)
        i4 = tl.load(I + c4 * SI)
        s4 = tl.maximum(i4, 0).to(tl.int64)
        c5 = tl.minimum(base + 5, R - 1)
        i5 = tl.load(I + c5 * SI)
        s5 = tl.maximum(i5, 0).to(tl.int64)
        c6 = tl.minimum(base + 6, R - 1)
        i6 = tl.load(I + c6 * SI)
        s6 = tl.maximum(i6, 0).to(tl.int64)
        c7 = tl.minimum(base + 7, R - 1)
        i7 = tl.load(I + c7 * SI)
        s7 = tl.maximum(i7, 0).to(tl.int64)
        c8 = tl.minimum(base + 8, R - 1)
        i8 = tl.load(I + c8 * SI)
        s8 = tl.maximum(i8, 0).to(tl.int64)
        c9 = tl.minimum(base + 9, R - 1)
        i9 = tl.load(I + c9 * SI)
        s9 = tl.maximum(i9, 0).to(tl.int64)
        c10 = tl.minimum(base + 10, R - 1)
        i10 = tl.load(I + c10 * SI)
        s10 = tl.maximum(i10, 0).to(tl.int64)
        c11 = tl.minimum(base + 11, R - 1)
        i11 = tl.load(I + c11 * SI)
        s11 = tl.maximum(i11, 0).to(tl.int64)
        c12 = tl.minimum(base + 12, R - 1)
        i12 = tl.load(I + c12 * SI)
        s12 = tl.maximum(i12, 0).to(tl.int64)
        c13 = tl.minimum(base + 13, R - 1)
        i13 = tl.load(I + c13 * SI)
        s13 = tl.maximum(i13, 0).to(tl.int64)
        c14 = tl.minimum(base + 14, R - 1)
        i14 = tl.load(I + c14 * SI)
        s14 = tl.maximum(i14, 0).to(tl.int64)
        c15 = tl.minimum(base + 15, R - 1)
        i15 = tl.load(I + c15 * SI)
        s15 = tl.maximum(i15, 0).to(tl.int64)
        v0 = tl.load(X + s0 * S0 + cols, mask=(i0 >= 0) & mask, other=0)
        v1 = tl.load(X + s1 * S0 + cols, mask=(i1 >= 0) & mask, other=0)
        v2 = tl.load(X + s2 * S0 + cols, mask=(i2 >= 0) & mask, other=0)
        v3 = tl.load(X + s3 * S0 + cols, mask=(i3 >= 0) & mask, other=0)
        v4 = tl.load(X + s4 * S0 + cols, mask=(i4 >= 0) & mask, other=0)
        v5 = tl.load(X + s5 * S0 + cols, mask=(i5 >= 0) & mask, other=0)
        v6 = tl.load(X + s6 * S0 + cols, mask=(i6 >= 0) & mask, other=0)
        v7 = tl.load(X + s7 * S0 + cols, mask=(i7 >= 0) & mask, other=0)
        v8 = tl.load(X + s8 * S0 + cols, mask=(i8 >= 0) & mask, other=0)
        v9 = tl.load(X + s9 * S0 + cols, mask=(i9 >= 0) & mask, other=0)
        v10 = tl.load(X + s10 * S0 + cols, mask=(i10 >= 0) & mask, other=0)
        v11 = tl.load(X + s11 * S0 + cols, mask=(i11 >= 0) & mask, other=0)
        v12 = tl.load(X + s12 * S0 + cols, mask=(i12 >= 0) & mask, other=0)
        v13 = tl.load(X + s13 * S0 + cols, mask=(i13 >= 0) & mask, other=0)
        v14 = tl.load(X + s14 * S0 + cols, mask=(i14 >= 0) & mask, other=0)
        v15 = tl.load(X + s15 * S0 + cols, mask=(i15 >= 0) & mask, other=0)
        tl.store(Y + c0 * (H * D) + cols, v0, mask=mask)
        tl.store(Y + c1 * (H * D) + cols, v1, mask=mask)
        tl.store(Y + c2 * (H * D) + cols, v2, mask=mask)
        tl.store(Y + c3 * (H * D) + cols, v3, mask=mask)
        tl.store(Y + c4 * (H * D) + cols, v4, mask=mask)
        tl.store(Y + c5 * (H * D) + cols, v5, mask=mask)
        tl.store(Y + c6 * (H * D) + cols, v6, mask=mask)
        tl.store(Y + c7 * (H * D) + cols, v7, mask=mask)
        tl.store(Y + c8 * (H * D) + cols, v8, mask=mask)
        tl.store(Y + c9 * (H * D) + cols, v9, mask=mask)
        tl.store(Y + c10 * (H * D) + cols, v10, mask=mask)
        tl.store(Y + c11 * (H * D) + cols, v11, mask=mask)
        tl.store(Y + c12 * (H * D) + cols, v12, mask=mask)
        tl.store(Y + c13 * (H * D) + cols, v13, mask=mask)
        tl.store(Y + c14 * (H * D) + cols, v14, mask=mask)
        tl.store(Y + c15 * (H * D) + cols, v15, mask=mask)


def scatter_to_padded(out_unpad, inv_indices, batch_size, seqlen):
    _, h, d = out_unpad.shape
    out = torch.empty(
        (batch_size, seqlen, h, d),
        dtype=out_unpad.dtype,
        device=out_unpad.device,
    )
    rows = batch_size * seqlen
    if rows * h * d:
        # torch_gcu stores int64 tensors as int32: a kernel-side int64
        # load would read interleaved garbage.  Row ids are < rows, well
        # inside int32, so normalize before the launch.
        if inv_indices.dtype == torch.int64:
            inv_indices = inv_indices.to(torch.int32)
        s0, s1, s2 = out_unpad.stride()
        si = inv_indices.stride(0)
        n = out_unpad.shape[0]
        block = triton.next_power_of_2(h * d)
        if n > 0 and s2 == 1 and s1 == d and block <= _MAX_ROW_BLOCK:
            if block <= 1024:
                rows_per_block, kernel, warps = 16, _scatter_rows_u16, 2
            elif block <= 2048:
                rows_per_block, kernel, warps = 8, _scatter_rows_u8, 4
            else:
                rows_per_block, kernel, warps = 4, _scatter_rows_u4, 4
            blocks = triton.cdiv(rows, rows_per_block)
            grid = blocks if blocks <= _PERSISTENT_GRID else _PERSISTENT_GRID
            kernel[(grid,)](
                out_unpad,
                inv_indices,
                out,
                rows,
                h,
                d,
                s0,
                si,
                block,
                num_warps=warps,
            )
        else:
            programs = min(rows, _GRID_X_MAX)
            _scatter[(programs,)](
                out_unpad,
                inv_indices,
                out,
                rows,
                h,
                d,
                *out_unpad.stride(),
                si,
                n,
                block,
                PROGRAMS=programs,
                STEPS=triton.cdiv(rows, programs),
                num_warps=8 if rows * h * d >= 4194304 else 4,
            )
    return out


__all__ = ["scatter_to_padded"]
