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

"""concat_mla_absorb_q for the Enflame GCU (gcu410, 24 SIPs).

Rewritten for the 2026-11-07 triton-gcu toolchain (job
j20261108141408_419b13c6); the generic implementation over-decomposes on
GCU and its dense tier breaks the grid.x <= 65535 limit:

- The generic dense kernel launches (num_rows+1)//2 two-row programs:
  at t=512 that is 32768 programs whose dispatch overhead dominates
  (3.91 ms vs torch.cat 54 us, 72x), and for num_rows > 131070
  (t > 1024 at 128 heads) grid.x exceeds the 65535 hard limit.
- Main path (contiguous inputs, 2/4-byte dtypes, i64-lane widths pow2,
  more than 49152 output lanes): persistent row-block kernel
  (``_concat_mla_absorb_q_rowblock_kernel``) with grid capped at the
  resident program count. Each iteration issues one flat contiguous
  load per input and one 1D store per input at computed offsets
  (``lane + (lane >> log2(AL)) * BL``), avoiding both the per-element
  div/mod of the flat kernels and the 2D-tile store form (which
  measured 1.7x slower than the flat-store form at t=512).
  Values are copied as raw i64 lanes: bitwise identical to torch.cat.
- Small tier (<= 49152 output lanes): one-shot wide-lane kernel,
  BLOCK=1024 / num_warps=2, the launch-floor winner at t=1
  (3.98 us vs torch.cat 5.13 us).
- Non-pow2 lane widths: wide-lane kernel, one-shot for small sizes and
  persistent (BLOCK=8192, grid <= 2*SIPs) for large ones.
- Other element sizes / non-divisible widths: flat kernel (generic
  behavior). Non-contiguous inputs: 1D flat persistent kernel with
  runtime strides (the generic 2D strided tile no longer compiles on
  this toolchain; see compiler_issues/sglang_concat_mla_absorb_q.md).
- Measured on gcu410, do_bench kernel mode (median, poisoned-output
  verified): t=512 3909 -> 55.6 us (70x, 97% of torch.cat 53.9);
  t=64 63.9 -> 11.8 us (torch 12.8); t=8 10.5 -> 4.7 us (torch 6.5);
  t=1 4.1 -> 4.0 us.
- Index math stays in int32 i64-lane units, same safety domain as the
  generic kernels: valid while the output fits in memory (lane offsets
  overflow 2^31 only past a ~17 GB output).
"""
import functools

import torch
import triton
import triton.language as tl

_I64 = tl.int64


@triton.jit
def _concat_mla_absorb_q_flat_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_out,
    A_LAST: tl.constexpr,
    B_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    OUT_LAST: tl.constexpr = A_LAST + B_LAST
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n_out
    row = idx // OUT_LAST
    col = idx % OUT_LAST
    in_a = col < A_LAST
    a_v = tl.load(a_ptr + row * A_LAST + col, mask=m & in_a, other=0.0)
    b_v = tl.load(
        b_ptr + row * B_LAST + (col - A_LAST), mask=m & (~in_a), other=0.0
    )
    tl.store(out_ptr + idx, tl.where(in_a, a_v, b_v), mask=m)


@triton.jit
def _concat_mla_absorb_q_wide_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_out,
    A_LAST: tl.constexpr,
    B_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pa = a_ptr.to(tl.pointer_type(_I64))
    pb = b_ptr.to(tl.pointer_type(_I64))
    po = out_ptr.to(tl.pointer_type(_I64))
    OUT_LAST: tl.constexpr = A_LAST + B_LAST
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n_out
    row = idx // OUT_LAST
    col = idx % OUT_LAST
    in_a = col < A_LAST
    a_v = tl.load(pa + row * A_LAST + col, mask=m & in_a, other=0)
    b_v = tl.load(
        pb + row * B_LAST + (col - A_LAST), mask=m & (~in_a), other=0
    )
    tl.store(po + idx, tl.where(in_a, a_v, b_v), mask=m)


@triton.jit
def _concat_mla_absorb_q_wide_persistent_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_out,
    A_LAST: tl.constexpr,
    B_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pa = a_ptr.to(tl.pointer_type(_I64))
    pb = b_ptr.to(tl.pointer_type(_I64))
    po = out_ptr.to(tl.pointer_type(_I64))
    OUT_LAST: tl.constexpr = A_LAST + B_LAST
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, n_out, step):
        idx = start + tl.arange(0, BLOCK)
        m = idx < n_out
        row = idx // OUT_LAST
        col = idx % OUT_LAST
        in_a = col < A_LAST
        a_v = tl.load(pa + row * A_LAST + col, mask=m & in_a, other=0)
        b_v = tl.load(
            pb + row * B_LAST + (col - A_LAST), mask=m & (~in_a), other=0
        )
        tl.store(po + idx, tl.where(in_a, a_v, b_v), mask=m)


@triton.jit
def _concat_mla_absorb_q_rowblock_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    num_rows,
    AL: tl.constexpr,
    BL: tl.constexpr,
    RB: tl.constexpr,
    LAL: tl.constexpr,
    LBL: tl.constexpr,
    EVEN: tl.constexpr,
):
    pa = a_ptr.to(tl.pointer_type(_I64))
    pb = b_ptr.to(tl.pointer_type(_I64))
    po = out_ptr.to(tl.pointer_type(_I64))
    pid = tl.program_id(0)
    step = tl.num_programs(0) * RB
    lane_a = tl.arange(0, RB * AL)
    lane_b = tl.arange(0, RB * BL)
    for r0 in range(pid * RB, num_rows, step):
        if EVEN:
            va = tl.load(pa + r0 * AL + lane_a)
            vb = tl.load(pb + r0 * BL + lane_b)
            off_a = r0 * (AL + BL) + lane_a + (lane_a >> LAL) * BL
            off_b = r0 * (AL + BL) + AL + lane_b + (lane_b >> LBL) * AL
            tl.store(po + off_a, va)
            tl.store(po + off_b, vb)
        else:
            valid = num_rows - r0
            ma = lane_a < valid * AL
            mb = lane_b < valid * BL
            va = tl.load(pa + r0 * AL + lane_a, mask=ma, other=0)
            vb = tl.load(pb + r0 * BL + lane_b, mask=mb, other=0)
            off_a = r0 * (AL + BL) + lane_a + (lane_a >> LAL) * BL
            off_b = r0 * (AL + BL) + AL + lane_b + (lane_b >> LBL) * AL
            tl.store(po + off_a, va, mask=ma)
            tl.store(po + off_b, vb, mask=mb)


@triton.jit
def _concat_mla_absorb_q_strided_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_out,
    H,
    sa0,
    sa1,
    sa2,
    sb0,
    sb1,
    sb2,
    A_LAST: tl.constexpr,
    B_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 1D flat form (runtime strides): the generic 2D [BLOCK_R, A_LAST]
    # masked tile no longer compiles on the 2026-11-07 toolchain
    # (memref_ext.slice_deslice_start 8 vs 7; see
    # compiler_issues/sglang_concat_mla_absorb_q.md).
    OUT_LAST: tl.constexpr = A_LAST + B_LAST
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, n_out, step):
        idx = start + tl.arange(0, BLOCK)
        m = idx < n_out
        row = idx // OUT_LAST
        col = idx % OUT_LAST
        tok = row // H
        head = row % H
        in_a = col < A_LAST
        a_v = tl.load(
            a_ptr + tok * sa0 + head * sa1 + col * sa2,
            mask=m & in_a,
            other=0.0,
        )
        b_v = tl.load(
            b_ptr + tok * sb0 + head * sb1 + (col - A_LAST) * sb2,
            mask=m & (~in_a),
            other=0.0,
        )
        tl.store(out_ptr + idx, tl.where(in_a, a_v, b_v), mask=m)


@functools.lru_cache(maxsize=None)
def _num_programs():
    """Resident program count = one program per SIP core."""
    driver = triton.runtime.driver.active
    try:
        props = driver.utils.get_device_properties(driver.get_current_device())
        return max(
            1,
            props["multiprocessor_count"]
            * driver.get_current_target().warp_size,
        )
    except Exception:
        return 24


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def _launch_rowblock(a, b, out, num_rows, al, bl):
    # Keep per-op lane counts within the probed compile domain
    # (RB*AL <= 16384 i64 lanes). RB=32 with grid=2*SIPs keeps two
    # programs resident per SIP (~72 KB DSM each).
    rb = 32
    while rb > 1 and (rb * al > 16384 or rb * bl > 16384):
        rb //= 2
    grid = (min(2 * _num_programs(), triton.cdiv(num_rows, rb)),)
    _concat_mla_absorb_q_rowblock_kernel[grid](
        a,
        b,
        out,
        num_rows,
        AL=al,
        BL=bl,
        RB=rb,
        LAL=(al - 1).bit_length(),
        LBL=(bl - 1).bit_length(),
        EVEN=num_rows % rb == 0,
        num_warps=4,
    )


def concat_mla_absorb_q(a, b):
    dim0, dim1, a_last = a.shape
    b_last = b.shape[-1]
    num_rows = dim0 * dim1
    out = torch.empty(
        dim0, dim1, a_last + b_last, dtype=a.dtype, device=a.device
    )
    if out.numel() == 0:
        return out

    if a.is_contiguous() and b.is_contiguous():
        esz = a.element_size()
        if (
            esz in (2, 4)
            and a_last % (8 // esz) == 0
            and b_last % (8 // esz) == 0
        ):
            epl = 8 // esz
            al = a_last // epl
            bl = b_last // epl
            n_out = out.numel() // epl
            if n_out <= 49152:
                # Launch-floor tier: one-shot wide-lane kernel.
                block = 1024
                grid = (triton.cdiv(n_out, block),)
                _concat_mla_absorb_q_wide_kernel[grid](
                    a,
                    b,
                    out,
                    n_out,
                    A_LAST=al,
                    B_LAST=bl,
                    BLOCK=block,
                    num_warps=2,
                )
            elif _is_pow2(al) and _is_pow2(bl):
                _launch_rowblock(a, b, out, num_rows, al, bl)
            else:
                # Non-pow2 lane widths: persistent wide-lane kernel.
                block = 8192
                grid = (min(2 * _num_programs(), triton.cdiv(n_out, block)),)
                _concat_mla_absorb_q_wide_persistent_kernel[grid](
                    a,
                    b,
                    out,
                    n_out,
                    A_LAST=al,
                    B_LAST=bl,
                    BLOCK=block,
                    num_warps=4,
                )
        else:
            n_out = out.numel()
            if n_out >= (1 << 28):
                block, num_warps = 2048, 8
            elif n_out >= (1 << 19):
                block, num_warps = 4096, 8
            else:
                block, num_warps = 1024, 4
            grid = ((n_out + block - 1) // block,)
            _concat_mla_absorb_q_flat_kernel[grid](
                a,
                b,
                out,
                n_out,
                A_LAST=a_last,
                B_LAST=b_last,
                BLOCK=block,
                num_warps=num_warps,
            )
    else:
        n_out = out.numel()
        block = 1024 if n_out <= 49152 else 2048
        grid = (min(2 * _num_programs(), (n_out + block - 1) // block),)
        _concat_mla_absorb_q_strided_kernel[grid](
            a,
            b,
            out,
            n_out,
            dim1,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            A_LAST=a_last,
            B_LAST=b_last,
            BLOCK=block,
            num_warps=4,
        )
    return out


__all__ = ["concat_mla_absorb_q"]
