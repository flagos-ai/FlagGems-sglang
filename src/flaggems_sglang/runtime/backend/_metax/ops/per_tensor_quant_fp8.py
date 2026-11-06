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

"""Standalone MetaX Triton implementation of per_tensor_quant_fp8."""

import torch
import triton
import triton.language as tl

# fp8_e4m3fn max, hoisted so the hot path does not touch torch.finfo per call.
_FP8_MAX = 448.0

# fused small path: one program covers the whole tensor.
_FUSED_MAX = 4096
# absmax sweep: block per element-count band.
_ABSMAX_BLOCK = 1024
_ABSMAX_WIDE_BLOCK = 2048
_ABSMAX_WIDE_MIN = 1 << 20  # >= 1M elements: wider block wins
_NUM_WARPS = 1


@triton.jit
def _fp32_to_e4m3_bits(x):
    """fp32 -> float8_e4m3fn codeword (as uint8), saturating like ``clamp``.

    e4m3fn: 4 exponent bits (bias 7), 3 mantissa bits, max 448, min normal
    2**-6, min subnormal 2**-9.
    """
    b = x.to(tl.int32, bitcast=True)
    sign = b & -2147483648
    a = b & 0x7FFFFFFF
    # Saturate: anything of magnitude >= 448 collapses onto 448.0 (finite).
    sat = tl.minimum(a, 0x43E00000)
    # Round the 3 mantissa bits to nearest, ties to even (carry propagates
    # into the exponent field for free).
    r = sat + 0x7FFFF + ((sat >> 20) & 1)
    normal = ((r >> 20) & 0x7FF) - 960  # (E - 120) * 8 + mantissa
    # Subnormal / underflow: scale by 2**9 and use the f32 add's RNE on the
    # integer grid.  Only the values selected below are meaningful.
    f = sat.to(tl.float32, bitcast=True)
    sub = (f * 512.0 + 12582912.0).to(tl.int32, bitcast=True) - 0x4B400000
    field = tl.where(a < 0x3C800000, sub, normal)  # 2**-6 splits the ranges
    field = tl.where(a > 0x7F800000, 127, field)  # NaN -> quiet NaN
    return (field | (sign >> 24)).to(tl.uint8)


@triton.jit
def _quant(x, IS_F32: tl.constexpr, OUT_DTYPE: tl.constexpr):
    """Saturating fp8 cast of a fp32 block already divided by the scale."""
    return _fp32_to_e4m3_bits(x).to(OUT_DTYPE, bitcast=True)


@triton.jit
def _fused_small_kernel(
    x_ptr,
    s_ptr,
    q_ptr,
    n_elements,
    IS_F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.max(tl.abs(x))
    m = tl.where(tl.max((x != x).to(tl.int32), 0) != 0, float("nan"), m)
    s = tl.div_rn(m, 448.0)
    tl.store(s_ptr, s)
    out_ty: tl.constexpr = q_ptr.dtype.element_ty
    tl.store(q_ptr + offs, _quant(tl.div_rn(x, s), IS_F32, out_ty), mask=mask)


@triton.jit
def _quant_flat_kernel(
    x_ptr,
    s_ptr,
    out_ptr,
    n_elements,
    IS_F32: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    out_ty: tl.constexpr = out_ptr.dtype.element_ty
    s = tl.load(s_ptr)
    if EVEN_N:
        x = tl.load(x_ptr + offs).to(tl.float32)
        tl.store(out_ptr + offs, _quant(tl.div_rn(x, s), IS_F32, out_ty))
    else:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + offs, _quant(tl.div_rn(x, s), IS_F32, out_ty), mask=mask
        )


@triton.jit
def _quant_dual_kernel(
    x_ptr,
    s_ptr,
    out_ptr,
    IS_F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Two independent blocks per program: ``n_elements % (2 * BLOCK) == 0``.

    The second load is issued before the first store, so the two blocks'
    memory latencies overlap instead of serializing.
    """
    pid = tl.program_id(0)
    offs1 = pid * (BLOCK * 2) + tl.arange(0, BLOCK)
    offs2 = offs1 + BLOCK
    out_ty: tl.constexpr = out_ptr.dtype.element_ty
    s = tl.load(s_ptr)
    x1 = tl.load(x_ptr + offs1).to(tl.float32)
    x2 = tl.load(x_ptr + offs2).to(tl.float32)
    tl.store(out_ptr + offs1, _quant(tl.div_rn(x1, s), IS_F32, out_ty))
    tl.store(out_ptr + offs2, _quant(tl.div_rn(x2, s), IS_F32, out_ty))


@triton.jit
def _quant_persist_kernel(
    x_ptr,
    s_ptr,
    out_ptr,
    IS_F32: tl.constexpr,
    N_TILES: tl.constexpr,
    N_PROG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Grid-stride persistent form: ``N_PROG`` programs sweep."""
    pid = tl.program_id(0)
    out_ty: tl.constexpr = out_ptr.dtype.element_ty
    s = tl.load(s_ptr)
    for tile in tl.range(pid, N_TILES, N_PROG):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs).to(tl.float32)
        tl.store(out_ptr + offs, _quant(tl.div_rn(x, s), IS_F32, out_ty))


def _launch_fused_small(input, output_q, output_s, n, is_f32):
    _fused_small_kernel[(1,)](
        input,
        output_s,
        output_q,
        n,
        IS_F32=is_f32,
        BLOCK=triton.next_power_of_2(n),
        num_warps=4,
        num_stages=1,
    )


def _launch_absmax(input, output_s, n):
    _reduce_scale(input, output_s)


def _launch_quant(input, output_q, output_s, n, is_f32):
    if n <= 1 << 20:
        # 64K..1M: flat one-tile-per-program, 1024-wide.
        # 4 warps up to 512K, 2 above.
        num_warps = 4 if n <= 1 << 19 else 2
        _quant_flat_kernel[(triton.cdiv(n, 1024),)](
            input,
            output_s,
            output_q,
            n,
            IS_F32=is_f32,
            EVEN_N=n % 1024 == 0,
            BLOCK=1024,
            num_warps=num_warps,
            num_stages=1,
        )
        return
    if n <= 1 << 21:
        # 1M..2M: flat 1024-wide, 2 warps.
        _quant_flat_kernel[(triton.cdiv(n, 1024),)](
            input,
            output_s,
            output_q,
            n,
            IS_F32=is_f32,
            EVEN_N=n % 1024 == 0,
            BLOCK=1024,
            num_warps=2,
            num_stages=1,
        )
        return
    if n <= 1 << 22:
        # 2M..4M: dual-block 2048-wide when the size divides, else flat.
        block = 2048
        if n % (block * 2) == 0:
            _quant_dual_kernel[(n // (block * 2),)](
                input,
                output_s,
                output_q,
                IS_F32=is_f32,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )
            return
    else:
        # > 4M: grid-stride persistent 4096-wide tiles, 8x SM programs.
        block = 4096
        if n % block == 0:
            tiles = n // block
            n_prog = min(tiles, 104 * 8)
            _quant_persist_kernel[(n_prog,)](
                input,
                output_s,
                output_q,
                IS_F32=is_f32,
                N_TILES=tiles,
                N_PROG=n_prog,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )
            return
    # Fallback for odd sizes in the wide bands (and 2M..4M non-divisible).
    block = 2048
    _quant_flat_kernel[(triton.cdiv(n, block),)](
        input,
        output_s,
        output_q,
        n,
        IS_F32=is_f32,
        EVEN_N=n % block == 0,
        BLOCK=block,
        num_warps=2,
        num_stages=1,
    )


def per_tensor_quant_fp8(input, output_q, output_s, is_static):
    if input.numel() == 0:
        return output_q, output_s
    input = _dense(input)
    q = (
        output_q
        if output_q.is_contiguous()
        else torch.empty(
            output_q.shape, dtype=output_q.dtype, device=input.device
        )
    )
    n = input.numel()
    is_f32 = input.dtype == torch.float32
    if not is_static and n <= _FUSED_MAX:
        _launch_fused_small(input, q, output_s, n, is_f32)
    else:
        if not is_static:
            _launch_absmax(input, output_s, n)
        _launch_quant(input, q, output_s, n, is_f32)
    if q is not output_q:
        _layout_copy(q, output_q)
    return output_q, output_s


@triton.jit
def _layout_copy_kernel(
    src,
    dst,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    SS: tl.constexpr,
    DS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    rem = flat
    si = tl.full((BLOCK,), 0, tl.int64)
    di = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = rem % SHAPE[d]
        rem = rem // SHAPE[d]
        si += coord * SS[d]
        di += coord * DS[d]
    value = tl.load(src + si, flat < N, other=0.0)
    tl.store(dst + di, value, flat < N)


def _layout_copy(src, dst):
    if src.numel():
        _layout_copy_kernel[(triton.cdiv(src.numel(), 1024),)](
            src,
            dst,
            src.numel(),
            tuple(src.shape),
            tuple(src.stride()),
            tuple(dst.stride()),
            BLOCK=1024,
            num_warps=4,
        )
    return dst


def _dense(x):
    if x.is_contiguous():
        return x
    return _layout_copy(
        x, torch.empty(x.shape, dtype=x.dtype, device=x.device)
    )


@triton.jit
def _scale_reduce_kernel(
    x,
    out,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    FINAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    rem = flat
    offset = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (rem % SHAPE[d]) * STRIDES[d]
        rem = rem // SHAPE[d]
    xval = tl.load(x + offset, flat < N, other=0.0).to(tl.float32)
    # Propagate NaN explicitly; max reduction backends may otherwise ignore it.
    has_nan = tl.max((xval != xval).to(tl.int32), 0)
    value = tl.max(tl.abs(xval), 0)
    value = tl.where(has_nan != 0, float("nan"), value)
    if FINAL:
        value = tl.div_rn(value, 448.0)
    tl.store(out + tl.program_id(0), value)


def _reduce_scale(input, output_s, block=1024, warps=4):
    if input.numel() == 0:
        return output_s
    current = input
    for _ in range(int(input.numel()).bit_length() // 10 + 2):
        n = current.numel()
        count = triton.cdiv(n, block)
        target = (
            output_s
            if count == 1
            else torch.empty(
                (count,), dtype=torch.float32, device=input.device
            )
        )
        _scale_reduce_kernel[(count,)](
            current,
            target,
            n,
            tuple(current.shape),
            tuple(current.stride()),
            count == 1,
            BLOCK=block,
            num_warps=warps,
            num_stages=1,
            enable_fp_fusion=False,
        )
        if count == 1:
            return output_s
        current = target


__all__ = ["per_tensor_quant_fp8"]
