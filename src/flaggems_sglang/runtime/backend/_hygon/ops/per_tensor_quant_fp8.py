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

_FP8_E4M3_MAX = 448.0

_FUSED_MAX = 1 << 12

_REDUNDANT_MAX = 1 << 15

_PARTIALS_CONFIGS = (
    (1 << 20, 8192, 4),
    (1 << 24, 8192, 4),
    (1 << 62, 4096, 8),
)

_QUANT_NARROW = (1024, 8)
_QUANT_WIDE = (2048, 4)


@triton.jit
def _fp8_rtne(y, EBIAS: tl.constexpr, MB: tl.constexpr, SIMPLE: tl.constexpr):
    u = y.to(tl.int32, bitcast=True)
    sign = (u >> 31) & 1
    a = u & 0x7FFFFFFF
    e = a >> 23

    DROP: tl.constexpr = 23 - MB
    RNDB: tl.constexpr = (1 << (DROP - 1)) - 1
    rn = a + RNDB + ((a >> DROP) & 1)
    mag_norm = (rn >> DROP) - ((127 - EBIAS) << MB)

    if SIMPLE:
        mag = mag_norm
    else:
        LESS_E: tl.constexpr = 128 - EBIAS
        SUBSCALE: tl.constexpr = float(1 << (EBIAS - 1 + MB))
        ay = a.to(tl.float32, bitcast=True)
        magic = ay * SUBSCALE + 12582912.0
        mag_sub = magic.to(tl.int32, bitcast=True) & 0x3FFFFF
        mag = tl.where(e >= LESS_E, mag_norm, mag_sub)

    return (sign << 7) | mag


@triton.jit
def _quant_lane(
    y,
    inv_scale,
    FP8_MAX: tl.constexpr,
    EBIAS: tl.constexpr,
    MB: tl.constexpr,
    SIMPLE: tl.constexpr,
    CLAMP: tl.constexpr,
):
    y = y.to(tl.float32) * inv_scale
    if CLAMP:
        y = tl.minimum(tl.maximum(y, -FP8_MAX), FP8_MAX)
    return _fp8_rtne(y, EBIAS, MB, SIMPLE)


@triton.jit
def _fused_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    n,
    inv_fp8_max,
    IS_BF16: tl.constexpr,
    EVEN: tl.constexpr,
    CHUNK: tl.constexpr,
):
    x16 = tl.cast(x_ptr, tl.pointer_type(tl.int16))
    x32 = tl.cast(x_ptr, tl.pointer_type(tl.int32))
    q16 = tl.cast(q_ptr, tl.pointer_type(tl.uint16))
    nchunks = tl.cdiv(n, CHUNK)
    offs = tl.arange(0, CHUNK)
    amax = 0
    for i in range(nchunks):
        o = i * CHUNK + offs
        if EVEN:
            bits = tl.load(x16 + o, eviction_policy="evict_first") & 0x7FFF
        else:
            bits = (
                tl.load(
                    x16 + o, mask=o < n, other=0, eviction_policy="evict_first"
                )
                & 0x7FFF
            )
        amax = tl.maximum(amax, tl.max(bits, axis=0).to(tl.int32))
    max_bits = amax.to(tl.uint16)
    if IS_BF16:
        local_max = max_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    else:
        local_max = max_bits.to(tl.float16, bitcast=True).to(tl.float32)
    scale = local_max * inv_fp8_max
    tl.store(scale_ptr, scale)
    inv_scale = 1.0 / scale

    npairs = n // 2
    poffs = tl.arange(0, CHUNK // 2)
    for i in range(tl.cdiv(npairs, CHUNK // 2)):
        o = i * (CHUNK // 2) + poffs
        m = o < npairs
        if EVEN:
            w = tl.load(x32 + o)
        else:
            w = tl.load(x32 + o, mask=m, other=0)
        w = w.to(tl.uint32, bitcast=True)
        lo = (w & 0xFFFF).to(tl.uint16)
        hi = (w >> 16).to(tl.uint16)
        if IS_BF16:
            lo_f = lo.to(tl.bfloat16, bitcast=True)
            hi_f = hi.to(tl.bfloat16, bitcast=True)
        else:
            lo_f = lo.to(tl.float16, bitcast=True)
            hi_f = hi.to(tl.float16, bitcast=True)
        b_lo = _quant_lane(lo_f, inv_scale, 448.0, 7, 3, True, False)
        b_hi = _quant_lane(hi_f, inv_scale, 448.0, 7, 3, True, False)
        pair = (b_lo | (b_hi << 8)).to(tl.uint16)
        tl.store(q16 + o, pair, mask=m, cache_modifier=".cs")


@triton.jit
def _redundant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    n,
    inv_fp8_max,
    IS_BF16: tl.constexpr,
    NEED_MASK: tl.constexpr,
    CHUNK: tl.constexpr,
    QWORDS: tl.constexpr,
):
    x64 = tl.cast(x_ptr, tl.pointer_type(tl.uint64))
    q32 = tl.cast(q_ptr, tl.pointer_type(tl.uint32))
    pid = tl.program_id(0)
    offs = tl.arange(0, CHUNK)
    amax = 0.0
    nchunks = tl.cdiv(n, CHUNK)
    for i in range(0, nchunks):
        o = i * CHUNK + offs
        if NEED_MASK:
            x = tl.load(
                x_ptr + o, mask=o < n, other=0.0, eviction_policy="evict_first"
            )
        else:
            x = tl.load(x_ptr + o, eviction_policy="evict_first")
        amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32)), axis=0))
    scale = amax * inv_fp8_max
    tl.store(scale_ptr, scale)
    inv_scale = 1.0 / scale

    nwords = n >> 2
    woff = pid * QWORDS + tl.arange(0, QWORDS)
    m = woff < nwords
    w = tl.load(x64 + woff, mask=m, other=0)
    e0 = (w & 0xFFFF).to(tl.uint16)
    e1 = ((w >> 16) & 0xFFFF).to(tl.uint16)
    e2 = ((w >> 32) & 0xFFFF).to(tl.uint16)
    e3 = ((w >> 48) & 0xFFFF).to(tl.uint16)
    if IS_BF16:
        e0 = e0.to(tl.bfloat16, bitcast=True)
        e1 = e1.to(tl.bfloat16, bitcast=True)
        e2 = e2.to(tl.bfloat16, bitcast=True)
        e3 = e3.to(tl.bfloat16, bitcast=True)
    else:
        e0 = e0.to(tl.float16, bitcast=True)
        e1 = e1.to(tl.float16, bitcast=True)
        e2 = e2.to(tl.float16, bitcast=True)
        e3 = e3.to(tl.float16, bitcast=True)
    b0 = _quant_lane(e0, inv_scale, 448.0, 7, 3, True, False)
    b1 = _quant_lane(e1, inv_scale, 448.0, 7, 3, True, False)
    b2 = _quant_lane(e2, inv_scale, 448.0, 7, 3, True, False)
    b3 = _quant_lane(e3, inv_scale, 448.0, 7, 3, True, False)
    quad = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).to(tl.uint32)
    tl.store(q32 + woff, quad, mask=m, cache_modifier=".cs")


@triton.jit
def _absmax_16b_chunk_kernel(
    x_ptr,
    part_ptr,
    n,
    inv_fp8_max,
    NEED_MASK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * CHUNK + tl.arange(0, CHUNK)
    if NEED_MASK:
        x = tl.load(
            x_ptr + offs,
            mask=offs < n,
            other=0.0,
            eviction_policy="evict_first",
        )
    else:
        x = tl.load(x_ptr + offs, eviction_policy="evict_first")
    local_max = tl.max(tl.abs(x).to(tl.float32), axis=0)
    tl.store(part_ptr + pid, local_max * inv_fp8_max)


@triton.jit
def _absmax_f32_partials_kernel(
    x_ptr,
    part_ptr,
    n,
    inv_fp8_max,
    EVEN: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    nchunks = tl.cdiv(n, CHUNK)
    offs = tl.arange(0, CHUNK)
    amax = 0.0
    for i in range(pid, nchunks, tl.num_programs(0)):
        o = i * CHUNK + offs
        if EVEN:
            x = tl.load(x_ptr + o, eviction_policy="evict_first")
        else:
            x = tl.load(
                x_ptr + o, mask=o < n, other=0.0, eviction_policy="evict_first"
            )
        amax = tl.maximum(amax, tl.max(tl.abs(x).to(tl.float32), axis=0))
    tl.store(part_ptr + pid, amax * inv_fp8_max)


@triton.jit
def _quant_partials_kernel(
    x_ptr,
    q_ptr,
    part_ptr,
    scale_ptr,
    npart,
    nwords,
    IS_BF16: tl.constexpr,
    STATIC: tl.constexpr,
    P_BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    BLOCK_W4: tl.constexpr,
):
    x64 = tl.cast(x_ptr, tl.pointer_type(tl.uint64))
    q32 = tl.cast(q_ptr, tl.pointer_type(tl.uint32))
    pid = tl.program_id(0)
    if STATIC:
        inv_scale = 1.0 / tl.load(scale_ptr)
    else:
        poffs = tl.arange(0, P_BLOCK)
        pv = tl.load(part_ptr + poffs, mask=poffs < npart, other=0.0)
        rmax = tl.max(pv, axis=0)
        tl.store(scale_ptr, rmax)
        inv_scale = 1.0 / rmax

    woff = pid * BLOCK_W4 + tl.arange(0, BLOCK_W4)
    if NEED_MASK:
        w = tl.load(
            x64 + woff,
            mask=woff < nwords,
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
    else:
        w = tl.load(
            x64 + woff, cache_modifier=".cg", eviction_policy="evict_first"
        )
    w = w.to(tl.uint64, bitcast=True)
    e0 = (w & 0xFFFF).to(tl.uint16)
    e1 = ((w >> 16) & 0xFFFF).to(tl.uint16)
    e2 = ((w >> 32) & 0xFFFF).to(tl.uint16)
    e3 = ((w >> 48) & 0xFFFF).to(tl.uint16)
    if IS_BF16:
        e0 = e0.to(tl.bfloat16, bitcast=True)
        e1 = e1.to(tl.bfloat16, bitcast=True)
        e2 = e2.to(tl.bfloat16, bitcast=True)
        e3 = e3.to(tl.bfloat16, bitcast=True)
    else:
        e0 = e0.to(tl.float16, bitcast=True)
        e1 = e1.to(tl.float16, bitcast=True)
        e2 = e2.to(tl.float16, bitcast=True)
        e3 = e3.to(tl.float16, bitcast=True)

    if STATIC:
        b0 = _quant_lane(e0, inv_scale, 448.0, 7, 3, False, True)
        b1 = _quant_lane(e1, inv_scale, 448.0, 7, 3, False, True)
        b2 = _quant_lane(e2, inv_scale, 448.0, 7, 3, False, True)
        b3 = _quant_lane(e3, inv_scale, 448.0, 7, 3, False, True)
    else:
        b0 = _quant_lane(e0, inv_scale, 448.0, 7, 3, True, False)
        b1 = _quant_lane(e1, inv_scale, 448.0, 7, 3, True, False)
        b2 = _quant_lane(e2, inv_scale, 448.0, 7, 3, True, False)
        b3 = _quant_lane(e3, inv_scale, 448.0, 7, 3, True, False)

    quad = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).to(tl.uint32)
    if NEED_MASK:
        tl.store(q32 + woff, quad, mask=woff < nwords, cache_modifier=".cs")
    else:
        tl.store(q32 + woff, quad, cache_modifier=".cs")


@triton.jit
def _quant_generic_kernel(
    x_ptr,
    q_ptr,
    part_ptr,
    scale_ptr,
    npart,
    n_elements,
    STATIC: tl.constexpr,
    P_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q8 = tl.cast(q_ptr, tl.pointer_type(tl.uint8))
    pid = tl.program_id(0)
    if STATIC:
        inv_scale = 1.0 / tl.load(scale_ptr)
    else:
        poffs = tl.arange(0, P_BLOCK)
        pv = tl.load(part_ptr + poffs, mask=poffs < npart, other=0.0)
        rmax = tl.max(pv, axis=0)
        tl.store(scale_ptr, rmax)
        inv_scale = 1.0 / rmax
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    bits = _quant_lane(x, inv_scale, 448.0, 7, 3, False, True)
    tl.store(
        q8 + offs,
        bits.to(tl.uint8),
        mask=mask,
    )


def _p2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def per_tensor_quant_fp8(input, output_q, output_s, is_static):
    n = input.numel()
    if n == 0:
        return output_q, output_s
    x = input if input.is_contiguous() else input.contiguous()
    is16 = x.dtype in (torch.float16, torch.bfloat16)

    if is_static:
        if (
            is16
            and n % 4 == 0
            and x.data_ptr() % 8 == 0
            and output_q.data_ptr() % 4 == 0
        ):
            nwords = n >> 2
            if n <= (1 << 21):
                block_w4, num_warps = 2048, 8
            else:
                block_w4, num_warps = 4096, 4
            _quant_partials_kernel[((nwords - 1) // block_w4 + 1,)](
                x,
                output_q,
                output_s,
                output_s,
                0,
                nwords,
                IS_BF16=(x.dtype == torch.bfloat16),
                STATIC=True,
                P_BLOCK=1,
                NEED_MASK=(nwords % block_w4 != 0),
                BLOCK_W4=block_w4,
                num_warps=num_warps,
            )
        else:
            _quant_generic_kernel[((n - 1) // 1024 + 1,)](
                x,
                output_q,
                output_s,
                output_s,
                0,
                n,
                STATIC=True,
                P_BLOCK=1,
                BLOCK=1024,
                num_warps=4,
            )
        return output_q, output_s

    if is16 and n <= _FUSED_MAX and n % 2 == 0:
        _fused_kernel[(1,)](
            x,
            output_q,
            output_s,
            n,
            1.0 / _FP8_E4M3_MAX,
            IS_BF16=(x.dtype == torch.bfloat16),
            EVEN=(n & 4095) == 0,
            CHUNK=4096,
            num_warps=8,
        )
        return output_q, output_s

    aligned16 = (
        is16
        and n % 4 == 0
        and x.data_ptr() % 8 == 0
        and output_q.data_ptr() % 4 == 0
    )
    if is16 and n <= _REDUNDANT_MAX and aligned16:
        nwords = n >> 2
        if n <= 4096:
            nprog, qw = 8, 2
            chunk = _p2(max(1024, n // 4))
        elif n <= 8192:
            nprog, qw, chunk = 32, 8, 4096
        elif n <= 16384:
            nprog, qw, chunk = 8, 2, 8192
        else:
            nprog, qw, chunk = 16, 4, 8192
        _redundant_kernel[(nprog,)](
            x,
            output_q,
            output_s,
            n,
            1.0 / _FP8_E4M3_MAX,
            IS_BF16=(x.dtype == torch.bfloat16),
            NEED_MASK=(n % chunk != 0),
            CHUNK=chunk,
            QWORDS=_p2((nwords + nprog - 1) // nprog),
            num_warps=qw,
        )
        return output_q, output_s

    if is16:
        for bound, chunk, warps in _PARTIALS_CONFIGS:
            if n <= bound:
                break
        npart = (n - 1) // chunk + 1
        partials = torch.empty(npart, dtype=torch.float32, device=x.device)
        _absmax_16b_chunk_kernel[(npart,)](
            x,
            partials,
            n,
            1.0 / _FP8_E4M3_MAX,
            NEED_MASK=(n % chunk != 0),
            CHUNK=chunk,
            num_warps=warps,
        )
    else:
        npart = (n - 1) // 2048 + 1
        if npart > 128:
            npart = 128
        partials = torch.empty(npart, dtype=torch.float32, device=x.device)
        _absmax_f32_partials_kernel[(npart,)](
            x,
            partials,
            n,
            1.0 / _FP8_E4M3_MAX,
            EVEN=(n & 2047) == 0,
            CHUNK=2048,
            num_warps=8,
        )

    if (
        is16
        and n % 4 == 0
        and x.data_ptr() % 8 == 0
        and output_q.data_ptr() % 4 == 0
    ):
        nwords = n >> 2
        block_w4, num_warps = _QUANT_NARROW if n <= (1 << 19) else _QUANT_WIDE
        _quant_partials_kernel[((nwords - 1) // block_w4 + 1,)](
            x,
            output_q,
            partials,
            output_s,
            npart,
            nwords,
            IS_BF16=(x.dtype == torch.bfloat16),
            STATIC=False,
            P_BLOCK=_p2(npart),
            NEED_MASK=(nwords % block_w4 != 0),
            BLOCK_W4=block_w4,
            num_warps=num_warps,
        )
    else:
        _quant_generic_kernel[((n - 1) // 1024 + 1,)](
            x,
            output_q,
            partials,
            output_s,
            npart,
            n,
            STATIC=False,
            P_BLOCK=_p2(npart),
            BLOCK=1024,
            num_warps=4,
        )
    return output_q, output_s


__all__ = ["per_tensor_quant_fp8"]
