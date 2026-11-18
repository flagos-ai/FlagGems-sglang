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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import triton
import triton.language as tl
from triton.language import bfloat16 as bf16
from triton.language import float16 as fp16
from triton.language import uint8 as u8
from triton.language import uint16 as u16
from triton.language import uint32 as u32


@triton.jit
def _encode(v, M: tl.constexpr, BIAS: tl.constexpr, MAX: tl.constexpr):
    raw = v.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    mag = tl.minimum(
        raw & 2147483647,
        tl.full((), MAX, tl.float32).to(tl.uint32, bitcast=True),
    )
    normal = (
        mag
        + tl.full(
            (), (1 << 22 - M) - 1 - (127 - BIAS << 23) & 4294967295, tl.uint32
        )
        + (mag >> 23 - M & 1)
        >> 23 - M
    )
    magic = tl.full((), 2.0 ** (24 - BIAS - M), tl.float32)
    sub = (mag.to(tl.float32, bitcast=True) + magic).to(
        tl.uint32, bitcast=True
    )
    code = tl.where(mag < 128 - BIAS << 23, sub, normal)
    return (code | sign).to(tl.uint8)


def fp8_quantize(x, scale_inv, fp8_dtype):
    if (
        torch.version.hip is not None
        and x.numel() >= 4194304
        and (x.numel() % 2 != 0)
        and (x.dtype == torch.bfloat16)
        and (scale_inv == 1.0)
        and x.is_contiguous()
    ):
        total = x.numel()
        out = torch.empty_like(x, dtype=fp8_dtype)
        if total >= 33554432 and x.data_ptr() % 16 == 0:
            _bf16_split_v648.run(
                x.data_ptr(),
                out.data_ptr(),
                total,
                fp8_dtype == torch.float8_e4m3fn,
                512,
                grid=((total + 1023) // 1024,),
                num_warps=8,
                warmup=False,
            )
        else:
            _bf16_fulltail_v650.run(
                x.data_ptr(),
                out.data_ptr(),
                total,
                fp8_dtype == torch.float8_e4m3fn,
                1024,
                grid=((total + 1023) // 1024,),
                num_warps=8,
                warmup=False,
            )
        return out
    if scale_inv == 0.0:
        out = torch.empty(x.shape, dtype=fp8_dtype, device=x.device)
        total = x.numel()
        if total:
            width = x.size(-1)
            pitch = _row_stride(x, width)
            _zero_scale_v587.run(
                x.data_ptr(),
                out.data_ptr(),
                total,
                width,
                pitch,
                math.copysign(1.0, scale_inv) < 0.0,
                x.dtype == torch.bfloat16,
                1024,
                grid=(min((total + 1023) // 1024, 4096),),
                warmup=False,
            )
        return out
    total = x.numel()
    if (
        33554432 <= total < 2147483648
        and total % 2 != 0
        and (scale_inv != 1.0)
        and x.is_contiguous()
    ):
        out = torch.empty_like(x, dtype=fp8_dtype)
        _general_scalar_v579.run(
            x.data_ptr(),
            out.data_ptr(),
            total,
            x.dtype == torch.bfloat16,
            scale_inv,
            fp8_dtype == torch.float8_e4m3fn,
            1024,
            grid=((total + 1023) // 1024,),
            num_warps=4,
            enable_fp_fusion=False,
            warmup=False,
        )
        return out
    if (
        total >= 8388608
        and total % 2 != 0
        and (x.dtype == torch.float16)
        and (scale_inv == 1.0)
        and (torch.version.hip is not None)
        and x.is_contiguous()
    ):
        out = torch.empty_like(x, dtype=fp8_dtype)
        address = x.data_ptr()
        if address % 4 == 0:
            _half_split_v494.run(
                address,
                out.data_ptr(),
                total,
                fp8_dtype == torch.float8_e4m3fn,
                512,
                grid=((total + 1023) // 1024,),
                num_warps=8,
                warmup=False,
            )
        else:
            _half_scalar_fulltail_v501.run(
                address,
                out.data_ptr(),
                total,
                fp8_dtype == torch.float8_e4m3fn,
                1024,
                grid=((total + 1023) // 1024,),
                num_warps=8,
                warmup=False,
            )
        return out
    if (
        total >= 8388608
        and total % 2 == 0
        and (x.dtype == torch.float16)
        and (scale_inv == 1.0)
        and x.is_contiguous()
        and (x.storage_offset() % 2 == 0)
        and (torch.version.hip is not None and fp8_dtype == torch.float8_e5m2)
    ):
        out = torch.empty_like(x, dtype=fp8_dtype)
        block = 1024
        _half_pair_v471.run(
            x.data_ptr(),
            out.data_ptr(),
            total // 2,
            fp8_dtype == torch.float8_e4m3fn,
            block // 2,
            grid=((total + block - 1) // block,),
            num_warps=8 if torch.version.hip is not None else 4,
            warmup=False,
        )
        return out
    out = torch.empty_like(x, dtype=fp8_dtype)
    if total:
        if x.is_contiguous():
            if (
                total >= (131072 if torch.version.hip is not None else 8388608)
                and total % 2 == 0
                and (x.dtype == torch.bfloat16)
                and (scale_inv == 1.0)
                and (x.storage_offset() % 2 == 0)
            ):
                if torch.version.hip is not None:
                    (
                        _pair_linear_kernel_bf16
                        if x.dtype == torch.bfloat16
                        else _pair_linear_kernel_fp16
                    ).run(
                        x.data_ptr(),
                        out.data_ptr(),
                        total // 2,
                        fp8_dtype == torch.float8_e4m3fn,
                        grid=((total + 1024 - 1) // 1024,),
                        warmup=False,
                        num_warps=8,
                    )
                else:
                    (
                        _pair_linear_kernel_bf16
                        if x.dtype == torch.bfloat16
                        else _pair_linear_kernel_fp16
                    ).run(
                        x.data_ptr(),
                        out.data_ptr(),
                        total // 2,
                        fp8_dtype == torch.float8_e4m3fn,
                        grid=((total + 1024 - 1) // 1024,),
                        warmup=False,
                    )
            elif scale_inv == 1.0 and fp8_dtype == torch.float8_e4m3fn:
                if total < 8388608 and total % 1024 == 0:
                    fn = (
                        _full_bf16_1024_0
                        if x.dtype == torch.bfloat16
                        else _full_fp16_1024_0
                    )
                    fn.run(
                        x.data_ptr(),
                        out.data_ptr(),
                        grid=(total // 1024,),
                        warmup=False,
                    )
                else:
                    (
                        _unit_bf16 if x.dtype == torch.bfloat16 else _unit_fp16
                    ).run(
                        x.data_ptr(),
                        out.data_ptr(),
                        total,
                        grid=((total + 1024 - 1) // 1024,),
                        warmup=False,
                    )
            else:
                (
                    _linear_kernel_bf16
                    if x.dtype == torch.bfloat16
                    else _linear_kernel_fp16
                ).run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total,
                    scale_inv,
                    fp8_dtype == torch.float8_e4m3fn,
                    grid=((total + 1024 - 1) // 1024,),
                    warmup=False,
                )
        else:
            n = x.size(-1)
            stride = _row_stride(x, n)
            (_kernel_bf16 if x.dtype == torch.bfloat16 else _kernel_fp16).run(
                x.data_ptr(),
                out.data_ptr(),
                total,
                n,
                stride,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                1024,
                grid=((total + 1024 - 1) // 1024,),
                warmup=False,
            )
    return out


@triton.jit
def _kernel_bf16(
    X: "*bf16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX: tl.constexpr = 448.0 if E4 else 57344.0
    tile = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647 or (TOTAL // N - 1) * STRIDE + N - 1 > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    src = off // N * STRIDE + off % N
    v = tl.load(X + src, off < TOTAL, other=0).to(tl.float32)
    if SCALE != 1.0:
        v = v * SCALE
    code = _encode(v, M, BIAS, MAX)
    tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code, off < TOTAL)


@triton.jit
def _kernel_fp16(
    X: "*fp16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX: tl.constexpr = 448.0 if E4 else 57344.0
    tile = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647 or (TOTAL // N - 1) * STRIDE + N - 1 > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    src = off // N * STRIDE + off % N
    v = tl.load(X + src, off < TOTAL, other=0).to(tl.float32)
    if SCALE != 1.0:
        v = v * SCALE
    code = _encode(v, M, BIAS, MAX)
    tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code, off < TOTAL)


@triton.jit
def _linear_kernel_bf16(
    X: "*bf16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
):
    BLOCK: tl.constexpr = 1024
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX: tl.constexpr = 448.0 if E4 else 57344.0
    tile = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    src = off
    v = tl.load(X + src, off < TOTAL, other=0).to(tl.float32)
    if SCALE != 1.0:
        v = v * SCALE
    code = _encode(v, M, BIAS, MAX)
    tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code, off < TOTAL)


@triton.jit
def _linear_kernel_fp16(
    X: "*fp16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
):
    BLOCK: tl.constexpr = 1024
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX: tl.constexpr = 448.0 if E4 else 57344.0
    tile = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    src = off
    v = tl.load(X + src, off < TOTAL, other=0).to(tl.float32)
    if SCALE != 1.0:
        v = v * SCALE
    code = _encode(v, M, BIAS, MAX)
    tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code, off < TOTAL)


@triton.jit
def _pair_linear_kernel_bf16(
    X: "*bf16", Y: "*u8", TOTAL: tl.constexpr, E4: tl.constexpr
):
    X = X.to(tl.pointer_type(tl.int32))
    Y = Y.to(tl.pointer_type(tl.int16))
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX16: tl.constexpr = 17376 if E4 else 18272
    BLOCK: tl.constexpr = 512
    off = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647
        else tl.program_id(0)
    ) * BLOCK + tl.arange(0, BLOCK)
    src = off
    if TOTAL % BLOCK == 0:
        raw = tl.load(X + src, cache_modifier=".cg").to(tl.uint32)
    else:
        raw = tl.load(X + src, off < TOTAL, other=0, cache_modifier=".cg").to(
            tl.uint32
        )
    a = tl.minimum(raw & 32767, MAX16)
    b = tl.minimum(raw >> 16 & 32767, MAX16)
    offset: tl.constexpr = -(127 - BIAS << M) & 255
    addend: tl.constexpr = 8388608.0 + offset - 8388608.0 * 2.0 ** (M - 7)
    na = ((a | 143 + M << 23).to(tl.float32, bitcast=True) + addend).to(
        tl.uint32, bitcast=True
    ) & 255
    nb = ((b | 143 + M << 23).to(tl.float32, bitcast=True) + addend).to(
        tl.uint32, bitcast=True
    ) & 255
    magic = tl.full((), 2.0 ** (24 - BIAS - M), tl.float32)
    sa = ((a << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.uint32, bitcast=True
    ) - magic.to(tl.uint32, bitcast=True)
    sb = ((b << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.uint32, bitcast=True
    ) - magic.to(tl.uint32, bitcast=True)
    ca = tl.where(a < 128 - BIAS << 7, sa, na)
    cb = tl.where(b < 128 - BIAS << 7, sb, nb)
    code = ca | cb << 8 | raw >> 8 & 128 | raw >> 16 & 32768
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code.to(tl.int16))
    else:
        tl.store(Y + off, code.to(tl.int16), off < TOTAL)


@triton.jit
def _pair_linear_kernel_fp16(
    X: "*fp16", Y: "*u8", TOTAL: tl.constexpr, E4: tl.constexpr
):
    X = X.to(tl.pointer_type(tl.int32))
    Y = Y.to(tl.pointer_type(tl.int16))
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX16: tl.constexpr = 17376 if E4 else 18272
    BLOCK: tl.constexpr = 512
    off = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647
        else tl.program_id(0)
    ) * BLOCK + tl.arange(0, BLOCK)
    src = off
    if TOTAL % BLOCK == 0:
        raw = tl.load(X + src, cache_modifier=".cg").to(tl.uint32)
    else:
        raw = tl.load(X + src, off < TOTAL, other=0, cache_modifier=".cg").to(
            tl.uint32
        )
    a = tl.minimum(raw & 32767, MAX16)
    b = tl.minimum(raw >> 16 & 32767, MAX16)
    offset: tl.constexpr = -(127 - BIAS << M) & 255
    addend: tl.constexpr = 8388608.0 + offset - 8388608.0 * 2.0 ** (M - 7)
    na = ((a | 143 + M << 23).to(tl.float32, bitcast=True) + addend).to(
        tl.uint32, bitcast=True
    ) & 255
    nb = ((b | 143 + M << 23).to(tl.float32, bitcast=True) + addend).to(
        tl.uint32, bitcast=True
    ) & 255
    magic = tl.full((), 2.0 ** (24 - BIAS - M), tl.float32)
    sa = ((a << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.uint32, bitcast=True
    ) - magic.to(tl.uint32, bitcast=True)
    sb = ((b << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.uint32, bitcast=True
    ) - magic.to(tl.uint32, bitcast=True)
    ca = tl.where(a < 128 - BIAS << 7, sa, na)
    cb = tl.where(b < 128 - BIAS << 7, sb, nb)
    code = ca | cb << 8 | raw >> 8 & 128 | raw >> 16 & 32768
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code.to(tl.int16))
    else:
        tl.store(Y + off, code.to(tl.int16), off < TOTAL)


@triton.jit
def _unit_bf16(X: "*bf16", Y: "*u8", TOTAL: tl.constexpr):
    _linear_kernel_bf16(X, Y, TOTAL, 1.0, True)


@triton.jit
def _unit_fp16(X: "*fp16", Y: "*u8", TOTAL: tl.constexpr):
    _linear_kernel_fp16(X, Y, TOTAL, 1.0, True)


@triton.jit
def _full_bf16_1024_0(X: "*bf16", Y: "*u8"):
    off = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    value = tl.load(X + off).to(tl.float32)
    code = _encode(value, 3, 7, 448.0)
    tl.store(Y + off, code, cache_modifier="")


@triton.jit
def _full_fp16_1024_0(X: "*fp16", Y: "*u8"):
    off = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    value = tl.load(X + off).to(tl.float32)
    code = _encode(value, 3, 7, 448.0)
    tl.store(Y + off, code, cache_modifier="")


def _row_stride(x, n):
    for axis in range(x.ndim - 2, -1, -1):
        if x.size(axis) > 1:
            return x.stride(axis)
    return n


@triton.jit
def _half_code_v471(raw, E4: tl.constexpr):
    sign = raw >> 8 & 128
    mag = tl.minimum(raw & 32767, 24320 if E4 else 31488)
    if E4:
        normal = (
            (mag | 143 << 23).to(tl.float32, bitcast=True) + 8323008.0
        ).to(tl.uint32, bitcast=True)
        sub = ((mag + 114688 << 13).to(tl.float32, bitcast=True) + 16384.0).to(
            tl.uint32, bitcast=True
        )
        code = tl.where(mag < 9216, sub, normal)
    else:
        code = ((mag | 142 << 23).to(tl.float32, bitcast=True) + 8355840.0).to(
            tl.uint32, bitcast=True
        )
    return code & 127 | sign


@triton.jit
def _half_scalar_v471(
    X: "*u16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    if TOTAL % BLOCK == 0:
        raw = tl.load(X + off).to(tl.uint32)
    else:
        raw = tl.load(X + off, off < TOTAL, other=0).to(tl.uint32)
    code = _half_code_v471(raw, E4).to(tl.uint8)
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code)
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _half_pair_v471(
    X: "*u32",
    Y: "*u16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    if TOTAL % BLOCK == 0:
        raw = tl.load(X + off)
    else:
        raw = tl.load(X + off, off < TOTAL, other=0)
    code = (_half_code_v471(raw, E4) | _half_code_v471(raw >> 16, E4) << 8).to(
        tl.uint16
    )
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code)
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _half_split_v494(
    X: "*u16",
    Y: "*u16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    source = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
    halves = tl.load(X + source, source < TOTAL // 2 * 2, other=0).to(
        tl.uint32
    )
    a, b = tl.split(tl.reshape(halves, (BLOCK, 2)))
    code = (_half_code_v471(a, E4) | _half_code_v471(b, E4) << 8).to(tl.uint16)
    tl.store(Y + off, code, off < TOTAL // 2)
    if TOTAL % 2:
        raw = tl.load(X + TOTAL - 1, tile == 0, other=0).to(tl.uint32)
        final = _half_code_v471(raw, E4).to(tl.uint8)
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + TOTAL - 1, final, tile == 0)


@triton.jit
def _half_scalar_fulltail_v501(
    X: "*u16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    if tile < TOTAL // BLOCK:
        raw = tl.load(X + off).to(tl.uint32)
    else:
        raw = tl.load(X + off, off < TOTAL, other=0).to(tl.uint32)
    code = _half_code_v471(raw, E4).to(tl.uint8)
    if tile < TOTAL // BLOCK:
        tl.store(Y + off, code)
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _general_scalar_v579(
    X: "*u16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    off = tile * BLOCK + tl.arange(0, BLOCK)
    if tile < TOTAL // BLOCK:
        raw = tl.load(X + off)
    else:
        raw = tl.load(X + off, off < TOTAL, other=0)
    v = (
        raw.to(tl.bfloat16, bitcast=True).to(tl.float32)
        if BF16
        else raw.to(tl.float16, bitcast=True).to(tl.float32)
    )
    v = v * SCALE
    code = _encode(
        v, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
    )
    code = code & 127 | raw >> 8 & 128 ^ (128 if SCALE < 0.0 else 0)
    if tile < TOTAL // BLOCK:
        tl.store(Y + off, code.to(tl.uint8))
    else:
        tl.store(Y + off, code.to(tl.uint8), off < TOTAL)


@triton.jit
def _zero_scale_v587(
    X: "*u16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    NEG: tl.constexpr,
    BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        tl.program_id(0), tl.cdiv(TOTAL, BLOCK), tl.num_programs(0)
    ):
        off = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        source = off // N * PITCH + off % N
        raw = tl.load(X + source, off < TOTAL, other=0).to(tl.int32)
        sign = raw >> 8 & 128
        mag = raw & 32767
        inf: tl.constexpr = 32640 if BF16 else 31744
        code = tl.where(
            mag > inf,
            sign | 127,
            tl.where(mag == inf, 255, sign ^ (128 if NEG else 0)),
        )
        tl.store(Y + off, code.to(tl.uint8), off < TOTAL)


@triton.jit
def _bf16_code_v648(raw, E4: tl.constexpr):
    v = raw.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    return _encode(
        v, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
    ).to(tl.uint16)


@triton.jit
def _bf16_fulltail_v650(
    X: "*u16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    if tile < TOTAL // BLOCK:
        raw = tl.load(X + off).to(tl.uint32)
    else:
        raw = tl.load(X + off, off < TOTAL, other=0).to(tl.uint32)
    code = _bf16_code_v648(raw, E4).to(tl.uint8)
    if tile < TOTAL // BLOCK:
        tl.store(Y + off, code)
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _bf16_split_v648(
    X: "*u16",
    Y: "*u16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    source = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
    halves = tl.load(X + source, source < TOTAL // 2 * 2, other=0).to(
        tl.uint32
    )
    a, b = tl.split(tl.reshape(halves, (BLOCK, 2)))
    code = (_bf16_code_v648(a, E4) | _bf16_code_v648(b, E4) << 8).to(tl.uint16)
    tl.store(Y + off, code, off < TOTAL // 2)
    if TOTAL % 2:
        raw = tl.load(X + TOTAL - 1, tile == 0, other=0).to(tl.uint32)
        final = _bf16_code_v648(raw, E4).to(tl.uint8)
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + TOTAL - 1, final, tile == 0)


__all__ = ["fp8_quantize"]
