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
from triton.language import int16 as i16
from triton.language import int32 as i32
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
    if scale_inv == 0.0:
        out = torch.empty(x.shape, dtype=fp8_dtype, device=x.device)
        total = x.numel()
        if total:
            if (
                8388608 <= total < 2147483648
                and (not x.is_contiguous())
                and (x.stride(-1) == 1)
            ):
                n = x.size(-1)
                pitch = _row_stride(x, n)
                if (
                    1024 <= n <= 65536
                    and n & n - 1 == 0
                    and ((total // n - 1) * pitch + n < 2147483648)
                ):
                    block = 8192 if total >= 33554432 and n >= 8192 else 4096
                    _zero_flat_v760.run(
                        x.data_ptr(),
                        out.data_ptr(),
                        total,
                        n,
                        pitch,
                        math.copysign(1.0, scale_inv) < 0.0,
                        x.dtype == torch.bfloat16,
                        block,
                        grid=((total + block - 1) // block,),
                        num_warps=32,
                        warmup=False,
                    )
                    return out
            if (
                16777216 <= total < 2147483648
                and (not x.is_contiguous())
                and (x.stride(-1) == 1)
            ):
                n = x.size(-1)
                pitch = _row_stride(x, n)
                if (
                    1 <= n <= 64
                    and n & n - 1 == 0
                    and (pitch > 2 * n)
                    and ((total // n - 1) * pitch + n < 2147483648)
                ):
                    packed_ok = (
                        n == 2 and pitch == 8 * n and (x.data_ptr() % 16 == 0)
                    )
                    span_ok = (
                        2 <= n <= 4
                        and 3 * n <= pitch <= 8 * n
                        and ((total // n - 1) * pitch + n <= 2147450880)
                    )
                    block = (
                        2048
                        if packed_ok
                        else 512 if span_ok else 1024 if n == 1 else 4096
                    )
                    (
                        _zero_packed_selected_v883
                        if packed_ok
                        else (
                            _zero_span_v855
                            if span_ok
                            else _zero_strided_pair_v725
                        )
                    ).run(
                        x.data_ptr(),
                        out.data_ptr(),
                        out.data_ptr(),
                        total,
                        n,
                        pitch,
                        math.copysign(1.0, scale_inv) < 0.0,
                        x.dtype == torch.bfloat16,
                        block,
                        grid=((total + 2 * block - 1) // (2 * block),),
                        num_warps=32 if packed_ok else 16 if span_ok else 32,
                        warmup=False,
                    )
                    return out
            if (
                33554432 <= total < 2147483648
                and total % 2 == 0
                and x.is_contiguous()
                and (x.storage_offset() % 2 == 0)
            ):
                _zero_pair_portable_v613.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total // 2,
                    math.copysign(1.0, scale_inv) < 0.0,
                    x.dtype == torch.bfloat16,
                    2048,
                    grid=((total + 4095) // 4096,),
                    num_warps=32,
                    warmup=False,
                )
                return out
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
        total >= 2097152
        and total % 2 == 0
        and (x.dtype == torch.float16)
        and (0.00390625 <= scale_inv <= 0.5)
        and x.is_contiguous()
        and (x.storage_offset() % 2 == 0)
    ):
        fraction, exponent = math.frexp(scale_inv)
        if fraction == 0.5:
            out = torch.empty_like(x, dtype=fp8_dtype)
            block = 4096 if total > 1048576 else 1024
            _half_pair_v490.run(
                x.data_ptr(),
                out.data_ptr(),
                total // 2,
                fp8_dtype == torch.float8_e4m3fn,
                1 - exponent,
                block // 2,
                grid=((total + block - 1) // block,),
                num_warps=(
                    32 if total >= 8388608 else 16 if total > 1048576 else 4
                ),
                enable_fp_fusion=False,
                warmup=False,
            )
            return out
    if (
        (
            total >= 4194304
            or (total >= 1048576 and fp8_dtype == torch.float8_e5m2)
        )
        and total % 2 == 0
        and (x.dtype == torch.float16)
        and (scale_inv == 1.0)
        and x.is_contiguous()
        and (x.storage_offset() % 2 == 0)
    ):
        out = torch.empty_like(x, dtype=fp8_dtype)
        block = 4096 if total > 1048576 else 1024
        _half_pair_v471.run(
            x.data_ptr(),
            out.data_ptr(),
            total // 2,
            fp8_dtype == torch.float8_e4m3fn,
            block // 2,
            grid=((total + block - 1) // block,),
            num_warps=32 if total >= 8388608 else 16 if total > 1048576 else 4,
            warmup=False,
        )
        return out
    if (
        2097152 <= total < 2147483648
        and total % 2 == 0
        and (scale_inv != 1.0)
        and x.is_contiguous()
        and (x.storage_offset() % 2 == 0)
    ):
        out = torch.empty_like(x, dtype=fp8_dtype)
        _general_pair_v561.run(
            x.data_ptr(),
            out.data_ptr(),
            total // 2,
            x.dtype == torch.bfloat16,
            scale_inv,
            fp8_dtype == torch.float8_e4m3fn,
            2048,
            grid=((total + 4095) // 4096,),
            num_warps=32 if total >= 8388608 else 16 if total > 1048576 else 4,
            enable_fp_fusion=False,
            warmup=False,
        )
        return out
    out = torch.empty_like(x, dtype=fp8_dtype)
    if (
        total >= 8388608
        and x.is_contiguous()
        and (x.dtype == torch.bfloat16)
        and (scale_inv == 1.0)
        and (fp8_dtype == torch.float8_e4m3fn)
    ):
        _stripmine.run(
            x.data_ptr(),
            out.data_ptr(),
            total,
            grid=((total + 8191) // 8192,),
            num_warps=32,
            warmup=False,
        )
        return out
    if total < 4096:
        block = 1024
        warps = 16 if total <= 1024 else 4
    elif total < 32768:
        block = 4096
        warps = 16
    elif total <= 65536:
        block = 8192
        warps = 32
    elif total <= 1048576:
        block = 1024
        warps = 4
    elif total <= 4194304:
        block = 4096
        warps = 16
    elif total < 8388608:
        block = 1024
        warps = 4
    else:
        block = 4096
        warps = 32
    if total:
        if x.is_contiguous():
            if scale_inv == 1.0 and fp8_dtype == torch.float8_e4m3fn:
                if total < 8388608 and total % block == 0:
                    if block == 8192:
                        fn = (
                            _full_bf16_8192_0
                            if x.dtype == torch.bfloat16
                            else _full_fp16_8192_0
                        )
                    elif block == 4096:
                        if total >= 262144 and total <= 4194304:
                            fn = (
                                _full_bf16_4096_1
                                if x.dtype == torch.bfloat16
                                else _full_fp16_4096_1
                            )
                        else:
                            fn = (
                                _full_bf16_4096_0
                                if x.dtype == torch.bfloat16
                                else _full_fp16_4096_0
                            )
                    elif total >= 262144 and total <= 4194304:
                        fn = (
                            _full_bf16_1024_1
                            if x.dtype == torch.bfloat16
                            else _full_fp16_1024_1
                        )
                    else:
                        fn = (
                            _full_bf16_1024_0
                            if x.dtype == torch.bfloat16
                            else _full_fp16_1024_0
                        )
                    fn.run(
                        x.data_ptr(),
                        out.data_ptr(),
                        grid=(total // block,),
                        warmup=False,
                        num_warps=warps,
                    )
                else:
                    (
                        _unit_bf16 if x.dtype == torch.bfloat16 else _unit_fp16
                    ).run(
                        x.data_ptr(),
                        out.data_ptr(),
                        total,
                        num_warps=warps,
                        grid=((total + block - 1) // block,),
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
                    block,
                    num_warps=warps,
                    grid=((total + block - 1) // block,),
                    warmup=False,
                )
        else:
            n = x.size(-1)
            stride = _row_stride(x, n)
            if (
                total > 2147483647
                or (total // n - 1) * stride + n - 1 > 2147483647
            ):
                block = min(block, n)
            (_kernel_bf16 if x.dtype == torch.bfloat16 else _kernel_fp16).run(
                x.data_ptr(),
                out.data_ptr(),
                total,
                n,
                stride,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                block,
                num_warps=warps,
                grid=((total + block - 1) // block,),
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
    if BLOCK <= N:
        src = (
            tile * BLOCK // N * STRIDE + tile * BLOCK % N + tl.arange(0, BLOCK)
        )
    else:
        src = off // N * STRIDE + off % N
    v = (
        tl.load(X + src)
        if TOTAL % BLOCK == 0
        else tl.load(X + src, off < TOTAL, other=0)
    ).to(tl.float32)
    if SCALE != 1.0:
        v = v * SCALE
    code = _encode(v, M, BIAS, MAX)
    if TOTAL % BLOCK == 0:
        tl.store(
            Y.to(tl.pointer_type(tl.uint8)) + off,
            code,
            cache_modifier=(
                ".cs" if TOTAL >= 262144 and TOTAL <= 4194304 else ""
            ),
        )
    else:
        tl.store(
            Y.to(tl.pointer_type(tl.uint8)) + off,
            code,
            off < TOTAL,
            cache_modifier=(
                ".cs" if TOTAL >= 262144 and TOTAL <= 4194304 else ""
            ),
        )


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
    if BLOCK <= N:
        src = (
            tile * BLOCK // N * STRIDE + tile * BLOCK % N + tl.arange(0, BLOCK)
        )
    else:
        src = off // N * STRIDE + off % N
    v = (
        tl.load(X + src)
        if TOTAL % BLOCK == 0
        else tl.load(X + src, off < TOTAL, other=0)
    ).to(tl.float32)
    if SCALE != 1.0:
        v = v * SCALE
    code = _encode(v, M, BIAS, MAX)
    if TOTAL % BLOCK == 0:
        tl.store(
            Y.to(tl.pointer_type(tl.uint8)) + off,
            code,
            cache_modifier=(
                ".cs" if TOTAL >= 262144 and TOTAL <= 4194304 else ""
            ),
        )
    else:
        tl.store(
            Y.to(tl.pointer_type(tl.uint8)) + off,
            code,
            off < TOTAL,
            cache_modifier=(
                ".cs" if TOTAL >= 262144 and TOTAL <= 4194304 else ""
            ),
        )


@triton.jit
def _linear_kernel_bf16(
    X: "*bf16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
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
    v = (
        tl.load(X + src)
        if TOTAL % BLOCK == 0
        else tl.load(X + src, off < TOTAL, other=0)
    ).to(tl.float32)
    if SCALE != 1.0:
        v = v * SCALE
    code = _encode(v, M, BIAS, MAX)
    if TOTAL % BLOCK == 0:
        tl.store(
            Y.to(tl.pointer_type(tl.uint8)) + off,
            code,
            cache_modifier=(
                ".cs" if TOTAL >= 262144 and TOTAL <= 4194304 else ""
            ),
        )
    else:
        tl.store(
            Y.to(tl.pointer_type(tl.uint8)) + off,
            code,
            off < TOTAL,
            cache_modifier=(
                ".cs" if TOTAL >= 262144 and TOTAL <= 4194304 else ""
            ),
        )


@triton.jit
def _linear_kernel_fp16(
    X: "*fp16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
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
    v = (
        tl.load(X + src)
        if TOTAL % BLOCK == 0
        else tl.load(X + src, off < TOTAL, other=0)
    ).to(tl.float32)
    if SCALE != 1.0:
        v = v * SCALE
    code = _encode(v, M, BIAS, MAX)
    if TOTAL % BLOCK == 0:
        tl.store(
            Y.to(tl.pointer_type(tl.uint8)) + off,
            code,
            cache_modifier=(
                ".cs" if TOTAL >= 262144 and TOTAL <= 4194304 else ""
            ),
        )
    else:
        tl.store(
            Y.to(tl.pointer_type(tl.uint8)) + off,
            code,
            off < TOTAL,
            cache_modifier=(
                ".cs" if TOTAL >= 262144 and TOTAL <= 4194304 else ""
            ),
        )


@triton.jit
def _unit_bf16(X: "*bf16", Y: "*u8", TOTAL: tl.constexpr):
    BLOCK: tl.constexpr = (
        8192
        if TOTAL >= 32768 and TOTAL <= 65536
        else (
            4096
            if TOTAL >= 4096
            and TOTAL < 32768
            or (TOTAL > 1048576 and TOTAL <= 4194304)
            or TOTAL >= 8388608
            else 1024
        )
    )
    _linear_kernel_bf16(X, Y, TOTAL, 1.0, True, BLOCK)


@triton.jit
def _unit_fp16(X: "*fp16", Y: "*u8", TOTAL: tl.constexpr):
    BLOCK: tl.constexpr = (
        8192
        if TOTAL >= 32768 and TOTAL <= 65536
        else (
            4096
            if TOTAL >= 4096
            and TOTAL < 32768
            or (TOTAL > 1048576 and TOTAL <= 4194304)
            or TOTAL >= 8388608
            else 1024
        )
    )
    _linear_kernel_fp16(X, Y, TOTAL, 1.0, True, BLOCK)


@triton.jit
def _stripmine(X: "*bf16", Y: "*u8", TOTAL: tl.constexpr):
    for part in tl.static_range(0, 2):
        off = (
            (
                tl.program_id(0).to(tl.int64)
                if TOTAL > 2147483647
                else tl.program_id(0)
            )
            * 2
            + part
        ) * 4096 + tl.arange(0, 4096)
        if TOTAL % 8192 == 0:
            v = tl.load(X + off).to(tl.float32)
        else:
            v = tl.load(X + off, off < TOTAL, other=0).to(tl.float32)
        code = _encode(v, 3, 7, 448.0)
        if TOTAL % 8192 == 0:
            tl.store(Y + off, code)
        else:
            tl.store(Y + off, code, off < TOTAL)


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


@triton.jit
def _full_bf16_4096_0(X: "*bf16", Y: "*u8"):
    off = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    value = tl.load(X + off).to(tl.float32)
    code = _encode(value, 3, 7, 448.0)
    tl.store(Y + off, code, cache_modifier="")


@triton.jit
def _full_fp16_4096_0(X: "*fp16", Y: "*u8"):
    off = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    value = tl.load(X + off).to(tl.float32)
    code = _encode(value, 3, 7, 448.0)
    tl.store(Y + off, code, cache_modifier="")


@triton.jit
def _full_bf16_8192_0(X: "*bf16", Y: "*u8"):
    off = tl.program_id(0) * 8192 + tl.arange(0, 8192)
    value = tl.load(X + off).to(tl.float32)
    code = _encode(value, 3, 7, 448.0)
    tl.store(Y + off, code, cache_modifier="")


@triton.jit
def _full_fp16_8192_0(X: "*fp16", Y: "*u8"):
    off = tl.program_id(0) * 8192 + tl.arange(0, 8192)
    value = tl.load(X + off).to(tl.float32)
    code = _encode(value, 3, 7, 448.0)
    tl.store(Y + off, code, cache_modifier="")


@triton.jit
def _full_bf16_1024_1(X: "*bf16", Y: "*u8"):
    off = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    value = tl.load(X + off).to(tl.float32)
    code = _encode(value, 3, 7, 448.0)
    tl.store(Y + off, code, cache_modifier=".cs")


@triton.jit
def _full_fp16_1024_1(X: "*fp16", Y: "*u8"):
    off = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    value = tl.load(X + off).to(tl.float32)
    code = _encode(value, 3, 7, 448.0)
    tl.store(Y + off, code, cache_modifier=".cs")


@triton.jit
def _full_bf16_4096_1(X: "*bf16", Y: "*u8"):
    off = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    value = tl.load(X + off).to(tl.float32)
    code = _encode(value, 3, 7, 448.0)
    tl.store(Y + off, code, cache_modifier=".cs")


@triton.jit
def _full_fp16_4096_1(X: "*fp16", Y: "*u8"):
    off = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    value = tl.load(X + off).to(tl.float32)
    code = _encode(value, 3, 7, 448.0)
    tl.store(Y + off, code, cache_modifier=".cs")


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
def _half_code_v490(raw, E4: tl.constexpr, SHIFT: tl.constexpr):
    sign = raw >> 8 & 128
    mag = tl.minimum(raw & 32767, (24320 if E4 else 31488) + SHIFT * 1024)
    shifted = tl.maximum(mag.to(tl.int32) - SHIFT * 1024, 0).to(tl.uint32)
    normal = (
        (shifted | (143 if E4 else 142) << 23).to(tl.float32, bitcast=True)
        + (8323008.0 if E4 else 8355840.0)
    ).to(tl.uint32, bitcast=True)
    subvalue = (mag + (112 - SHIFT) * 1024 << 13).to(tl.float32, bitcast=True)
    if not E4 and SHIFT <= 2:
        subvalue = tl.where(
            mag < 1024, mag.to(tl.float32) * 2.0 ** (-24 - SHIFT), subvalue
        )
    sub = (subvalue + (16384.0 if E4 else 128.0)).to(tl.uint32, bitcast=True)
    code = tl.where(mag < (9216 if E4 else 1024) + SHIFT * 1024, sub, normal)
    return code & 127 | sign


@triton.jit
def _half_pair_v490(
    X: "*u32",
    Y: "*u16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    SHIFT: tl.constexpr,
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
    code = (
        _half_code_v490(raw, E4, SHIFT)
        | _half_code_v490(raw >> 16, E4, SHIFT) << 8
    ).to(tl.uint16)
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code)
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _general_pair_v561(
    X: "*u32",
    Y: "*u16",
    TOTAL: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw = (
        tl.load(X + off)
        if TOTAL % BLOCK == 0
        else tl.load(X + off, off < TOTAL, other=0)
    )
    lo = raw.to(tl.uint16)
    hi = (raw >> 16).to(tl.uint16)
    if BF16:
        a = lo.to(tl.bfloat16, bitcast=True).to(tl.float32)
        b = hi.to(tl.bfloat16, bitcast=True).to(tl.float32)
    else:
        a = lo.to(tl.float16, bitcast=True).to(tl.float32)
        b = hi.to(tl.float16, bitcast=True).to(tl.float32)
    a = a * SCALE
    b = b * SCALE
    ca = _encode(
        a, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
    ).to(tl.uint16)
    cb = _encode(
        b, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
    ).to(tl.uint16)
    code = ca | cb << 8
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code)
    else:
        tl.store(Y + off, code, off < TOTAL)


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
def _zero_pair_portable_v613(
    X: "*i32",
    Y: "*i16",
    PAIRS: tl.constexpr,
    NEG: tl.constexpr,
    BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    off = tile * BLOCK + tl.arange(0, BLOCK)
    raw = (
        tl.load(X + off)
        if PAIRS % BLOCK == 0
        else tl.load(X + off, off < PAIRS, other=0)
    )
    a = raw & 32767
    b = raw >> 16 & 32767
    sa = (raw >> 8 & 128).to(tl.int16)
    sb = (raw >> 24 & 128).to(tl.int16)
    inf: tl.constexpr = 32640 if BF16 else 31744
    ca = tl.where(
        a > inf, sa | 127, tl.where(a == inf, 255, sa ^ (128 if NEG else 0))
    ).to(tl.int16)
    cb = tl.where(
        b > inf, sb | 127, tl.where(b == inf, 255, sb ^ (128 if NEG else 0))
    ).to(tl.int16)
    code = ca | cb << 8
    if PAIRS % BLOCK == 0:
        tl.store(Y + off, code)
    else:
        tl.store(Y + off, code, off < PAIRS)


@triton.jit
def _zero_strided_pair_v725(
    X: "*u16",
    Y: "*u16",
    YB: "*u8",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    NEG: tl.constexpr,
    BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    logical = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
    source = logical // N * PITCH + logical % N
    raw = tl.load(X + source, logical < TOTAL, other=0).to(tl.int32)
    sign = raw >> 8 & 128
    mag = raw & 32767
    inf: tl.constexpr = 32640 if BF16 else 31744
    code = tl.where(
        mag > inf,
        sign | 127,
        tl.where(mag == inf, 255, sign ^ (128 if NEG else 0)),
    ).to(tl.uint16)
    lo, hi = tl.split(tl.reshape(code, (BLOCK, 2)))
    pairs = tile * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + pairs, lo | hi << 8, pairs < TOTAL // 2)
    if TOTAL % 2 != 0:
        tl.store(YB + logical, code.to(tl.uint8), logical == TOTAL - 1)


@triton.jit
def _zero_flat_v760(
    X: "*u16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    NEG: tl.constexpr,
    BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    off = tile * BLOCK + tl.arange(0, BLOCK)
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
def _zero_span_v855(
    X: "*u16",
    Y: "*u16",
    YB: "*u8",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    NEG: tl.constexpr,
    BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    SPAN: tl.constexpr = triton.next_power_of_2(
        (2 * BLOCK // N - 1) * PITCH + N
    )
    LAST: tl.constexpr = (TOTAL // N - 1) * PITCH + N
    tile = tl.program_id(0)
    start = tile * 2 * BLOCK // N * PITCH
    physical = start + tl.arange(0, SPAN)
    values = tl.load(X + physical, physical < LAST, other=0)
    local = tl.arange(0, 2 * BLOCK)
    indices = local // N * PITCH + local % N
    raw = tl.gather(values, indices, 0).to(tl.int32)
    sign = raw >> 8 & 128
    mag = raw & 32767
    inf: tl.constexpr = 32640 if BF16 else 31744
    code = tl.where(
        mag > inf,
        sign | 127,
        tl.where(mag == inf, 255, sign ^ (128 if NEG else 0)),
    ).to(tl.uint16)
    lo, hi = tl.split(tl.reshape(code, (BLOCK, 2)))
    pairs = tile * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + pairs, lo | hi << 8, pairs < TOTAL // 2)
    if TOTAL % 2 != 0:
        logical = tile * 2 * BLOCK + local
        tl.store(YB + logical, code.to(tl.uint8), logical == TOTAL - 1)


@triton.jit
def _zero_packed_selected_v883(
    X: "*i32",
    Y: "*i16",
    YB: "*u8",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    NEG: tl.constexpr,
    BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    PAIRS: tl.constexpr = TOTAL // 2
    tile = tl.program_id(0)
    off = tile * BLOCK + tl.arange(0, BLOCK)
    source = off // (N // 2) * (PITCH // 2) + off % (N // 2)
    raw = (
        tl.load(X + source)
        if PAIRS % BLOCK == 0
        else tl.load(X + source, off < PAIRS, other=0)
    )
    a = raw & 32767
    b = raw >> 16 & 32767
    sa = (raw >> 8 & 128).to(tl.int16)
    sb = (raw >> 24 & 128).to(tl.int16)
    inf: tl.constexpr = 32640 if BF16 else 31744
    ca = tl.where(
        a > inf, sa | 127, tl.where(a == inf, 255, sa ^ (128 if NEG else 0))
    ).to(tl.int16)
    cb = tl.where(
        b > inf, sb | 127, tl.where(b == inf, 255, sb ^ (128 if NEG else 0))
    ).to(tl.int16)
    code = ca | cb << 8
    if PAIRS % BLOCK == 0:
        tl.store(Y + off, code)
    else:
        tl.store(Y + off, code, off < PAIRS)


__all__ = ["fp8_quantize"]
