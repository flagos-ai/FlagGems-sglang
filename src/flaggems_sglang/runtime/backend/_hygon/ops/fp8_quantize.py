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
def _encode_bf16_magic(
    v, M: tl.constexpr, BIAS: tl.constexpr, MAX16: tl.constexpr
):
    raw = v.to(tl.uint16, bitcast=True).to(tl.uint32)
    sign = raw >> 8 & 128
    mag = tl.minimum(raw & 32767, MAX16)
    offset: tl.constexpr = -(127 - BIAS << M) & 255
    addend: tl.constexpr = 8388608.0 + offset - 8388608.0 * 2.0 ** (M - 7)
    normal = ((mag | 143 + M << 23).to(tl.float32, bitcast=True) + addend).to(
        tl.uint32, bitcast=True
    )
    magic = tl.full((), 2.0 ** (24 - BIAS - M), tl.float32)
    sub = ((mag << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.uint32, bitcast=True
    )
    code = tl.where(mag < 128 - BIAS << 7, sub, normal)
    return (code | sign).to(tl.uint8)


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


@triton.jit
def _kernel(
    X,
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
    v = (
        tl.load(X + src)
        if TOTAL % BLOCK == 0
        else tl.load(X + src, off < TOTAL, other=0)
    )
    if X.dtype.element_ty == tl.bfloat16 and SCALE == 1.0:
        code = (
            _encode_bf16_max(v, M, BIAS, 17376 if E4 else 18272)
            if TOTAL > 1048576 and TOTAL <= 4194304
            else _encode_bf16_magic(v, M, BIAS, 17376 if E4 else 18272)
        )
    else:
        v = v.to(tl.float32)
        if SCALE != 1.0:
            v = v * SCALE
        code = _encode(v, M, BIAS, MAX)
    if TOTAL % BLOCK == 0:
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code)
    else:
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code, off < TOTAL)


@triton.jit
def _linear_kernel(
    X,
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
    )
    if X.dtype.element_ty == tl.bfloat16 and SCALE == 1.0:
        code = (
            _encode_bf16_max(v, M, BIAS, 17376 if E4 else 18272)
            if TOTAL > 1048576 and TOTAL <= 4194304
            else _encode_bf16_magic(v, M, BIAS, 17376 if E4 else 18272)
        )
    else:
        v = v.to(tl.float32)
        if SCALE != 1.0:
            v = v * SCALE
        code = _encode(v, M, BIAS, MAX)
    if TOTAL % BLOCK == 0:
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code)
    else:
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code, off < TOTAL)


def fp8_quantize(x, scale_inv, fp8_dtype):
    total = x.numel()
    if scale_inv == 0.0:
        out = torch.empty(x.shape, dtype=fp8_dtype, device=x.device)
        if total:
            if (
                16777216 <= total < 2147483648
                and (not x.is_contiguous())
                and (x.stride(-1) == 1)
            ):
                n = x.size(-1)
                pitch = _row_stride(x, n)
                if (
                    n > 0
                    and n & n - 1 == 0
                    and ((total // n - 1) * pitch + n < 2147483648)
                ):
                    _zero_strided_pair_v725.run(
                        x.data_ptr(),
                        out.data_ptr(),
                        out.data_ptr(),
                        total,
                        n,
                        pitch,
                        math.copysign(1.0, scale_inv) < 0.0,
                        x.dtype == torch.bfloat16,
                        1024,
                        grid=((total + 2047) // 2048,),
                        num_warps=8,
                        warmup=False,
                    )
                    return out
            if (
                1048576 <= total < 2147483648
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
                    num_warps=8,
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
    if total >= 1048576:
        if (
            1048576 <= total < 2147483648
            and scale_inv != 0.0
            and (not x.is_contiguous())
            and (x.stride(-1) == 1)
        ):
            width = x.size(-1)
            pitch = _row_stride(x, width)
            if (
                1 <= width <= 1024
                and width & width - 1 == 0
                and (pitch > 2 * width)
                and ((total // width - 1) * pitch + width < 2147483648)
            ):
                if (
                    total >= 16777216
                    and width >= 256
                    and (x.dtype == torch.bfloat16)
                    and (2.0 <= abs(scale_inv) <= 256.0)
                ):
                    fraction, exponent = math.frexp(abs(scale_inv))
                    if fraction == 0.5:
                        out = torch.empty_like(
                            x,
                            dtype=fp8_dtype,
                            memory_format=torch.contiguous_format,
                        )
                        tail = (total - 1) // width * pitch + (
                            total - 1
                        ) % width
                        _sparse_power_v958.run(
                            x.data_ptr(),
                            out.data_ptr(),
                            x.data_ptr() + 2 * tail,
                            out.data_ptr() + total - 1,
                            total,
                            width,
                            pitch,
                            x.dtype == torch.bfloat16,
                            scale_inv,
                            fp8_dtype == torch.float8_e4m3fn,
                            (
                                2048
                                if width >= 1024 and total >= 16777216
                                else 1024
                            ),
                            exponent - 1,
                            grid=(
                                (
                                    total
                                    + 2
                                    * (
                                        2048
                                        if width >= 1024 and total >= 16777216
                                        else 1024
                                    )
                                    - 1
                                )
                                // (
                                    2
                                    * (
                                        2048
                                        if width >= 1024 and total >= 16777216
                                        else 1024
                                    )
                                ),
                            ),
                            num_warps=8,
                            enable_fp_fusion=False,
                            warmup=False,
                        )
                        return out
                out = torch.empty_like(
                    x, dtype=fp8_dtype, memory_format=torch.contiguous_format
                )
                tail = (total - 1) // width * pitch + (total - 1) % width
                _sparse_pair_v701.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    x.data_ptr() + 2 * tail,
                    out.data_ptr() + total - 1,
                    total,
                    width,
                    pitch,
                    x.dtype == torch.bfloat16,
                    scale_inv,
                    fp8_dtype == torch.float8_e4m3fn,
                    2048 if width >= 1024 and total >= 16777216 else 1024,
                    grid=(
                        (
                            total
                            + 2
                            * (
                                2048
                                if width >= 1024 and total >= 16777216
                                else 1024
                            )
                            - 1
                        )
                        // (
                            2
                            * (
                                2048
                                if width >= 1024 and total >= 16777216
                                else 1024
                            )
                        ),
                    ),
                    num_warps=8,
                    enable_fp_fusion=False,
                    warmup=False,
                )
                return out
            if (
                1 <= width <= 65536
                and width & width - 1 == 0
                and (pitch <= 2 * width or width >= 2048)
                and ((total // width - 1) * pitch + width < 2147483648)
            ) and (
                width == 1
                or width >= 1024
                or (pitch == 2 * width and total >= 8388608)
            ):
                out = torch.empty_like(
                    x, dtype=fp8_dtype, memory_format=torch.contiguous_format
                )
                tail = (total - 1) // width * pitch + (total - 1) % width
                _hw_strided_pair_v606_mx664.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    x.data_ptr() + 2 * tail,
                    out.data_ptr() + total - 1,
                    total,
                    width,
                    pitch,
                    x.dtype == torch.bfloat16,
                    scale_inv,
                    fp8_dtype == torch.float8_e4m3fn,
                    1024,
                    grid=((total + 2047) // 2048,),
                    num_warps=8,
                    enable_fp_fusion=False,
                    warmup=False,
                )
                return out
        if (
            33554432 <= total < 2147483648
            and x.dtype == torch.bfloat16
            and (scale_inv == -1.0)
            and x.is_contiguous()
            and (total % 2 != 0)
            and (x.data_ptr() % 16 == 0)
        ):
            out = torch.empty_like(x, dtype=fp8_dtype)
            _negative_split_v994.run(
                x.data_ptr(),
                out.data_ptr(),
                total,
                fp8_dtype == torch.float8_e4m3fn,
                2048,
                grid=((total + 4095) // 4096,),
                num_warps=8,
                warmup=False,
            )
            return out
        if (
            total >= 4194304
            and x.dtype == torch.bfloat16
            and (scale_inv == 1.0)
            and x.is_contiguous()
            and (total % 2 != 0 or x.data_ptr() % 16 != 0)
        ):
            out = torch.empty_like(x, dtype=fp8_dtype)
            if total >= 33554432 and x.data_ptr() % 16 == 0:
                _bf16_split_v648.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total,
                    fp8_dtype == torch.float8_e4m3fn,
                    2048,
                    grid=((total + 4095) // 4096,),
                    num_warps=8,
                    warmup=False,
                )
            else:
                _bf16_fulltail_v650.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total,
                    fp8_dtype == torch.float8_e4m3fn,
                    4096,
                    grid=((total + 4095) // 4096,),
                    num_warps=8,
                    warmup=False,
                )
            return out
        if (
            8388608 <= total < 2147483648
            and (not x.is_contiguous())
            and (x.stride(-1) == 1)
        ):
            width = x.size(-1)
            pitch = _row_stride(x, width)
            if (
                width in (2, 4)
                and pitch <= width * 2
                and (width == 2 or total >= 33554432)
                and ((total // width - 1) * pitch + width < 2147483648)
            ):
                out = torch.empty_like(
                    x, dtype=fp8_dtype, memory_format=torch.contiguous_format
                )
                _strided_pair_v582.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total,
                    width,
                    pitch,
                    x.dtype == torch.bfloat16,
                    scale_inv,
                    fp8_dtype == torch.float8_e4m3fn,
                    2048,
                    grid=((total + 4095) // 4096,),
                    num_warps=8 if total >= 8388608 else 4,
                    enable_fp_fusion=False,
                    warmup=False,
                )
                return out
        if (
            total >= 8388608
            and x.dtype == torch.float16
            and (0.00390625 <= scale_inv <= 0.5)
            and x.is_contiguous()
            and (total % 2 != 0 or x.data_ptr() % 4 != 0)
        ):
            fraction, exponent = math.frexp(scale_inv)
            if fraction == 0.5:
                out = torch.empty_like(x, dtype=fp8_dtype)
                block = 4096 if total >= 1048576 else 1024
                _half_split_scaled_v499.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total,
                    fp8_dtype == torch.float8_e4m3fn,
                    1 - exponent,
                    block // 2,
                    grid=((total + block - 1) // block,),
                    num_warps=8 if total >= 8388608 else 4,
                    enable_fp_fusion=False,
                    warmup=False,
                )
                return out
        if (
            16777216 <= total < 2147483648
            and x.dtype == torch.float16
            and (scale_inv == -1.0)
            and x.is_contiguous()
            and (
                total % 2 != 0
                and x.data_ptr() % 16 == 0
                or (total % 4096 == 0 and x.data_ptr() % 4 != 0)
            )
        ):
            address = x.data_ptr()
            out = torch.empty_like(x, dtype=fp8_dtype)
            if total % 4096 == 0:
                _half_scalar_v471_negative_v975.run(
                    address,
                    out.data_ptr(),
                    total,
                    fp8_dtype == torch.float8_e4m3fn,
                    4096,
                    grid=((total + 4095) // 4096,),
                    num_warps=8,
                    warmup=False,
                )
            else:
                _half_negative_split_v1014.run(
                    address,
                    out.data_ptr(),
                    total,
                    fp8_dtype == torch.float8_e4m3fn,
                    2048,
                    grid=((total + 4095) // 4096,),
                    num_warps=8,
                    warmup=False,
                )
            return out
        if (
            total >= 8388608
            and x.dtype == torch.float16
            and (scale_inv == 1.0)
            and x.is_contiguous()
        ):
            address = x.data_ptr()
            if total % 2 != 0 or address % 4 != 0:
                out = torch.empty_like(x, dtype=fp8_dtype)
                if address % 4 != 0 and total % 4096 == 0:
                    _half_scalar_v471.run(
                        address,
                        out.data_ptr(),
                        total,
                        fp8_dtype == torch.float8_e4m3fn,
                        4096,
                        grid=((total + 4095) // 4096,),
                        num_warps=8,
                        warmup=False,
                    )
                elif address % 4 != 0 and fp8_dtype == torch.float8_e4m3fn:
                    _half_scalar_fulltail_v501.run(
                        address,
                        out.data_ptr(),
                        total,
                        True,
                        4096,
                        grid=((total + 4095) // 4096,),
                        num_warps=8,
                        warmup=False,
                    )
                else:
                    _half_split_v494.run(
                        address,
                        out.data_ptr(),
                        total,
                        fp8_dtype == torch.float8_e4m3fn,
                        2048,
                        grid=((total + 4095) // 4096,),
                        num_warps=8,
                        warmup=False,
                    )
                return out
        if (
            total >= 8388608
            and total % 2 == 0
            and (x.dtype == torch.float16)
            and (0.00390625 <= scale_inv <= 0.5)
            and x.is_contiguous()
            and (x.storage_offset() % 2 == 0)
        ):
            fraction, exponent = math.frexp(scale_inv)
            if fraction == 0.5:
                out = torch.empty_like(x, dtype=fp8_dtype)
                block = 4096 if total >= 1048576 else 1024
                _half_pair_v490.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total // 2,
                    fp8_dtype == torch.float8_e4m3fn,
                    1 - exponent,
                    block // 2,
                    grid=((total + block - 1) // block,),
                    num_warps=8 if total >= 8388608 else 4,
                    enable_fp_fusion=False,
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
        ):
            out = torch.empty_like(x, dtype=fp8_dtype)
            block = 4096 if total >= 1048576 else 1024
            (
                _half_pair_v471
                if fp8_dtype == torch.float8_e4m3fn
                else _half_scalar_v471
            ).run(
                x.data_ptr(),
                out.data_ptr(),
                total // 2 if fp8_dtype == torch.float8_e4m3fn else total,
                fp8_dtype == torch.float8_e4m3fn,
                block // 2 if fp8_dtype == torch.float8_e4m3fn else block,
                grid=((total + block - 1) // block,),
                num_warps=8 if total >= 8388608 else 4,
                warmup=False,
            )
            return out
        if (
            8388608 <= total < 2147483648
            and scale_inv != 1.0
            and x.is_contiguous()
        ):
            if total >= 16777216 and (
                scale_inv == -1.0
                and x.dtype == torch.float16
                and (total % 2 == 0)
                and (x.data_ptr() % 4 == 0)
            ):
                out = torch.empty_like(x, dtype=fp8_dtype)
                e4 = fp8_dtype == torch.float8_e4m3fn
                (
                    _half_pair_v471_negative_v975
                    if e4
                    else _half_scalar_v471_negative_v975
                ).run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total // 2 if e4 else total,
                    e4,
                    2048 if e4 else 4096,
                    grid=((total + 4095) // 4096,),
                    num_warps=8,
                    warmup=False,
                )
                return out
            if (
                total >= 16777216
                and x.dtype == torch.bfloat16
                and (total % 2 == 0)
                and (x.data_ptr() % 4 == 0)
                and (0.00390625 <= abs(scale_inv) <= 256.0)
                and (scale_inv != 1.0)
            ):
                fraction, exponent = math.frexp(abs(scale_inv))
                if fraction == 0.5:
                    out = torch.empty_like(x, dtype=fp8_dtype)
                    _bf16_power_pair_v933.run(
                        x.data_ptr(),
                        out.data_ptr(),
                        total // 2,
                        fp8_dtype == torch.float8_e4m3fn,
                        exponent - 1,
                        scale_inv < 0.0,
                        2048,
                        grid=((total + 4095) // 4096,),
                        num_warps=8,
                        enable_fp_fusion=False,
                        warmup=False,
                    )
                    return out
            if (
                16777216 <= total
                and x.dtype == torch.float16
                and (total % 2 == 0)
                and (x.data_ptr() % 4 == 0)
                and (
                    2.0
                    <= abs(scale_inv)
                    <= (256.0 if fp8_dtype == torch.float8_e4m3fn else 2.0)
                )
            ):
                fraction, exponent = math.frexp(abs(scale_inv))
                if fraction == 0.5:
                    out = torch.empty_like(x, dtype=fp8_dtype)
                    _half_up_pair_v901.run(
                        x.data_ptr(),
                        out.data_ptr(),
                        total // 2,
                        fp8_dtype == torch.float8_e4m3fn,
                        exponent - 1,
                        scale_inv < 0.0,
                        2048,
                        grid=((total + 4095) // 4096,),
                        num_warps=8,
                        enable_fp_fusion=False,
                        warmup=False,
                    )
                    return out
            out = torch.empty_like(x, dtype=fp8_dtype)
            _general_split_v564.run(
                x.data_ptr(),
                out.data_ptr(),
                total,
                x.dtype == torch.bfloat16,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                2048,
                grid=((total + 4095) // 4096,),
                num_warps=8 if total >= 8388608 else 4,
                enable_fp_fusion=False,
                warmup=False,
            )
            return out
        if total >= 8388608:
            out = torch.empty_like(x, dtype=torch.uint8)
            if x.is_contiguous():
                _linear_kernel.run(
                    x,
                    out,
                    total,
                    scale_inv,
                    fp8_dtype == torch.float8_e4m3fn,
                    4096,
                    grid=((total + 4095) // 4096,),
                    warmup=False,
                    num_warps=8,
                )
            else:
                n = x.size(-1)
                stride = _row_stride(x, n)
                _kernel.run(
                    x,
                    out,
                    total,
                    n,
                    stride,
                    scale_inv,
                    fp8_dtype == torch.float8_e4m3fn,
                    4096,
                    grid=((total + 4095) // 4096,),
                    warmup=False,
                )
            return out.view(fp8_dtype)
    out = torch.empty_like(x, dtype=fp8_dtype)
    if total:
        block = 4096 if total >= 1048576 else 1024
        if x.is_contiguous():
            if scale_inv == 1.0 and fp8_dtype == torch.float8_e4m3fn:
                if total < 8388608 and total % block == 0:
                    if block == 1024:
                        fn = (
                            _full_bf16_1024_0
                            if x.dtype == torch.bfloat16
                            else _full_fp16_1024_0
                        )
                    else:
                        fn = (
                            _full_bf16_4096_0
                            if x.dtype == torch.bfloat16
                            else _full_fp16_4096_0
                        )
                        if (
                            total > 1048576
                            and total <= 4194304
                            and (x.dtype == torch.bfloat16)
                        ):
                            fn = _full_bf16_4096_1
                    fn.run(
                        x.data_ptr(),
                        out.data_ptr(),
                        grid=(total // block,),
                        warmup=False,
                    )
                else:
                    (
                        _unit_bf16 if x.dtype == torch.bfloat16 else _unit_fp16
                    ).run(
                        x.data_ptr(),
                        out.data_ptr(),
                        total,
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
                    grid=((total + block - 1) // block,),
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
                block,
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
    src = off // N * STRIDE + off % N
    v = (
        tl.load(X + src)
        if TOTAL % BLOCK == 0
        else tl.load(X + src, off < TOTAL, other=0)
    )
    if X.dtype.element_ty == tl.bfloat16 and SCALE == 1.0:
        code = (
            _encode_bf16_max(v, M, BIAS, 17376 if E4 else 18272)
            if TOTAL > 1048576 and TOTAL <= 4194304
            else _encode_bf16_magic(v, M, BIAS, 17376 if E4 else 18272)
        )
    else:
        v = v.to(tl.float32)
        if SCALE != 1.0:
            v = v * SCALE
        code = _encode(v, M, BIAS, MAX)
    if TOTAL % BLOCK == 0:
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code)
    else:
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
    v = (
        tl.load(X + src)
        if TOTAL % BLOCK == 0
        else tl.load(X + src, off < TOTAL, other=0)
    )
    if X.dtype.element_ty == tl.bfloat16 and SCALE == 1.0:
        code = (
            _encode_bf16_max(v, M, BIAS, 17376 if E4 else 18272)
            if TOTAL > 1048576 and TOTAL <= 4194304
            else _encode_bf16_magic(v, M, BIAS, 17376 if E4 else 18272)
        )
    else:
        v = v.to(tl.float32)
        if SCALE != 1.0:
            v = v * SCALE
        code = _encode(v, M, BIAS, MAX)
    if TOTAL % BLOCK == 0:
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code)
    else:
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code, off < TOTAL)


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
    )
    if X.dtype.element_ty == tl.bfloat16 and SCALE == 1.0:
        code = (
            _encode_bf16_max(v, M, BIAS, 17376 if E4 else 18272)
            if TOTAL > 1048576 and TOTAL <= 4194304
            else _encode_bf16_magic(v, M, BIAS, 17376 if E4 else 18272)
        )
    else:
        v = v.to(tl.float32)
        if SCALE != 1.0:
            v = v * SCALE
        code = _encode(v, M, BIAS, MAX)
    if TOTAL % BLOCK == 0:
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code)
    else:
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code, off < TOTAL)


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
    )
    if X.dtype.element_ty == tl.bfloat16 and SCALE == 1.0:
        code = (
            _encode_bf16_max(v, M, BIAS, 17376 if E4 else 18272)
            if TOTAL > 1048576 and TOTAL <= 4194304
            else _encode_bf16_magic(v, M, BIAS, 17376 if E4 else 18272)
        )
    else:
        v = v.to(tl.float32)
        if SCALE != 1.0:
            v = v * SCALE
        code = _encode(v, M, BIAS, MAX)
    if TOTAL % BLOCK == 0:
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code)
    else:
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + off, code, off < TOTAL)


@triton.jit
def _unit_bf16(X: "*bf16", Y: "*u8", TOTAL: tl.constexpr):
    BLOCK: tl.constexpr = 4096 if TOTAL >= 1048576 else 1024
    _linear_kernel_bf16(X, Y, TOTAL, 1.0, True, BLOCK)


@triton.jit
def _unit_fp16(X: "*fp16", Y: "*u8", TOTAL: tl.constexpr):
    BLOCK: tl.constexpr = 4096 if TOTAL >= 1048576 else 1024
    _linear_kernel_fp16(X, Y, TOTAL, 1.0, True, BLOCK)


@triton.jit
def _encode_bf16_max(
    v, M: tl.constexpr, BIAS: tl.constexpr, MAX16: tl.constexpr
):
    raw = v.to(tl.uint16, bitcast=True).to(tl.uint32)
    sign = (raw >> 8 & 128).to(tl.uint16)
    mag = tl.minimum(raw & 32767, MAX16)
    addend: tl.constexpr = (
        8388608.0 - (127 - BIAS << M) - 8388608.0 * 2.0 ** (M - 7)
    )
    normal = (
        (mag + (143 + M << 23)).to(tl.float32, bitcast=True) + addend
    ).to(tl.int32, bitcast=True)
    magic: tl.constexpr = 2.0 ** (24 - BIAS - M)
    clipped = tl.minimum(mag, 128 - BIAS << 7)
    sub = ((clipped << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.int32, bitcast=True
    )
    code = tl.maximum(normal.to(tl.int16), sub.to(tl.int16)).to(tl.uint16)
    return (code | sign).to(tl.uint8)


@triton.jit
def _full_bf16_1024_0(X: "*bf16", Y: "*u8"):
    off = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    value = tl.load(X + off)
    code = _encode_bf16_magic(value, 3, 7, 17376)
    tl.store(Y + off, code)


@triton.jit
def _full_fp16_1024_0(X: "*fp16", Y: "*u8"):
    off = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    value = tl.load(X + off)
    code = _encode(value.to(tl.float32), 3, 7, 448.0)
    tl.store(Y + off, code)


@triton.jit
def _full_bf16_4096_0(X: "*bf16", Y: "*u8"):
    off = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    value = tl.load(X + off)
    code = _encode_bf16_magic(value, 3, 7, 17376)
    tl.store(Y + off, code)


@triton.jit
def _full_fp16_4096_0(X: "*fp16", Y: "*u8"):
    off = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    value = tl.load(X + off)
    code = _encode(value.to(tl.float32), 3, 7, 448.0)
    tl.store(Y + off, code)


@triton.jit
def _full_bf16_4096_1(X: "*bf16", Y: "*u8"):
    off = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    value = tl.load(X + off)
    code = _encode_bf16_max(value, 3, 7, 17376)
    tl.store(Y + off, code)


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
def _half_split_scaled_v499(
    X: "*u16",
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
    source = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
    halves = tl.load(X + source, source < TOTAL // 2 * 2, other=0).to(
        tl.uint32
    )
    a, b = tl.split(tl.reshape(halves, (BLOCK, 2)))
    code = (
        _half_code_v490(a, E4, SHIFT) | _half_code_v490(b, E4, SHIFT) << 8
    ).to(tl.uint16)
    tl.store(Y + off, code, off < TOTAL // 2)
    if TOTAL % 2:
        raw = tl.load(X + TOTAL - 1, tile == 0, other=0).to(tl.uint32)
        final = _half_code_v490(raw, E4, SHIFT).to(tl.uint8)
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + TOTAL - 1, final, tile == 0)


@triton.jit
def _general_split_v564(
    X: "*u16",
    Y: "*u16",
    TOTAL: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pairs: tl.constexpr = TOTAL // 2
    lo = (
        tl.load(X + pair * 2)
        if pairs % BLOCK == 0
        else tl.load(X + pair * 2, pair < pairs, other=0)
    )
    hi = (
        tl.load(X + pair * 2 + 1)
        if pairs % BLOCK == 0
        else tl.load(X + pair * 2 + 1, pair < pairs, other=0)
    )
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
    if pairs % BLOCK == 0:
        tl.store(Y + pair, code)
    else:
        tl.store(Y + pair, code, pair < pairs)
    if TOTAL % 2 != 0:
        if tl.program_id(0) == 0:
            raw = tl.load(X + TOTAL - 1)
            v = (
                raw.to(tl.bfloat16, bitcast=True).to(tl.float32)
                if BF16
                else raw.to(tl.float16, bitcast=True).to(tl.float32)
            )
            v = v * SCALE
            last = _encode(
                v, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
            )
            tl.store(Y.to(tl.pointer_type(tl.uint8)) + TOTAL - 1, last)


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
def _strided_pair_v582(
    X: "*u16",
    Y: "*u16",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    logical = pair * 2
    source = logical // N * PITCH + logical % N
    lo = tl.load(X + source, logical < TOTAL // 2 * 2, other=0)
    hi = tl.load(
        X + source + (PITCH if N == 1 else 1),
        logical < TOTAL // 2 * 2,
        other=0,
    )
    a = (
        lo.to(tl.bfloat16, bitcast=True).to(tl.float32)
        if BF16
        else lo.to(tl.float16, bitcast=True).to(tl.float32)
    )
    b = (
        hi.to(tl.bfloat16, bitcast=True).to(tl.float32)
        if BF16
        else hi.to(tl.float16, bitcast=True).to(tl.float32)
    )
    a = a * SCALE
    b = b * SCALE
    ca = _encode(
        a, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
    ).to(tl.uint16)
    cb = _encode(
        b, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
    ).to(tl.uint16)
    ca = ca & 127 | lo >> 8 & 128 ^ (128 if SCALE < 0.0 else 0)
    cb = cb & 127 | hi >> 8 & 128 ^ (128 if SCALE < 0.0 else 0)
    tl.store(Y + pair, ca | cb << 8, pair < TOTAL // 2)
    if TOTAL % 2 != 0:
        if tl.program_id(0) == 0:
            tail_source = (TOTAL - 1) // N * PITCH + (TOTAL - 1) % N
            raw = tl.load(X + tail_source)
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
            tl.store(
                Y.to(tl.pointer_type(tl.uint8)) + TOTAL - 1, code.to(tl.uint8)
            )


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
def _bf16_code_v648(raw, E4: tl.constexpr):
    v = raw.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    return _encode_bf16_magic(
        v, 3 if E4 else 2, 7 if E4 else 15, 17376 if E4 else 18272
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


@triton.jit
def _encode_half_pair_direct_v486_mx664(u, v, E4: tl.constexpr):
    a = tl.minimum(u.to(tl.int32) & 32767, 24320 if E4 else 31488)
    b = tl.minimum(v.to(tl.int32) & 32767, 24320 if E4 else 31488)
    signs = u.to(tl.uint16) >> 8 & 128 | v.to(tl.uint16) & 32768
    an = (a + (1199570944 if E4 else 1191182336)).to(tl.float32, bitcast=True)
    bn = (b + (1199570944 if E4 else 1191182336)).to(tl.float32, bitcast=True)
    na = (
        (an + (8323008.0 if E4 else 8355840.0))
        .to(tl.int32, bitcast=True)
        .to(tl.int16)
    )
    nb = (
        (bn + (8323008.0 if E4 else 8355840.0))
        .to(tl.int32, bitcast=True)
        .to(tl.int16)
    )
    if E4:
        ac = tl.minimum(a, 9216)
        bc = tl.minimum(b, 9216)
        af = (ac + 114688 << 13).to(tl.float32, bitcast=True)
        bf = (bc + 114688 << 13).to(tl.float32, bitcast=True)
        sa = (af + 16384.0).to(tl.int32, bitcast=True).to(tl.int16)
        sb = (bf + 16384.0).to(tl.int32, bitcast=True).to(tl.int16)
        na = tl.maximum(na, sa)
        nb = tl.maximum(nb, sb)
    return na.to(tl.uint16) | nb.to(tl.uint16) << 8 | signs


@triton.jit
def _encode_pair_direct_v486_mx664(u, v, E4: tl.constexpr):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX16: tl.constexpr = 17376 if E4 else 18272
    addend: tl.constexpr = (
        8388608.0 - (127 - BIAS << M) - 8388608.0 * 2.0 ** (M - 7)
    )
    magic = tl.full((), 2.0 ** (24 - BIAS - M), tl.float32)
    am = u.to(tl.int32) & 32767
    bh = v.to(tl.int32)
    a = tl.minimum(am, MAX16)
    bm = bh & 32767
    b = tl.minimum(bm, MAX16)
    signs = u.to(tl.uint16) >> 8 & 128 | v.to(tl.uint16) & 32768
    an = (a + (143 + M << 23)).to(tl.float32, bitcast=True)
    bn = (b + (143 + M << 23)).to(tl.float32, bitcast=True)
    na = (an + addend).to(tl.int32, bitcast=True)
    nb = (bn + addend).to(tl.int32, bitcast=True)
    ac = tl.minimum(a, 128 - BIAS << 7)
    bc = tl.minimum(b, 128 - BIAS << 7)
    af = (ac << 16).to(tl.float32, bitcast=True)
    bf = (bc << 16).to(tl.float32, bitcast=True)
    sa = (af + magic).to(tl.int32, bitcast=True)
    sb = (bf + magic).to(tl.int32, bitcast=True)
    na16 = na.to(tl.int16)
    nb16 = nb.to(tl.int16)
    sa16 = sa.to(tl.int16)
    sb16 = sb.to(tl.int16)
    ca = tl.maximum(na16, sa16).to(tl.uint16)
    cb = tl.maximum(nb16, sb16).to(tl.uint16)
    code = ca | cb << 8 | signs
    return code


@triton.jit
def _general_code_hw_v571_mx664(v, E4: tl.constexpr):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX: tl.constexpr = 448.0 if E4 else 57344.0
    raw = v.to(tl.int32, bitcast=True)
    mag = tl.minimum(
        raw & 2147483647,
        tl.full((), MAX, tl.float32).to(tl.int32, bitcast=True),
    )
    normal = (
        mag + ((1 << 22 - M) - 1 - (127 - BIAS << 23)) + (mag >> 23 - M & 1)
        >> 23 - M
    )
    clipped = tl.minimum(mag, 128 - BIAS << 23).to(tl.float32, bitcast=True)
    sub = (clipped + (16384.0 if E4 else 128.0)).to(tl.int32, bitcast=True)
    return tl.maximum(normal.to(tl.int16), sub.to(tl.int16)).to(tl.uint16)


@triton.jit
def _hw_strided_pair_v606_mx664(
    X: "*i16",
    Y: "*i16",
    XT: "*i16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    PAIRS: tl.constexpr = TOTAL // 2
    SPAN: tl.constexpr = triton.next_power_of_2(
        (BLOCK * 2 - 1) // N * PITCH + (BLOCK * 2 - 1) % N + 1
    )
    LAST: tl.constexpr = (TOTAL - 1) // N * PITCH + (TOTAL - 1) % N + 1
    tile = tl.program_id(0)
    pair = tile * BLOCK + tl.arange(0, BLOCK)
    base = tile * BLOCK * 2 // N * PITCH + tile * BLOCK * 2 % N
    span_offset = tl.arange(0, SPAN)
    physical = base + span_offset
    values = tl.load(X + physical, physical < LAST, other=0)
    values = values.to(tl.bfloat16 if BF16 else tl.float16, bitcast=True)
    if (PITCH == 2 * N and N < 2 * BLOCK) and N >= 512:
        valid, padding = tl.split(
            tl.permute(tl.reshape(values, (BLOCK * 2 // N, 2, N)), (0, 2, 1))
        )
        lo, hi = tl.split(tl.reshape(valid, (BLOCK, 2)))
        lo = lo.to(tl.int16, bitcast=True)
        hi = hi.to(tl.int16, bitcast=True)
    else:
        local = tl.arange(0, BLOCK) * 2
        index = local // N * PITCH + local % N
        lo = tl.gather(values, index, 0).to(tl.int16, bitcast=True)
        hi = tl.gather(values, index + (PITCH if N == 1 else 1), 0).to(
            tl.int16, bitcast=True
        )
    if SCALE == 1.0 and TOTAL >= 16777216:
        if BF16:
            code = _encode_pair_direct_v486_mx664(lo, hi, E4).to(tl.int16)
        else:
            code = _encode_half_pair_direct_v486_mx664(lo, hi, E4).to(tl.int16)
    else:
        if BF16:
            a = lo.to(tl.bfloat16, bitcast=True).to(tl.float32)
            b = hi.to(tl.bfloat16, bitcast=True).to(tl.float32)
        else:
            a = lo.to(tl.float16, bitcast=True).to(tl.float32)
            b = hi.to(tl.float16, bitcast=True).to(tl.float32)
        a = a * SCALE
        b = b * SCALE
        ca = _general_code_hw_v571_mx664(a, E4)
        cb = _general_code_hw_v571_mx664(b, E4)
        signs = (lo >> 8 & 128 | hi & -32768).to(tl.uint16) ^ (
            32896 if SCALE < 0.0 else 0
        )
        code = (ca | cb << 8 | signs).to(tl.int16)
    if PAIRS % BLOCK == 0:
        tl.store(Y + pair, code)
    else:
        tl.store(Y + pair, code, pair < PAIRS)
    if TOTAL % 2 != 0:
        r = tl.arange(0, 128)
        raw = tl.load(XT + r, r == 0, other=0)
        v = (
            raw.to(tl.bfloat16, bitcast=True).to(tl.float32)
            if BF16
            else raw.to(tl.float16, bitcast=True).to(tl.float32)
        )
        v = v * SCALE
        last = _general_code_hw_v571_mx664(v, E4) | raw >> 8 & 128 ^ (
            128 if SCALE < 0.0 else 0
        )
        tl.store(YT + r, last.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


@triton.jit
def _sparse_pair_v701(
    X: "*i16",
    Y: "*i16",
    XT: "*i16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    PAIRS: tl.constexpr = TOTAL // 2
    tile = tl.program_id(0)
    pair = tile * BLOCK + tl.arange(0, BLOCK)
    logical = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
    source = logical // N * PITCH + logical % N
    values = tl.load(X + source, logical < TOTAL, other=0)
    values = values.to(tl.bfloat16 if BF16 else tl.float16, bitcast=True)
    lo, hi = tl.split(tl.reshape(values, (BLOCK, 2)))
    lo = lo.to(tl.int16, bitcast=True)
    hi = hi.to(tl.int16, bitcast=True)
    if SCALE == 1.0 and TOTAL >= 16777216:
        if BF16:
            code = _encode_pair_direct_v486_mx664(lo, hi, E4).to(tl.int16)
        else:
            code = _encode_half_pair_direct_v486_mx664(lo, hi, E4).to(tl.int16)
    else:
        if BF16:
            a = lo.to(tl.bfloat16, bitcast=True).to(tl.float32)
            b = hi.to(tl.bfloat16, bitcast=True).to(tl.float32)
        else:
            a = lo.to(tl.float16, bitcast=True).to(tl.float32)
            b = hi.to(tl.float16, bitcast=True).to(tl.float32)
        a = a * SCALE
        b = b * SCALE
        ca = _general_code_hw_v571_mx664(a, E4)
        cb = _general_code_hw_v571_mx664(b, E4)
        signs = (lo >> 8 & 128 | hi & -32768).to(tl.uint16) ^ (
            32896 if SCALE < 0.0 else 0
        )
        code = (ca | cb << 8 | signs).to(tl.int16)
    if PAIRS % BLOCK == 0:
        tl.store(Y + pair, code)
    else:
        tl.store(Y + pair, code, pair < PAIRS)
    if TOTAL % 2 != 0:
        r = tl.arange(0, 128)
        raw = tl.load(XT + r, r == 0, other=0)
        v = (
            raw.to(tl.bfloat16, bitcast=True).to(tl.float32)
            if BF16
            else raw.to(tl.float16, bitcast=True).to(tl.float32)
        )
        v = v * SCALE
        last = _general_code_hw_v571_mx664(v, E4) | raw >> 8 & 128 ^ (
            128 if SCALE < 0.0 else 0
        )
        tl.store(YT + r, last.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


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
def _half_up_code_v901(raw, E4: tl.constexpr, Q: tl.constexpr):
    sign = raw >> 8 & 128
    mag = tl.minimum(raw & 32767, (24320 if E4 else 31488) - Q * 1024)
    shifted = mag + Q * 1024
    normal = (
        (shifted | (143 if E4 else 142) << 23).to(tl.float32, bitcast=True)
        + (8323008.0 if E4 else 8355840.0)
    ).to(tl.uint32, bitcast=True)
    if E4:
        sub = (
            (mag + (112 + Q) * 1024 << 13).to(tl.float32, bitcast=True)
            + 16384.0
        ).to(tl.uint32, bitcast=True)
        code = tl.where(mag < 9216 - Q * 1024, sub, normal)
        if Q > 4:
            halfsub = (
                (mag | 135 + Q << 23).to(tl.float32, bitcast=True)
                + (8388608.0 - 2.0 ** (8 + Q))
            ).to(tl.uint32, bitcast=True)
            code = tl.where(mag < 1024, halfsub, code)
    else:
        mapped = mag
        if Q >= 2:
            mapped += tl.minimum(mag, 512)
        if Q >= 3:
            mapped += tl.minimum(mag, 256) * 2
        sub = (
            (mapped | 143 << 23).to(tl.float32, bitcast=True) + 8323072.0
        ).to(tl.uint32, bitcast=True)
        code = tl.where(mag < 1024, sub, normal)
    return code & 127 | sign


@triton.jit
def _half_up_pair_v901(
    X: "*u32",
    Y: "*u16",
    PAIRS: tl.constexpr,
    E4: tl.constexpr,
    Q: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if PAIRS % BLOCK == 0:
        raw = tl.load(X + off)
    else:
        raw = tl.load(X + off, off < PAIRS, other=0)
    code = (
        _half_up_code_v901(raw, E4, Q)
        | _half_up_code_v901(raw >> 16, E4, Q) << 8
    ) ^ (
        tl.where(raw & 32767 > 31744, 0, 128)
        | tl.where(raw >> 16 & 32767 > 31744, 0, 32768)
        if NEG
        else 0
    )
    if PAIRS % BLOCK == 0:
        tl.store(Y + off, code.to(tl.uint16))
    else:
        tl.store(Y + off, code.to(tl.uint16), off < PAIRS)


@triton.jit
def _bf16_power_code_v933(raw, E4: tl.constexpr, Q: tl.constexpr):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    mag = tl.maximum(
        tl.minimum(
            (raw & 32767).to(tl.int32), (17376 if E4 else 18272) - Q * 128
        )
        + Q * 128,
        0,
    ).to(tl.uint32)
    offset: tl.constexpr = -(127 - BIAS) << M & 255
    addend: tl.constexpr = 8388608.0 + offset - 8388608.0 * 2.0 ** (M - 7)
    normal = ((mag | 143 + M << 23).to(tl.float32, bitcast=True) + addend).to(
        tl.uint32, bitcast=True
    )
    sub = (
        (mag << 16).to(tl.float32, bitcast=True) + 2.0 ** (24 - BIAS - M)
    ).to(tl.uint32, bitcast=True)
    code = tl.where(mag < 128 - BIAS << 7, sub, normal)
    return code & 127 | raw >> 8 & 128


@triton.jit
def _bf16_power_pair_v933(
    X: "*u32",
    Y: "*u16",
    PAIRS: tl.constexpr,
    E4: tl.constexpr,
    Q: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw = (
        tl.load(X + off)
        if PAIRS % BLOCK == 0
        else tl.load(X + off, off < PAIRS, other=0)
    )
    code = (
        _bf16_power_code_v933(raw, E4, Q)
        | _bf16_power_code_v933(raw >> 16, E4, Q) << 8
    )
    if NEG:
        if Q == 0:
            code ^= 32896
        else:
            code ^= tl.where(raw & 32767 > 32640, 0, 128) | tl.where(
                raw >> 16 & 32767 > 32640, 0, 32768
            )
    if PAIRS % BLOCK == 0:
        tl.store(Y + off, code.to(tl.uint16))
    else:
        tl.store(Y + off, code.to(tl.uint16), off < PAIRS)


@triton.jit
def _sparse_power_v958(
    X: "*i16",
    Y: "*i16",
    XT: "*i16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
    Q: tl.constexpr,
):
    PAIRS: tl.constexpr = TOTAL // 2
    tile = tl.program_id(0)
    pair = tile * BLOCK + tl.arange(0, BLOCK)
    logical = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
    source = logical // N * PITCH + logical % N
    values = tl.load(X + source, logical < TOTAL, other=0)
    values = values.to(tl.bfloat16 if BF16 else tl.float16, bitcast=True)
    lo, hi = tl.split(tl.reshape(values, (BLOCK, 2)))
    lo = lo.to(tl.int16, bitcast=True)
    hi = hi.to(tl.int16, bitcast=True)
    code = (
        (
            _bf16_power_code_v933(lo.to(tl.uint16).to(tl.uint32), E4, Q)
            | _bf16_power_code_v933(hi.to(tl.uint16).to(tl.uint32), E4, Q) << 8
        )
        ^ (32896 if SCALE < 0.0 else 0)
    ).to(tl.int16)
    if PAIRS % BLOCK == 0:
        tl.store(Y + pair, code)
    else:
        tl.store(Y + pair, code, pair < PAIRS)
    if TOTAL % 2 != 0:
        r = tl.arange(0, 128)
        raw = tl.load(XT + r, r == 0, other=0)
        v = (
            raw.to(tl.bfloat16, bitcast=True).to(tl.float32)
            if BF16
            else raw.to(tl.float16, bitcast=True).to(tl.float32)
        )
        v = v * SCALE
        last = _general_code_hw_v571_mx664(v, E4) | raw >> 8 & 128 ^ (
            128 if SCALE < 0.0 else 0
        )
        tl.store(YT + r, last.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


@triton.jit
def _half_pair_v471_negative_v975(
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
    ) ^ 32896
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code)
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _half_scalar_v471_negative_v975(
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
    code = _half_code_v471(raw, E4).to(tl.uint8) ^ 128
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code)
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _negative_split_v994(
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
    code = (_bf16_code_v648(a, E4) | _bf16_code_v648(b, E4) << 8).to(
        tl.uint16
    ) ^ 32896
    tl.store(Y + off, code, off < TOTAL // 2)
    if TOTAL % 2:
        raw = tl.load(X + TOTAL - 1, tile == 0, other=0).to(tl.uint32)
        final = _bf16_code_v648(raw, E4).to(tl.uint8) ^ 128
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + TOTAL - 1, final, tile == 0)


@triton.jit
def _half_negative_split_v1014(
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
    code = (_half_code_v471(a, E4) | _half_code_v471(b, E4) << 8).to(
        tl.uint16
    ) ^ 32896
    tl.store(Y + off, code, off < TOTAL // 2)
    if TOTAL % 2:
        raw = tl.load(X + TOTAL - 1, tile == 0, other=0).to(tl.uint32)
        final = _half_code_v471(raw, E4).to(tl.uint8) ^ 128
        tl.store(Y.to(tl.pointer_type(tl.uint8)) + TOTAL - 1, final, tile == 0)


__all__ = ["fp8_quantize"]
