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


@triton.jit
def _encode_bf16_magic(
    v, M: tl.constexpr, BIAS: tl.constexpr, MAX16: tl.constexpr
):
    raw = v.to(tl.uint16, bitcast=True).to(tl.int32)
    sign = raw >> 8 & 128
    mag = tl.minimum(raw & 32767, MAX16)
    offset: tl.constexpr = -(127 - BIAS << M) & 255
    addend: tl.constexpr = 8388608.0 + offset - 8388608.0 * 2.0 ** (M - 7)
    normal = ((mag | 143 + M << 23).to(tl.float32, bitcast=True) + addend).to(
        tl.int32, bitcast=True
    )
    magic = tl.full((), 2.0 ** (24 - BIAS - M), tl.float32)
    sub = ((mag << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.int32, bitcast=True
    )
    code = tl.where(mag < 128 - BIAS << 7, sub, normal)
    return (code | sign).to(tl.uint8)


@triton.jit
def _encode(v, M: tl.constexpr, BIAS: tl.constexpr, MAX: tl.constexpr):
    raw = v.to(tl.int32, bitcast=True)
    sign = raw >> 24 & 128
    mag = tl.minimum(
        raw & 2147483647,
        tl.full((), MAX, tl.float32).to(tl.int32, bitcast=True),
    )
    normal = (
        mag
        + tl.full((), (1 << 22 - M) - 1 - (127 - BIAS << 23), tl.int32)
        + (mag >> 23 - M & 1)
        >> 23 - M
    )
    magic = tl.full((), 2.0 ** (24 - BIAS - M), tl.float32)
    sub = (mag.to(tl.float32, bitcast=True) + magic).to(tl.int32, bitcast=True)
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
    PADDED: tl.constexpr = (TOTAL + BLOCK - 1) // BLOCK * BLOCK
    WIDE_INDEX: tl.constexpr = (
        PADDED - 1 > 2147483647
        or (PADDED - 1) // N * STRIDE + N - 1 > 2147483647
    )
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX: tl.constexpr = 448.0 if E4 else 57344.0
    for tile in range(
        tl.program_id(0).to(tl.int64) if WIDE_INDEX else tl.program_id(0),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        src = (
            tile * BLOCK // N * STRIDE + tile * BLOCK % N + tl.arange(0, BLOCK)
            if STRIDE != N
            else off
        )
        if TOTAL % BLOCK == 0:
            v = tl.load(X + src)
        else:
            v = tl.load(X + src, off < TOTAL, other=0)
        if X.dtype.element_ty == tl.bfloat16 and SCALE == 1.0:
            code = _encode_bf16_magic(v, M, BIAS, 17376 if E4 else 18272)
        elif X.dtype.element_ty == tl.float16 and SCALE == 1.0 and (not E4):
            code = _encode_half_direct(v, E4)
        else:
            v = v.to(tl.float32)
            if SCALE != 1.0:
                v = v * SCALE
            code = _encode(v, M, BIAS, MAX)
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code)
        else:
            tl.store(Y + off, code, off < TOTAL)


def fp8_quantize(x, scale_inv, fp8_dtype):
    if scale_inv == 0.0:
        out = torch.empty(x.shape, dtype=fp8_dtype, device=x.device)
        total = x.numel()
        if total:
            if (
                262144 <= total < 2147483648
                and (not x.is_contiguous())
                and (x.stride(-1) == 1)
            ):
                width = x.size(-1)
                pitch = _row_stride(x, width)
                if (
                    1 <= width <= 8
                    and width & width - 1 == 0
                    and (2 * width < pitch <= 8 * width)
                    and ((total // width - 1) * pitch + width < 2147450880)
                ):
                    block = (
                        (2048 if width == 1 else 1024)
                        if total >= 16777216
                        else 256
                    )
                    tail = (total - 1) // width * pitch + (total - 1) % width
                    _hw_zero_span_v777.run(
                        x.data_ptr(),
                        out.data_ptr(),
                        x.data_ptr() + 2 * tail,
                        out.data_ptr() + total - 1,
                        total,
                        width,
                        pitch,
                        x.dtype == torch.bfloat16,
                        math.copysign(1.0, scale_inv) < 0.0,
                        block,
                        grid=(
                            min((total + 2 * block - 1) // (2 * block), 40),
                        ),
                        warmup=False,
                    )
                    return out
            if (
                1048576 <= total < 2147483648
                and (not x.is_contiguous())
                and (x.stride(-1) == 1)
            ):
                width = x.size(-1)
                pitch = _row_stride(x, width)
                if (
                    16 <= width <= 1024
                    and width & width - 1 == 0
                    and ((total // width - 1) * pitch + width < 2147483648)
                ) and pitch > 2 * width:
                    block = 2048 if total >= 16777216 else 1024
                    _hw_zero_matrix_v734.run(
                        x.data_ptr(),
                        out.data_ptr(),
                        total,
                        width,
                        pitch,
                        math.copysign(1.0, scale_inv) < 0.0,
                        x.dtype == torch.bfloat16,
                        block,
                        grid=(
                            min((total + 2 * block - 1) // (2 * block), 40),
                        ),
                        warmup=False,
                    )
                    return out
            if (
                4194304 <= total < 2147483648
                and total % 2 == 0
                and x.is_contiguous()
                and (x.storage_offset() % 2 == 0)
            ):
                block = 4096 if total >= 33554432 else 2048
                _zero_pair_hw_v592.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total // 2,
                    math.copysign(1.0, scale_inv) < 0.0,
                    x.dtype == torch.bfloat16,
                    block,
                    grid=(min((total + block * 2 - 1) // (block * 2), 40),),
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
                grid=(min((total + 1023) // 1024, 40),),
                warmup=False,
            )
        return out
    out = torch.empty_like(
        x, dtype=fp8_dtype, memory_format=torch.contiguous_format
    )
    total = x.numel()
    if (
        16384 <= total < 2147483648
        and (not x.is_contiguous())
        and (x.stride(-1) == 1)
    ):
        width = x.size(-1)
        pitch = _row_stride(x, width)
        if (
            1 <= width <= 8
            and width & width - 1 == 0
            and (2 * width < pitch <= (8 if width == 1 else 32) * width)
            and ((total // width - 1) * pitch + width < 2147450880)
        ) and total >= (16384 if width == 1 else 65536):
            block = (
                (2048 if pitch <= 4 and total >= 16777216 else 1024)
                if width == 1 and total >= 4194304
                else (
                    512
                    if width >= 2
                    and pitch <= 8 * width
                    and (total >= 16777216)
                    else 256
                )
            )
            tail = (total - 1) // width * pitch + (total - 1) % width
            _hw_strided_pair_v606.run(
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
                block,
                grid=(min((total + 2 * block - 1) // (2 * block), 40),),
                enable_fp_fusion=False,
                warmup=False,
            )
            return out
    if (
        1048576 <= total < 2147483648
        and (not x.is_contiguous())
        and (x.stride(-1) == 1)
    ):
        width = x.size(-1)
        pitch = _row_stride(x, width)
        if (
            width & width - 1 == 0
            and (total // width - 1) * pitch + width < 2147483648
            and (
                16 <= width <= 1024
                and pitch > 2 * width
                and (width <= 128 or total >= 16777216)
                or (64 <= width <= 128 and pitch == 2 * width)
                or (width == 2048 and total >= 4194304 and (pitch > 2 * width))
            )
        ):
            tail_offset = (total - 1) // width * pitch + (total - 1) % width
            _hw_sparse_v720.run(
                x.data_ptr(),
                out.data_ptr(),
                x.data_ptr() + tail_offset * 2,
                out.data_ptr() + total - 1,
                total,
                width,
                pitch,
                x.dtype == torch.bfloat16,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                1024,
                grid=(min((total + 2047) // 2048, 40),),
                enable_fp_fusion=False,
                warmup=False,
            )
            return out
    if (
        1048576 <= total < 2147483648
        and (not x.is_contiguous())
        and (x.stride(-1) == 1)
    ):
        width = x.size(-1)
        pitch = _row_stride(x, width)
        if (
            1 <= width <= 2048
            and width & width - 1 == 0
            and (pitch <= 2 * width)
            and ((total // width - 1) * pitch + width < 2147450880)
        ) and (width <= 128 or total >= 16777216):
            tail_offset = (total - 1) // width * pitch + (total - 1) % width
            _hw_strided_pair_v606.run(
                x.data_ptr(),
                out.data_ptr(),
                x.data_ptr() + tail_offset * 2,
                out.data_ptr() + total - 1,
                total,
                width,
                pitch,
                x.dtype == torch.bfloat16,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                2048 if total >= 16777216 and scale_inv != 1.0 else 1024,
                grid=(
                    min(
                        (
                            total
                            + 2
                            * (
                                2048
                                if total >= 16777216 and scale_inv != 1.0
                                else 1024
                            )
                            - 1
                        )
                        // (
                            2
                            * (
                                2048
                                if total >= 16777216 and scale_inv != 1.0
                                else 1024
                            )
                        ),
                        40,
                    ),
                ),
                enable_fp_fusion=False,
                warmup=False,
            )
            return out
    if (
        8388608 <= total < 2147483648
        and x.dtype == torch.float16
        and (0.00390625 <= scale_inv <= 0.5)
        and x.is_contiguous()
        and (total % 2 != 0 or x.storage_offset() % 2 != 0)
    ):
        fraction, exponent = math.frexp(scale_inv)
        if fraction == 0.5:
            if total % 2:
                _hw_scaled_split_odd_v512.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    x.data_ptr() + (total - 1) * 2,
                    out.data_ptr() + total - 1,
                    total // 2,
                    fp8_dtype == torch.float8_e4m3fn,
                    1 - exponent,
                    4096,
                    grid=(min((total + 8191) // 8192, 40),),
                    enable_fp_fusion=False,
                    warmup=False,
                )
            else:
                _hw_scaled_split_kernel_v512.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total // 2,
                    fp8_dtype == torch.float8_e4m3fn,
                    1 - exponent,
                    4096,
                    grid=(min((total + 8191) // 8192, 40),),
                    enable_fp_fusion=False,
                    warmup=False,
                )
            return out
    if (
        total >= 1048576
        and total % 2 == 0
        and (x.dtype == torch.float16)
        and (0.00390625 <= scale_inv <= 0.5)
        and x.is_contiguous()
        and (x.storage_offset() % 2 == 0)
    ):
        fraction, exponent = math.frexp(scale_inv)
        if fraction == 0.5:
            _half_scaled_pair_kernel_v507.run(
                x.data_ptr(),
                out.data_ptr(),
                total // 2,
                fp8_dtype == torch.float8_e4m3fn,
                1 - exponent,
                4096,
                grid=(min((total + 8191) // 8192, 40),),
                enable_fp_fusion=False,
                warmup=False,
            )
            return out
    if (
        32768 <= total < 2097152
        and x.dtype == torch.bfloat16
        and (scale_inv != 1.0)
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 0)
    ):
        _bf16_general_packed_v1012.run(
            x.data_ptr(),
            out.data_ptr(),
            x.data_ptr() + (total - 1) * 2,
            out.data_ptr() + total - 1,
            total,
            scale_inv,
            fp8_dtype == torch.float8_e4m3fn,
            1024,
            True,
            grid=(min((total + 2047) // 2048, 40),),
            enable_fp_fusion=False,
            warmup=False,
        )
        return out
    if (
        2097152 <= total < 2147483648
        and scale_inv != 1.0
        and x.is_contiguous()
    ):
        if (
            x.dtype == torch.bfloat16
            and x.data_ptr() % 4 != 0
            and (0.00390625 <= abs(scale_inv) <= 256.0)
            and (abs(scale_inv) != 1.0)
        ):
            fraction, exponent = math.frexp(abs(scale_inv))
            if fraction == 0.5:
                _hw_split_power_v992.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    x.data_ptr() + (total - 1) * 2,
                    out.data_ptr() + total - 1,
                    total,
                    fp8_dtype == torch.float8_e4m3fn,
                    exponent - 1,
                    scale_inv,
                    4096,
                    enable_fp_fusion=False,
                    grid=(min((total + 8191) // 8192, 40),),
                    warmup=False,
                )
                return out
        if (
            x.dtype == torch.bfloat16
            and x.data_ptr() % 4 == 0
            and (0.00390625 <= abs(scale_inv) <= 256.0)
            and (abs(scale_inv) != 1.0)
        ):
            fraction, exponent = math.frexp(abs(scale_inv))
            if fraction == 0.5:
                if total % 2:
                    _hw_power_odd_v986.run(
                        x.data_ptr(),
                        out.data_ptr(),
                        x.data_ptr() + (total - 1) * 2,
                        out.data_ptr() + total - 1,
                        total // 2,
                        fp8_dtype == torch.float8_e4m3fn,
                        4096,
                        exponent - 1,
                        scale_inv,
                        enable_fp_fusion=False,
                        grid=(min((total + 8191) // 8192, 40),),
                        warmup=False,
                    )
                    return out
                _hw_packed_power_v968.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total // 2,
                    fp8_dtype == torch.float8_e4m3fn,
                    4096,
                    exponent - 1,
                    scale_inv < 0.0,
                    enable_fp_fusion=False,
                    grid=(min((total + 8191) // 8192, 40),),
                    warmup=False,
                )
                return out
        if (
            scale_inv == -1.0
            and total >= 16777216
            and (total % 2 != 0 or x.data_ptr() % 4 != 0)
        ):
            _negative_tail_v987.run(
                x.data_ptr(),
                out.data_ptr(),
                x.data_ptr() + (total - 1) * 2,
                out.data_ptr() + total - 1,
                total,
                x.dtype == torch.bfloat16,
                fp8_dtype == torch.float8_e4m3fn,
                4096,
                x.data_ptr() % 4 == 0,
                x.data_ptr(),
                enable_fp_fusion=False,
                grid=(min((total + 8191) // 8192, 40),),
                warmup=False,
            )
            return out
        if (
            scale_inv == -1.0
            and total >= 16777216
            and (total % 2 == 0)
            and (x.data_ptr() % 4 == 0)
        ):
            _negative_unit_pair_v965.run(
                x.data_ptr(),
                out.data_ptr(),
                total // 2,
                fp8_dtype == torch.float8_e4m3fn,
                4096,
                x.dtype == torch.bfloat16,
                enable_fp_fusion=False,
                grid=(min((total + 8191) // 8192, 40),),
                warmup=False,
            )
            return out
        if x.dtype == torch.bfloat16 and x.data_ptr() % 4 == 0:
            _bf16_general_packed_v1012.run(
                x.data_ptr(),
                out.data_ptr(),
                x.data_ptr() + (total - 1) * 2,
                out.data_ptr() + total - 1,
                total,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                4096,
                grid=(min((total + 8191) // 8192, 40),),
                enable_fp_fusion=False,
                warmup=False,
            )
            return out
        if (
            x.dtype == torch.float16
            and x.data_ptr() % 4 == 0
            and (total >= 16777216)
            and (total % 8192 == 0)
        ):
            _half_general_packed_v1019.run(
                x.data_ptr(),
                out.data_ptr(),
                x.data_ptr() + (total - 1) * 2,
                out.data_ptr() + total - 1,
                total,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                4096,
                grid=(min((total + 8191) // 8192, 40),),
                enable_fp_fusion=False,
                warmup=False,
            )
            return out
        if total >= 33554432 and total // 2 % 4096 == 0:
            _general_serial_full_v612.run(
                x.data_ptr(),
                out.data_ptr(),
                x.data_ptr() + (total - 1) * 2,
                out.data_ptr() + total - 1,
                total,
                x.dtype == torch.bfloat16,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                4096,
                grid=(min((total + 8191) // 8192, 40),),
                enable_fp_fusion=False,
                warmup=False,
            )
            return out
        if total // 2 % 2048 != 0:
            block = 4096
            _general_fulltail_v1000.run(
                x.data_ptr(),
                out.data_ptr(),
                x.data_ptr() + (total - 1) * 2,
                out.data_ptr() + total - 1,
                total,
                x.dtype == torch.bfloat16,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                block,
                grid=(min((total + block * 2 - 1) // (block * 2), 40),),
                enable_fp_fusion=False,
                warmup=False,
            )
            return out
        _general_split_hw_v573.run(
            x.data_ptr(),
            out.data_ptr(),
            x.data_ptr() + (total - 1) * 2,
            out.data_ptr() + total - 1,
            total,
            x.dtype == torch.bfloat16,
            scale_inv,
            fp8_dtype == torch.float8_e4m3fn,
            2048,
            grid=(min((total + 4095) // 4096, 40),),
            enable_fp_fusion=False,
            warmup=False,
        )
        return out
    if total:
        contiguous = x.is_contiguous()
        if (
            8388608 <= total < 2147483648
            and scale_inv == 1.0
            and contiguous
            and (x.storage_offset() % 2 == 1)
        ):
            if total % 2:
                fn = (
                    _pair_kernel_split_v485_odd_v489
                    if x.dtype == torch.bfloat16
                    else _half_pair_kernel_split_v485_odd_v489
                )
                fn.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    x.data_ptr() + (total - 1) * 2,
                    out.data_ptr() + total - 1,
                    total // 2,
                    fp8_dtype == torch.float8_e4m3fn,
                    4096,
                    enable_fp_fusion=False,
                    grid=(40,),
                    warmup=False,
                )
            else:
                fn = (
                    _pair_kernel_split_v485
                    if x.dtype == torch.bfloat16
                    else _half_pair_kernel_split_v485
                )
                fn.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total // 2,
                    fp8_dtype == torch.float8_e4m3fn,
                    4096,
                    enable_fp_fusion=False,
                    grid=(40,),
                    warmup=False,
                )
            return out
        if (
            8388609 <= total < 2147483648
            and total % 2 == 1
            and (scale_inv == 1.0)
            and contiguous
            and (x.storage_offset() % 2 == 0)
        ):
            fn = (
                _pair_kernel_odd_v479
                if x.dtype == torch.bfloat16
                else _half_pair_kernel_odd_v479
            )
            fn.run(
                x.data_ptr(),
                out.data_ptr(),
                x.data_ptr() + (total - 1) * 2,
                out.data_ptr() + total - 1,
                total // 2,
                fp8_dtype == torch.float8_e4m3fn,
                4096,
                enable_fp_fusion=False,
                grid=(40,),
                warmup=False,
            )
            return out
        if (
            (
                total >= 8388608
                or (total >= 131072 and fp8_dtype == torch.float8_e5m2)
            )
            and total % 2 == 0
            and (x.dtype == torch.float16)
            and (scale_inv == 1.0)
            and contiguous
            and (x.storage_offset() % 2 == 0)
        ):
            _half_pair_kernel.run(
                x.data_ptr(),
                out.data_ptr(),
                total // 2,
                fp8_dtype == torch.float8_e4m3fn,
                4096,
                enable_fp_fusion=False,
                grid=(min((total + 8191) // 8192, 40),),
                warmup=False,
            )
            return out
        if (
            total >= 8388608
            and total % 2 == 0
            and (x.dtype == torch.bfloat16)
            and (scale_inv == 1.0)
            and contiguous
            and (x.storage_offset() % 2 == 0)
        ):
            _pair_kernel.run(
                x.data_ptr(),
                out.data_ptr(),
                total // 2,
                fp8_dtype == torch.float8_e4m3fn,
                4096,
                enable_fp_fusion=False,
                grid=(min((total + 8191) // 8192, 40),),
                warmup=False,
            )
        elif (
            scale_inv == 1.0
            and fp8_dtype == torch.float8_e4m3fn
            and contiguous
        ):
            (_unit_bf16 if x.dtype == torch.bfloat16 else _unit_fp16).run(
                x.data_ptr(),
                out.data_ptr(),
                total,
                enable_fp_fusion=False,
                grid=(min((total + 4095) // 4096, 40),),
                warmup=False,
            )
        else:
            n = x.size(-1)
            stride = _row_stride(x, n)
            block = 4096 if stride == n else min(4096, n)
            (_kernel_bf16 if x.dtype == torch.bfloat16 else _kernel_fp16).run(
                x.data_ptr(),
                out.data_ptr(),
                total,
                n,
                stride,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                block,
                enable_fp_fusion=False,
                grid=(min((total + block - 1) // block, 40),),
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
    _kernel(X, Y, TOTAL, N, STRIDE, SCALE, E4, BLOCK)


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
    _kernel(X, Y, TOTAL, N, STRIDE, SCALE, E4, BLOCK)


@triton.jit
def _unit_bf16(X: "*bf16", Y: "*u8", TOTAL: tl.constexpr):
    _kernel(X, Y, TOTAL, TOTAL, TOTAL, 1.0, True, 4096)


@triton.jit
def _unit_fp16(X: "*fp16", Y: "*u8", TOTAL: tl.constexpr):
    _kernel(X, Y, TOTAL, TOTAL, TOTAL, 1.0, True, 4096)


@triton.jit
def _encode_pair(raw, E4: tl.constexpr):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX16: tl.constexpr = 17376 if E4 else 18272
    addend: tl.constexpr = (
        8388608.0 - (127 - BIAS << M) - 8388608.0 * 2.0 ** (M - 7)
    )
    magic = tl.full((), 2.0 ** (24 - BIAS - M), tl.float32)
    am = raw & 32767
    bh = raw >> 16
    a = tl.minimum(am, MAX16)
    bm = bh & 32767
    b = tl.minimum(bm, MAX16)
    signs = ((raw & -2147450880) * 257 >> 16).to(tl.uint16)
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
def _pair_kernel(
    X: "*i32",
    Y: "*i16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        if TOTAL % BLOCK == 0:
            raw = tl.load(X + off)
        else:
            raw = tl.load(X + off, off < TOTAL, other=0)
        code = _encode_pair(raw, E4)
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)


def _row_stride(x, n):
    for axis in range(x.ndim - 2, -1, -1):
        if x.size(axis) > 1:
            return x.stride(axis)
    return n


@triton.jit
def _encode_half_direct(value, E4: tl.constexpr):
    raw = value.to(tl.int16, bitcast=True).to(tl.int32)
    sign = raw >> 8 & 128
    mag = tl.minimum(raw & 32767, 31488)
    code = ((mag | 1191182336).to(tl.float32, bitcast=True) + 8355840.0).to(
        tl.int32, bitcast=True
    )
    return (code | sign).to(tl.uint8)


@triton.jit
def _encode_half_pair(raw, E4: tl.constexpr):
    a = tl.minimum(raw & 32767, 24320 if E4 else 31488)
    b = tl.minimum(raw >> 16 & 32767, 24320 if E4 else 31488)
    signs = ((raw & -2147450880) * 257 >> 16).to(tl.uint16)
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
def _half_pair_kernel(
    X: "*i32",
    Y: "*i16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        if TOTAL % BLOCK == 0:
            raw = tl.load(X + off)
        else:
            raw = tl.load(X + off, off < TOTAL, other=0)
        code = _encode_half_pair(raw, E4)
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)


@triton.jit
def _pair_kernel_odd_v479(
    X: "*i32",
    Y: "*i16",
    XT: "*bf16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        if TOTAL % BLOCK == 0:
            raw = tl.load(X + off)
        else:
            raw = tl.load(X + off, off < TOTAL, other=0)
        code = _encode_pair(raw, E4)
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)
    tail = tl.arange(0, 128)
    value = tl.load(XT + tail, tail < 1, other=0).to(tl.float32)
    code = _encode(
        value, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
    )
    tl.store(YT + tail, code, (tail < 1) & (tl.program_id(0) == 0))


@triton.jit
def _half_pair_kernel_odd_v479(
    X: "*i32",
    Y: "*i16",
    XT: "*fp16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        if TOTAL % BLOCK == 0:
            raw = tl.load(X + off)
        else:
            raw = tl.load(X + off, off < TOTAL, other=0)
        code = _encode_half_pair(raw, E4)
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)
    tail = tl.arange(0, 128)
    value = tl.load(XT + tail, tail < 1, other=0).to(tl.float32)
    code = _encode(
        value, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
    )
    tl.store(YT + tail, code, (tail < 1) & (tl.program_id(0) == 0))


@triton.jit
def _pair_kernel_split_v485(
    X: "*i16",
    Y: "*i16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        source = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
        if TOTAL % BLOCK == 0:
            halves = tl.load(X + source)
        else:
            halves = tl.load(X + source, source < TOTAL * 2, other=0)
        a, b = tl.split(tl.reshape(halves, (BLOCK, 2)))
        code = _encode_pair_direct_v486(a, b, E4)
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)


@triton.jit
def _half_pair_kernel_split_v485(
    X: "*i16",
    Y: "*i16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        source = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
        if TOTAL % BLOCK == 0:
            halves = tl.load(X + source)
        else:
            halves = tl.load(X + source, source < TOTAL * 2, other=0)
        a, b = tl.split(tl.reshape(halves, (BLOCK, 2)))
        code = _encode_half_pair_direct_v486(a, b, E4)
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)


@triton.jit
def _encode_pair_direct_v486(u, v, E4: tl.constexpr):
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
def _encode_half_pair_direct_v486(u, v, E4: tl.constexpr):
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
def _pair_kernel_split_v485_odd_v489(
    X: "*i16",
    Y: "*i16",
    XT: "*bf16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        source = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
        if TOTAL % BLOCK == 0:
            halves = tl.load(X + source)
        else:
            halves = tl.load(X + source, source < TOTAL * 2, other=0)
        a, b = tl.split(tl.reshape(halves, (BLOCK, 2)))
        code = _encode_pair_direct_v486(a, b, E4)
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)
    tail = tl.arange(0, 128)
    value = tl.load(XT + tail, tail < 1, other=0).to(tl.float32)
    code = _encode(
        value, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
    )
    tl.store(YT + tail, code, (tail < 1) & (tl.program_id(0) == 0))


@triton.jit
def _half_pair_kernel_split_v485_odd_v489(
    X: "*i16",
    Y: "*i16",
    XT: "*fp16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        source = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
        if TOTAL % BLOCK == 0:
            halves = tl.load(X + source)
        else:
            halves = tl.load(X + source, source < TOTAL * 2, other=0)
        a, b = tl.split(tl.reshape(halves, (BLOCK, 2)))
        code = _encode_half_pair_direct_v486(a, b, E4)
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)
    tail = tl.arange(0, 128)
    value = tl.load(XT + tail, tail < 1, other=0).to(tl.float32)
    code = _encode(
        value, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
    )
    tl.store(YT + tail, code, (tail < 1) & (tl.program_id(0) == 0))


@triton.jit
def _half_code_v490(raw, E4: tl.constexpr, SHIFT: tl.constexpr):
    sign = (raw >> 8 & 128).to(tl.int16)
    mag = tl.minimum(raw & 32767, (24320 if E4 else 31488) + SHIFT * 1024)
    shifted = tl.maximum(mag.to(tl.int32) - SHIFT * 1024, 0).to(tl.int32)
    normal = (
        (
            (shifted | (143 if E4 else 142) << 23).to(tl.float32, bitcast=True)
            + (8323008.0 if E4 else 8355840.0)
        )
        .to(tl.int32, bitcast=True)
        .to(tl.int16)
    )
    subvalue = (mag + (112 - SHIFT) * 1024 << 13).to(tl.float32, bitcast=True)
    if not E4 and SHIFT <= 2:
        subvalue = tl.where(
            mag < 1024, mag.to(tl.float32) * 2.0 ** (-24 - SHIFT), subvalue
        )
    sub = (
        (subvalue + (16384.0 if E4 else 128.0))
        .to(tl.int32, bitcast=True)
        .to(tl.int16)
    )
    code = tl.where(mag < (9216 if E4 else 1024) + SHIFT * 1024, sub, normal)
    return code & 127 | sign


@triton.jit
def _encode_scaled_half_pair_v507(raw, E4: tl.constexpr, SHIFT: tl.constexpr):
    maximum: tl.constexpr = (24320 if E4 else 31488) + SHIFT * 1024
    threshold: tl.constexpr = (9216 if E4 else 1024) + SHIFT * 1024
    a = tl.minimum(raw & 32767, maximum)
    b = tl.minimum(raw >> 16 & 32767, maximum)
    signs = ((raw & -2147450880) * 257 >> 16).to(tl.uint16)
    an = (
        (
            (tl.maximum(a - SHIFT * 1024, 0) | (143 if E4 else 142) << 23).to(
                tl.float32, bitcast=True
            )
            + (8323008.0 if E4 else 8355840.0)
        )
        .to(tl.int32, bitcast=True)
        .to(tl.int16)
    )
    bn = (
        (
            (tl.maximum(b - SHIFT * 1024, 0) | (143 if E4 else 142) << 23).to(
                tl.float32, bitcast=True
            )
            + (8323008.0 if E4 else 8355840.0)
        )
        .to(tl.int32, bitcast=True)
        .to(tl.int16)
    )
    ac = tl.minimum(a, threshold)
    bc = tl.minimum(b, threshold)
    if not E4 and SHIFT <= 2:
        if SHIFT == 1:
            av = (ac | 141 << 23).to(tl.float32, bitcast=True)
            bv = (bc | 141 << 23).to(tl.float32, bitcast=True)
        else:
            av = (tl.maximum(ac, (ac - 1024) * 2) | 140 << 23).to(
                tl.float32, bitcast=True
            )
            bv = (tl.maximum(bc, (bc - 1024) * 2) | 140 << 23).to(
                tl.float32, bitcast=True
            )
        suba = (
            (av + (8372224.0 if SHIFT == 1 else 8380416.0))
            .to(tl.int32, bitcast=True)
            .to(tl.int16)
        )
        subb = (
            (bv + (8372224.0 if SHIFT == 1 else 8380416.0))
            .to(tl.int32, bitcast=True)
            .to(tl.int16)
        )
    else:
        av = (ac + (112 - SHIFT) * 1024 << 13).to(tl.float32, bitcast=True)
        bv = (bc + (112 - SHIFT) * 1024 << 13).to(tl.float32, bitcast=True)
        suba = (
            (av + (16384.0 if E4 else 128.0))
            .to(tl.int32, bitcast=True)
            .to(tl.int16)
        )
        subb = (
            (bv + (16384.0 if E4 else 128.0))
            .to(tl.int32, bitcast=True)
            .to(tl.int16)
        )
    ca = tl.maximum(an, suba).to(tl.uint16)
    cb = tl.maximum(bn, subb).to(tl.uint16)
    return ca | cb << 8 | signs


@triton.jit
def _half_scaled_pair_kernel_v507(
    X: "*i32",
    Y: "*i16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        if TOTAL % BLOCK == 0:
            raw = tl.load(X + off)
        else:
            raw = tl.load(X + off, off < TOTAL, other=0)
        code = _encode_scaled_half_pair_v507(raw, E4, SHIFT)
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)


@triton.jit
def _encode_scaled_split_v512(u, v, E4: tl.constexpr, SHIFT: tl.constexpr):
    maximum: tl.constexpr = (24320 if E4 else 31488) + SHIFT * 1024
    threshold: tl.constexpr = (9216 if E4 else 1024) + SHIFT * 1024
    a = tl.minimum(u.to(tl.int32) & 32767, maximum)
    b = tl.minimum(v.to(tl.int32) & 32767, maximum)
    signs = u.to(tl.uint16) >> 8 & 128 | v.to(tl.uint16) & 32768
    an = (
        (
            (tl.maximum(a - SHIFT * 1024, 0) | (143 if E4 else 142) << 23).to(
                tl.float32, bitcast=True
            )
            + (8323008.0 if E4 else 8355840.0)
        )
        .to(tl.int32, bitcast=True)
        .to(tl.int16)
    )
    bn = (
        (
            (tl.maximum(b - SHIFT * 1024, 0) | (143 if E4 else 142) << 23).to(
                tl.float32, bitcast=True
            )
            + (8323008.0 if E4 else 8355840.0)
        )
        .to(tl.int32, bitcast=True)
        .to(tl.int16)
    )
    ac = tl.minimum(a, threshold)
    bc = tl.minimum(b, threshold)
    if not E4 and SHIFT <= 2:
        if SHIFT == 1:
            av = (ac | 141 << 23).to(tl.float32, bitcast=True)
            bv = (bc | 141 << 23).to(tl.float32, bitcast=True)
        else:
            av = (tl.maximum(ac, (ac - 1024) * 2) | 140 << 23).to(
                tl.float32, bitcast=True
            )
            bv = (tl.maximum(bc, (bc - 1024) * 2) | 140 << 23).to(
                tl.float32, bitcast=True
            )
        suba = (
            (av + (8372224.0 if SHIFT == 1 else 8380416.0))
            .to(tl.int32, bitcast=True)
            .to(tl.int16)
        )
        subb = (
            (bv + (8372224.0 if SHIFT == 1 else 8380416.0))
            .to(tl.int32, bitcast=True)
            .to(tl.int16)
        )
    else:
        av = (ac + (112 - SHIFT) * 1024 << 13).to(tl.float32, bitcast=True)
        bv = (bc + (112 - SHIFT) * 1024 << 13).to(tl.float32, bitcast=True)
        suba = (
            (av + (16384.0 if E4 else 128.0))
            .to(tl.int32, bitcast=True)
            .to(tl.int16)
        )
        subb = (
            (bv + (16384.0 if E4 else 128.0))
            .to(tl.int32, bitcast=True)
            .to(tl.int16)
        )
    ca = tl.maximum(an, suba).to(tl.uint16)
    cb = tl.maximum(bn, subb).to(tl.uint16)
    return ca | cb << 8 | signs


@triton.jit
def _hw_scaled_split_kernel_v512(
    X: "*i16",
    Y: "*i16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        source = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
        if TOTAL % BLOCK == 0:
            halves = tl.load(X + source)
        else:
            halves = tl.load(X + source, source < TOTAL * 2, other=0)
        a, b = tl.split(tl.reshape(halves, (BLOCK, 2)))
        code = _encode_scaled_split_v512(a, b, E4, SHIFT)
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)


@triton.jit
def _hw_scaled_split_odd_v512(
    X: "*i16",
    Y: "*i16",
    XT: "*i16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _hw_scaled_split_kernel_v512(X, Y, TOTAL, E4, SHIFT, BLOCK)
    r = tl.arange(0, 128)
    value = tl.load(XT + r, r == 0, other=0).to(tl.int32)
    code = _half_code_v490(value, E4, SHIFT).to(tl.uint8)
    tl.store(YT + r, code, (r == 0) & (tl.program_id(0) == 0))


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
def _zero_pair_hw_v592(
    X: "*i32",
    Y: "*i16",
    PAIRS: tl.constexpr,
    NEG: tl.constexpr,
    BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        tl.program_id(0), tl.cdiv(PAIRS, BLOCK), tl.num_programs(0)
    ):
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
            a > inf,
            sa | 127,
            tl.where(a == inf, 255, sa ^ (128 if NEG else 0)),
        ).to(tl.int16)
        cb = tl.where(
            b > inf,
            sb | 127,
            tl.where(b == inf, 255, sb ^ (128 if NEG else 0)),
        ).to(tl.int16)
        code = ca | cb << 8
        if PAIRS % BLOCK == 0:
            tl.store(Y + off, code)
        else:
            tl.store(Y + off, code, off < PAIRS)


@triton.jit
def _general_code_hw_v571(v, E4: tl.constexpr):
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
def _general_split_hw_v573(
    X: "*i16",
    Y: "*i16",
    XT: "*i16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    PAIRS: tl.constexpr = TOTAL // 2
    for tile in range(
        tl.program_id(0), tl.cdiv(PAIRS, BLOCK), tl.num_programs(0)
    ):
        pair = tile * BLOCK + tl.arange(0, BLOCK)
        lo = (
            tl.load(X + pair * 2)
            if PAIRS % BLOCK == 0
            else tl.load(X + pair * 2, pair < PAIRS, other=0)
        )
        hi = (
            tl.load(X + pair * 2 + 1)
            if PAIRS % BLOCK == 0
            else tl.load(X + pair * 2 + 1, pair < PAIRS, other=0)
        )
        if BF16:
            a = lo.to(tl.bfloat16, bitcast=True).to(tl.float32)
            b = hi.to(tl.bfloat16, bitcast=True).to(tl.float32)
        else:
            a = lo.to(tl.float16, bitcast=True).to(tl.float32)
            b = hi.to(tl.float16, bitcast=True).to(tl.float32)
        a = a * SCALE
        b = b * SCALE
        ca = _general_code_hw_v571(a, E4)
        cb = _general_code_hw_v571(b, E4)
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
        last = _general_code_hw_v571(v, E4) | raw >> 8 & 128 ^ (
            128 if SCALE < 0.0 else 0
        )
        tl.store(YT + r, last.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


@triton.jit
def _general_halfword_v605(
    raw, BF16: tl.constexpr, SCALE: tl.constexpr, E4: tl.constexpr
):
    v = (
        raw.to(tl.bfloat16, bitcast=True).to(tl.float32)
        if BF16
        else raw.to(tl.float16, bitcast=True).to(tl.float32)
    )
    v = v * SCALE
    code = _general_code_hw_v571(v, E4)
    return code | (raw >> 8 & 128 ^ (128 if SCALE < 0.0 else 0)).to(tl.uint16)


@triton.jit
def _general_serial_full_v612(
    X: "*i16",
    Y: "*i16",
    XT: "*i16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    PAIRS: tl.constexpr = TOTAL // 2
    for tile in range(
        tl.program_id(0), tl.cdiv(PAIRS, BLOCK), tl.num_programs(0)
    ):
        pair = tile * BLOCK + tl.arange(0, BLOCK)
        lo = (
            tl.load(X + pair * 2)
            if PAIRS % BLOCK == 0
            else tl.load(X + pair * 2, pair < PAIRS, other=0)
        )
        ca = _general_halfword_v605(lo, BF16, SCALE, E4)
        hi = (
            tl.load(X + pair * 2 + 1)
            if PAIRS % BLOCK == 0
            else tl.load(X + pair * 2 + 1, pair < PAIRS, other=0)
        )
        cb = _general_halfword_v605(hi, BF16, SCALE, E4)
        code = (ca | cb << 8).to(tl.int16)
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
        last = _general_code_hw_v571(v, E4) | raw >> 8 & 128 ^ (
            128 if SCALE < 0.0 else 0
        )
        tl.store(YT + r, last.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


@triton.jit
def _hw_strided_pair_v606(
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
    for tile in range(
        tl.program_id(0), tl.cdiv(PAIRS, BLOCK), tl.num_programs(0)
    ):
        pair = tile * BLOCK + tl.arange(0, BLOCK)
        base = tile * BLOCK * 2 // N * PITCH
        span_offset = tl.arange(0, SPAN)
        physical = base + span_offset
        values = tl.load(X + physical, physical < LAST, other=0)
        values = values.to(tl.bfloat16 if BF16 else tl.float16, bitcast=True)
        local = tl.arange(0, BLOCK) * 2
        index = local // N * PITCH + local % N
        lo = tl.gather(values, index, 0).to(tl.int16, bitcast=True)
        hi = tl.gather(values, index + (PITCH if N == 1 else 1), 0).to(
            tl.int16, bitcast=True
        )
        if SCALE == 1.0 and TOTAL >= 16777216:
            if BF16:
                code = _encode_pair_direct_v486(lo, hi, E4).to(tl.int16)
            else:
                code = _encode_half_pair_direct_v486(lo, hi, E4).to(tl.int16)
        else:
            if BF16:
                a = lo.to(tl.bfloat16, bitcast=True).to(tl.float32)
                b = hi.to(tl.bfloat16, bitcast=True).to(tl.float32)
            else:
                a = lo.to(tl.float16, bitcast=True).to(tl.float32)
                b = hi.to(tl.float16, bitcast=True).to(tl.float32)
            a = a * SCALE
            b = b * SCALE
            ca = _general_code_hw_v571(a, E4)
            cb = _general_code_hw_v571(b, E4)
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
        last = _general_code_hw_v571(v, E4) | raw >> 8 & 128 ^ (
            128 if SCALE < 0.0 else 0
        )
        tl.store(YT + r, last.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


@triton.jit
def _hw_sparse_v720(
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
    for tile in range(
        tl.program_id(0), tl.cdiv(PAIRS, BLOCK), tl.num_programs(0)
    ):
        pair = tile * BLOCK + tl.arange(0, BLOCK)
        if N == 2 * BLOCK:
            col = tl.arange(0, 2 * BLOCK)
            values = tl.load(
                X + tile * PITCH + col, tile < TOTAL // N, other=0
            )
        else:
            row = tile * (BLOCK * 2 // N) + tl.arange(0, BLOCK * 2 // N)
            col = tl.arange(0, N)
            source = row[:, None] * PITCH + col[None, :]
            matrix = tl.load(X + source, row[:, None] < TOTAL // N, other=0)
            values = tl.reshape(matrix, (BLOCK * 2,))
        values = values.to(tl.bfloat16 if BF16 else tl.float16, bitcast=True)
        lo, hi = tl.split(tl.reshape(values, (BLOCK, 2)))
        lo = lo.to(tl.int16, bitcast=True)
        hi = hi.to(tl.int16, bitcast=True)
        if SCALE == 1.0 and TOTAL >= 16777216:
            if BF16:
                code = _encode_pair_direct_v486(lo, hi, E4).to(tl.int16)
            else:
                code = _encode_half_pair_direct_v486(lo, hi, E4).to(tl.int16)
        else:
            if BF16:
                a = lo.to(tl.bfloat16, bitcast=True).to(tl.float32)
                b = hi.to(tl.bfloat16, bitcast=True).to(tl.float32)
            else:
                a = lo.to(tl.float16, bitcast=True).to(tl.float32)
                b = hi.to(tl.float16, bitcast=True).to(tl.float32)
            a = a * SCALE
            b = b * SCALE
            ca = _general_code_hw_v571(a, E4)
            cb = _general_code_hw_v571(b, E4)
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
        last = _general_code_hw_v571(v, E4) | raw >> 8 & 128 ^ (
            128 if SCALE < 0.0 else 0
        )
        tl.store(YT + r, last.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


@triton.jit
def _hw_zero_matrix_v734(
    X: "*u16",
    Y: "*u16",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    NEG: tl.constexpr,
    BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    inf: tl.constexpr = 32640 if BF16 else 31744
    for tile in range(
        tl.program_id(0), tl.cdiv(TOTAL, 2 * BLOCK), tl.num_programs(0)
    ):
        row = tile * (2 * BLOCK // N) + tl.arange(0, 2 * BLOCK // N)
        col = tl.arange(0, N)
        matrix = tl.load(
            X + row[:, None] * PITCH + col[None, :],
            row[:, None] < TOTAL // N,
            other=0,
        )
        raw = tl.reshape(matrix, (2 * BLOCK,)).to(tl.int32)
        sign = raw >> 8 & 128
        mag = raw & 32767
        code = tl.where(
            mag > inf,
            sign | 127,
            tl.where(mag == inf, 255, sign ^ (128 if NEG else 0)),
        ).to(tl.uint16)
        lo, hi = tl.split(tl.reshape(code, (BLOCK, 2)))
        off = tile * BLOCK + tl.arange(0, BLOCK)
        tl.store(Y + off, lo | hi << 8, off < TOTAL // 2)


@triton.jit
def _hw_zero_code16_v759(raw, NEG: tl.constexpr, BF16: tl.constexpr):
    sign = raw >> 8 & 128
    mag = raw & 32767
    inf: tl.constexpr = 32640 if BF16 else 31744
    return tl.where(
        mag > inf,
        sign | 127,
        tl.where(mag == inf, 255, sign ^ (128 if NEG else 0)),
    ).to(tl.uint16)


@triton.jit
def _hw_zero_span_v777(
    X: "*u16",
    Y: "*u16",
    XT: "*u16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    BF16: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    SPAN: tl.constexpr = triton.next_power_of_2(
        (2 * BLOCK - 1) // N * PITCH + (2 * BLOCK - 1) % N + 1
    )
    LAST: tl.constexpr = (TOTAL // N - 1) * PITCH + N
    for tile in range(
        tl.program_id(0), tl.cdiv(TOTAL // 2, BLOCK), tl.num_programs(0)
    ):
        pair = tile * BLOCK + tl.arange(0, BLOCK)
        physical = tile * BLOCK * 2 // N * PITCH + tl.arange(0, SPAN)
        values = tl.load(X + physical, physical < LAST, other=0).to(
            tl.float16, bitcast=True
        )
        local = tl.arange(0, BLOCK) * 2
        index = local // N * PITCH + local % N
        lo = tl.gather(values, index, 0).to(tl.uint16, bitcast=True)
        hi = tl.gather(values, index + (PITCH if N == 1 else 1), 0).to(
            tl.uint16, bitcast=True
        )
        a = _hw_zero_code16_v759(lo, NEG, BF16)
        b = _hw_zero_code16_v759(hi, NEG, BF16)
        tl.store(Y + pair, a | b << 8, pair < TOTAL // 2)
    if TOTAL % 2:
        r = tl.arange(0, 128)
        raw = tl.load(XT + r, r == 0, other=0)
        code = _hw_zero_code16_v759(raw, NEG, BF16)
        tl.store(YT + r, code.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


@triton.jit
def _negative_unit_pair_v965(
    X: "*i32",
    Y: "*i16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
    BF16: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        if TOTAL % BLOCK == 0:
            raw = tl.load(X + off)
        else:
            raw = tl.load(X + off, off < TOTAL, other=0)
        code = (
            _encode_pair(raw, E4) if BF16 else _encode_half_pair(raw, E4)
        ) ^ 32896
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)


@triton.jit
def _negative_tail_v987(
    X: "*i32",
    Y: "*i16",
    XT: "*i16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    BF16: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
    ALIGNED: tl.constexpr,
    XH: "*i16",
):
    PAIRS: tl.constexpr = TOTAL // 2
    for tile in range(
        tl.program_id(0), tl.cdiv(PAIRS, BLOCK), tl.num_programs(0)
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        if ALIGNED:
            raw = (
                tl.load(X + off)
                if PAIRS % BLOCK == 0
                else tl.load(X + off, off < PAIRS, other=0)
            )
            code = (
                _encode_pair(raw, E4) if BF16 else _encode_half_pair(raw, E4)
            ) ^ 32896
        else:
            src = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
            halves = (
                tl.load(XH + src)
                if PAIRS % BLOCK == 0
                else tl.load(XH + src, src < PAIRS * 2, other=0)
            )
            a, b = tl.split(tl.reshape(halves, (BLOCK, 2)))
            code = (
                _encode_pair_direct_v486(a, b, E4)
                if BF16
                else _encode_half_pair_direct_v486(a, b, E4)
            ) ^ 32896
        if PAIRS % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < PAIRS)
    if TOTAL % 2:
        r = tl.arange(0, 128)
        raw = tl.load(XT + r, r == 0, other=0)
        code = _general_halfword_v605(raw, BF16, -1.0, E4)
        tl.store(YT + r, code.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


@triton.jit
def _hw_packed_power_code_v968(raw, E4: tl.constexpr, Q: tl.constexpr):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX16: tl.constexpr = 17376 if E4 else 18272
    addend: tl.constexpr = (
        8388608.0 - (127 - BIAS << M) - 8388608.0 * 2.0 ** (M - 7)
    )
    magic = tl.full((), 2.0 ** (24 - BIAS - M), tl.float32)
    am = raw & 32767
    bh = raw >> 16
    a = (
        tl.minimum(am, MAX16 - Q * 128) + Q * 128
        if Q >= 0
        else tl.maximum(tl.minimum(am, MAX16 - Q * 128) + Q * 128, 0)
    )
    bm = bh & 32767
    b = (
        tl.minimum(bm, MAX16 - Q * 128) + Q * 128
        if Q >= 0
        else tl.maximum(tl.minimum(bm, MAX16 - Q * 128) + Q * 128, 0)
    )
    signs = ((raw & -2147450880) * 257 >> 16).to(tl.uint16)
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
def _hw_packed_power_v968(
    X: "*i32",
    Y: "*i16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
    Q: tl.constexpr,
    NEG: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        if TOTAL % BLOCK == 0:
            raw = tl.load(X + off)
        else:
            raw = tl.load(X + off, off < TOTAL, other=0)
        code = _hw_packed_power_code_v968(raw, E4, Q) ^ (32896 if NEG else 0)
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)


@triton.jit
def _hw_power_odd_v986(
    X: "*i32",
    Y: "*i16",
    XT: "*i16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
    Q: tl.constexpr,
    SCALE: tl.constexpr,
):
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if TOTAL > 2147483647
            else tl.program_id(0)
        ),
        tl.cdiv(TOTAL, BLOCK),
        tl.num_programs(0),
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        if TOTAL % BLOCK == 0:
            raw = tl.load(X + off)
        else:
            raw = tl.load(X + off, off < TOTAL, other=0)
        code = _hw_packed_power_code_v968(raw, E4, Q) ^ (
            32896 if SCALE < 0.0 else 0
        )
        if TOTAL % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < TOTAL)
    tail = tl.arange(0, 128)
    raw_tail = tl.load(XT + tail, tail < 1, other=0)
    code = _general_halfword_v605(raw_tail, True, SCALE, E4).to(tl.uint8)
    tl.store(YT + tail, code, (tail < 1) & (tl.program_id(0) == 0))


@triton.jit
def _hw_split_power_code_v992(u, v, E4: tl.constexpr, Q: tl.constexpr):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX16: tl.constexpr = 17376 if E4 else 18272
    addend: tl.constexpr = (
        8388608.0 - (127 - BIAS << M) - 8388608.0 * 2.0 ** (M - 7)
    )
    magic = tl.full((), 2.0 ** (24 - BIAS - M), tl.float32)
    am = u.to(tl.int32) & 32767
    bh = v.to(tl.int32)
    a = (
        tl.minimum(am, MAX16 - Q * 128) + Q * 128
        if Q >= 0
        else tl.maximum(tl.minimum(am, MAX16 - Q * 128) + Q * 128, 0)
    )
    bm = bh & 32767
    b = (
        tl.minimum(bm, MAX16 - Q * 128) + Q * 128
        if Q >= 0
        else tl.maximum(tl.minimum(bm, MAX16 - Q * 128) + Q * 128, 0)
    )
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
def _hw_split_power_v992(
    X: "*i16",
    Y: "*i16",
    XT: "*i16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    Q: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    PAIRS: tl.constexpr = TOTAL // 2
    for tile in range(
        tl.program_id(0), tl.cdiv(PAIRS, BLOCK), tl.num_programs(0)
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        src = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
        if PAIRS % BLOCK == 0:
            halves = tl.load(X + src)
        else:
            halves = tl.load(X + src, src < PAIRS * 2, other=0)
        a, b = tl.split(tl.reshape(halves, (BLOCK, 2)))
        code = _hw_split_power_code_v992(a, b, E4, Q) ^ (
            32896 if SCALE < 0.0 else 0
        )
        if PAIRS % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < PAIRS)
    if TOTAL % 2:
        r = tl.arange(0, 128)
        raw = tl.load(XT + r, r == 0, other=0)
        code = _general_halfword_v605(raw, True, SCALE, E4)
        tl.store(YT + r, code.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


@triton.jit
def _general_fulltail_v1000(
    X: "*i16",
    Y: "*i16",
    XT: "*i16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    PAIRS: tl.constexpr = TOTAL // 2
    for tile in range(tl.program_id(0), PAIRS // BLOCK, tl.num_programs(0)):
        pair = tile * BLOCK + tl.arange(0, BLOCK)
        lo = tl.load(X + pair * 2)
        ca = _general_halfword_v605(lo, BF16, SCALE, E4)
        hi = tl.load(X + pair * 2 + 1)
        cb = _general_halfword_v605(hi, BF16, SCALE, E4)
        code = (ca | cb << 8).to(tl.int16)
        tl.store(Y + pair, code)
    if PAIRS % BLOCK != 0:
        if tl.program_id(0) == 0:
            pair = PAIRS // BLOCK * BLOCK + tl.arange(0, BLOCK)
            src = PAIRS // BLOCK * BLOCK * 2 + tl.arange(0, BLOCK * 2)
            halves = tl.load(X + src, src < PAIRS * 2, other=0)
            lo, hi = tl.split(tl.reshape(halves, (BLOCK, 2)))
            ca = _general_halfword_v605(lo, BF16, SCALE, E4)
            cb = _general_halfword_v605(hi, BF16, SCALE, E4)
            tl.store(Y + pair, (ca | cb << 8).to(tl.int16), pair < PAIRS)
    if TOTAL % 2 != 0:
        r = tl.arange(0, 128)
        raw = tl.load(XT + r, r == 0, other=0)
        v = (
            raw.to(tl.bfloat16, bitcast=True).to(tl.float32)
            if BF16
            else raw.to(tl.float16, bitcast=True).to(tl.float32)
        )
        v = v * SCALE
        last = _general_code_hw_v571(v, E4) | raw >> 8 & 128 ^ (
            128 if SCALE < 0.0 else 0
        )
        tl.store(YT + r, last.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


@triton.jit
def _bf16_general_packed_v1012(
    X: "*i32",
    Y: "*i16",
    XT: "*i16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
    PRODUCT_SIGN: tl.constexpr = False,
):
    PAIRS: tl.constexpr = TOTAL // 2
    for tile in range(tl.program_id(0), PAIRS // BLOCK, tl.num_programs(0)):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        raw = tl.load(X + off)
        code = _bf16_packed_codes_v1051(raw, SCALE, E4, PRODUCT_SIGN)
        tl.store(Y + off, code.to(tl.int16))
    if PAIRS % BLOCK != 0:
        if tl.program_id(0) == 0:
            off = PAIRS // BLOCK * BLOCK + tl.arange(0, BLOCK)
            raw = tl.load(X + off, off < PAIRS, other=0)
            code = _bf16_packed_codes_v1051(raw, SCALE, E4, PRODUCT_SIGN)
            tl.store(Y + off, code.to(tl.int16), off < PAIRS)
    if TOTAL % 2:
        r = tl.arange(0, 128)
        raw = tl.load(XT + r, r == 0, other=0)
        if PRODUCT_SIGN:
            v = raw.to(tl.bfloat16, bitcast=True).to(tl.float32) * SCALE
            code = (
                _general_code_hw_v571(v, E4)
                | v.to(tl.int32, bitcast=True) >> 24 & 128
            )
        else:
            code = _general_halfword_v605(raw, True, SCALE, E4)
        tl.store(YT + r, code.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


@triton.jit
def _half_general_packed_v1019(
    X: "*i32",
    Y: "*i16",
    XT: "*i16",
    YT: "*u8",
    TOTAL: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    PAIRS: tl.constexpr = TOTAL // 2
    for tile in range(
        tl.program_id(0), tl.cdiv(PAIRS, BLOCK), tl.num_programs(0)
    ):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        if PAIRS % BLOCK == 0:
            raw = tl.load(X + off)
        else:
            raw = tl.load(X + off, off < PAIRS, other=0)
        a = (
            raw.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
            * SCALE
        )
        ca = _general_code_hw_v571(a, E4).to(tl.int16)
        b = (raw >> 16).to(tl.int16).to(tl.float16, bitcast=True).to(
            tl.float32
        ) * SCALE
        cb = _general_code_hw_v571(b, E4).to(tl.int16)
        signs = ((raw & -2147450880) * 257 >> 16).to(tl.uint16) ^ (
            32896 if SCALE < 0.0 else 0
        )
        code = ca.to(tl.uint16) | cb.to(tl.uint16) << 8 | signs
        if PAIRS % BLOCK == 0:
            tl.store(Y + off, code.to(tl.int16))
        else:
            tl.store(Y + off, code.to(tl.int16), off < PAIRS)
    if TOTAL % 2:
        r = tl.arange(0, 128)
        raw = tl.load(XT + r, r == 0, other=0)
        code = _general_halfword_v605(raw, False, SCALE, E4)
        tl.store(YT + r, code.to(tl.uint8), (r == 0) & (tl.program_id(0) == 0))


@triton.jit
def _bf16_packed_codes_v1051(
    raw, SCALE: tl.constexpr, E4: tl.constexpr, PRODUCT_SIGN: tl.constexpr
):
    a = (raw << 16).to(tl.float32, bitcast=True) * SCALE
    ca = _general_code_hw_v571(a, E4).to(tl.int16)
    b = (raw & -65536).to(tl.float32, bitcast=True) * SCALE
    cb = _general_code_hw_v571(b, E4).to(tl.int16)
    if PRODUCT_SIGN:
        signs = (
            a.to(tl.int32, bitcast=True) >> 24 & 128
            | (b.to(tl.int32, bitcast=True) >> 24 & 128) << 8
        )
    else:
        signs = ((raw & -2147450880) * 257 >> 16).to(tl.uint16) ^ (
            32896 if SCALE < 0.0 else 0
        )
    code = ca.to(tl.uint16) | cb.to(tl.uint16) << 8 | signs
    return code


__all__ = ["fp8_quantize"]
