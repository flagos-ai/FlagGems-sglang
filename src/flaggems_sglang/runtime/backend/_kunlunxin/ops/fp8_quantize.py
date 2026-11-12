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
from triton.language import int16 as i16
from triton.language import int32 as i32
from triton.language import uint8 as u8
from triton.language import uint16 as u16
from triton.language import uint32 as u32
from triton.language import uint64 as u64


@triton.jit
def _decode_fp16(raw):
    raw = raw.to(tl.int32)
    mag = raw & 32767
    normal = ((mag << 13) + (112 << 23)).to(tl.float32, bitcast=True)
    sub = (mag & 1023).to(tl.float32) * 5.960464477539063e-08
    value = tl.where(mag < 1024, sub, normal)
    return (value.to(tl.int32, bitcast=True) | (raw & 32768) << 16).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _encode16(raw, M: tl.constexpr, BIAS: tl.constexpr, MAX16: tl.constexpr):
    raw = raw.to(tl.int32)
    a = tl.minimum(raw & 32767, MAX16)
    addend: tl.constexpr = (
        8388608.0 - (127 - BIAS << M) - 8388608.0 * 2.0 ** (M - 7)
    )
    normal = ((a | 143 + M << 23).to(tl.float32, bitcast=True) + addend).to(
        tl.int32, bitcast=True
    )
    magic: tl.constexpr = 2.0 ** (24 - BIAS - M)
    sub = ((a << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.int32, bitcast=True
    )
    code = tl.where(a < 128 - BIAS << 7, sub, normal) & 127 | raw >> 8 & 128
    return code.to(tl.uint8)


@triton.jit
def _encode32(v, M: tl.constexpr, BIAS: tl.constexpr, MAX: tl.constexpr):
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
    Y,
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    BF16: tl.constexpr,
    M: tl.constexpr,
    BIAS: tl.constexpr,
    MAX: tl.constexpr,
    MAX16: tl.constexpr,
    BLOCK: tl.constexpr,
    Q: tl.constexpr = 99,
):
    X = X.to(tl.pointer_type(tl.int16))
    off = (
        tl.program_id(0).to(tl.int64)
        if (TOTAL + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
        or ((TOTAL + BLOCK - 1) // BLOCK * BLOCK - 1) // N * STRIDE + N - 1
        > 2147483647
        else tl.program_id(0)
    ) * BLOCK + tl.arange(0, BLOCK)
    src = off // N * STRIDE + off % N
    if TOTAL % BLOCK == 0 and (STRIDE == N or BLOCK <= N):
        v = tl.load(X + src, offset_state_policy="continuous")
    else:
        v = tl.load(X + src, off < TOTAL, other=0)
    if BF16:
        if SCALE == 1.0:
            code = _encode16(v, M, BIAS, MAX16)
        elif Q >= -8 and Q <= 8:
            code = (
                _encode16(v, M, BIAS + Q, MAX16 - 128 * Q)
                ^ (128 if SCALE < 0.0 else 0)
            ).to(tl.uint8)
        else:
            value = (v.to(tl.uint32) << 16).to(
                tl.float32, bitcast=True
            ) * SCALE
            code = _encode32(value, M, BIAS, MAX)
            if SCALE != 1.0:
                scale_sign = (
                    tl.full((), SCALE, tl.float32).to(tl.int32, bitcast=True)
                    >> 24
                    & 128
                )
                if BF16:
                    input_sign = v.to(tl.int32) >> 8 & 128
                else:
                    input_sign = v.to(tl.int32) >> 8 & 128
                code = (code.to(tl.int32) & 127 | input_sign ^ scale_sign).to(
                    tl.uint8
                )
    elif SCALE == 1.0 or SCALE == -1.0:
        code = (
            _half_code_v471(v.to(tl.uint32), M == 3)
            ^ (128 if SCALE < 0.0 else 0)
        ).to(tl.uint8)
    elif Q >= -8 and Q <= (8 if M == 3 else 3) and (Q != 0):
        if Q < 0:
            code = _half_code_v490(v.to(tl.uint32), M == 3, -Q)
        else:
            code = _half_up_code_v901(v.to(tl.uint32), M == 3, Q)
        code = (code ^ (128 if SCALE < 0.0 else 0)).to(tl.uint8)
    else:
        value = _decode_fp16(v.to(tl.uint32) & 32767) * (
            SCALE if SCALE >= 0.0 else -SCALE
        )
        code = _encode_magnitude32_v1151(value, M, BIAS, MAX)
        if SCALE != 1.0:
            scale_sign = (
                tl.full((), SCALE, tl.float32).to(tl.int32, bitcast=True) >> 24
                & 128
            )
            if BF16:
                input_sign = v.to(tl.int32) >> 8 & 128
            else:
                input_sign = v.to(tl.int32) >> 8 & 128
            code = (code.to(tl.int32) & 127 | input_sign ^ scale_sign).to(
                tl.uint8
            )
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


def _unpaired(x, scale_inv, fp8_dtype):
    out = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    n = x.shape[-1]
    stride = _rs(x, n)
    total = x.numel()
    m = 3 if fp8_dtype == torch.float8_e4m3fn else 2
    bias = 7 if m == 3 else 15
    maximum = 448.0 if m == 3 else 57344.0
    if total <= 65536:
        if total:
            _kernel_old[triton.cdiv(total, 1024),](
                x, out, total, n, stride, scale_inv, m, bias, maximum, 1024
            )
        return out.view(fp8_dtype)
    max16 = 17376 if m == 3 else 18272
    bf16 = x.dtype == torch.bfloat16
    source = x
    block = (
        min(n, 16384)
        if stride != n and n >= 2048 and (total >= 1048576)
        else (
            32768
            if stride == n and total >= 1048576
            else (
                8192
                if stride == n
                else (
                    (
                        16384
                        if total >= 4194304
                        and 64 <= n <= 1024
                        and (total % 16384 == 0)
                        else 4096
                    )
                    if total >= 1048576 and n <= 1024
                    else 1024
                )
            )
        )
    )
    if total:
        fraction, exponent = math.frexp(scale_inv)
        _kernel.run(
            source,
            out,
            total,
            n,
            stride,
            scale_inv,
            bf16,
            m,
            bias,
            maximum,
            max16,
            block,
            grid=(triton.cdiv(total, block),),
            warmup=False,
            enable_fp_fusion=False,
            Q=exponent - 1 if abs(fraction) == 0.5 else 99,
        )
    return out.view(fp8_dtype)


@triton.jit
def _encode_old(v, M: tl.constexpr, BIAS: tl.constexpr, MAX: tl.constexpr):
    raw = v.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    mag = raw & 2147483647
    limit = tl.full((), MAX, tl.float32).to(tl.uint32, bitcast=True)
    mag = tl.minimum(mag, limit)
    exp = (mag >> 23).to(tl.int32) - 127
    normal = (mag + ((1 << 22 - M) - 1) + (mag >> 23 - M & 1) >> 23 - M) - (
        127 - BIAS << M
    )
    shift = 24 - BIAS - M - exp
    safe_shift = tl.minimum(tl.maximum(shift, 1), 24)
    mant = mag & 8388607 | 8388608
    sub = (
        mant + ((1 << safe_shift - 1) - 1) + (mant >> safe_shift & 1)
        >> safe_shift
    )
    sub = tl.where(shift > 24, 0, sub)
    code = tl.where(exp >= 1 - BIAS, normal, sub)
    return (code | sign).to(tl.uint8)


@triton.jit
def _kernel_old(
    X,
    Y,
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    M: tl.constexpr,
    BIAS: tl.constexpr,
    MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = (
        tl.program_id(0).to(tl.int64)
        if (TOTAL + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
        or ((TOTAL + BLOCK - 1) // BLOCK * BLOCK - 1) // N * STRIDE + N - 1
        > 2147483647
        else tl.program_id(0)
    ) * BLOCK + tl.arange(0, BLOCK)
    src = off // N * STRIDE + off % N
    v = tl.load(X + src, off < TOTAL, other=0).to(tl.float32)
    if SCALE != 1.0:
        v = v * SCALE
    code = _encode_old(v, M, BIAS, MAX)
    if SCALE != 1.0:
        original = tl.load(X + src, off < TOTAL, other=0).to(tl.float32)
        input_sign = original.to(tl.int32, bitcast=True) >> 24 & 128
        scale_sign = (
            tl.full((), SCALE, tl.float32).to(tl.int32, bitcast=True) >> 24
            & 128
        )
        code = (code.to(tl.int32) & 127 | input_sign ^ scale_sign).to(tl.uint8)
    tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _pair_kernel(
    X,
    Y,
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    E4: tl.constexpr,
    SHIFT: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    X = X.to(tl.pointer_type(tl.int32))
    Y = Y.to(tl.pointer_type(tl.int16))
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = (7 if E4 else 15) + SHIFT
    MAX16: tl.constexpr = (17376 if E4 else 18272) - 128 * SHIFT
    off = (
        tl.program_id(0).to(tl.int64)
        if (TOTAL + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
        or ((TOTAL + BLOCK - 1) // BLOCK * BLOCK - 1) // N * STRIDE + N - 1
        > 2147483647
        else tl.program_id(0)
    ) * BLOCK + tl.arange(0, BLOCK)
    src = off if N == STRIDE else off // N * STRIDE + off % N
    if TOTAL % BLOCK == 0 and (STRIDE == N or BLOCK <= N):
        raw = tl.load(X + src, offset_state_policy="continuous")
    else:
        raw = tl.load(X + src, off < TOTAL, other=0)
    a = tl.minimum(raw & 32767, MAX16)
    b = tl.minimum(raw >> 16 & 32767, MAX16)
    addend: tl.constexpr = (
        8388608.0 - (127 - BIAS << M) - 8388608.0 * 2.0 ** (M - 7)
    )
    an = ((a | 143 + M << 23).to(tl.float32, bitcast=True) + addend).to(
        tl.int32, bitcast=True
    )
    bn = ((b | 143 + M << 23).to(tl.float32, bitcast=True) + addend).to(
        tl.int32, bitcast=True
    )
    magic: tl.constexpr = 2.0 ** (24 - BIAS - M)
    sa = ((a << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.int32, bitcast=True
    )
    sb = ((b << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.int32, bitcast=True
    )
    ca = (tl.where(a < 128 - BIAS << 7, sa, an) & 127).to(tl.uint16)
    cb = (tl.where(b < 128 - BIAS << 7, sb, bn) & 127).to(tl.uint16)
    signs = (raw >> 8 & 128 | raw >> 16 & 32768).to(tl.uint16)
    code = (ca | cb << 8 | signs) ^ (32896 if NEG else 0)
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code.to(tl.int16), offset_state_policy="continuous")
    else:
        tl.store(Y + off, code.to(tl.int16), off < TOTAL)


def _general_quantize(x, scale_inv, fp8_dtype):
    total = x.numel()
    if total <= 65536:
        return _small(x, scale_inv, fp8_dtype, total)
    n = x.shape[-1]
    stride = _rs(x, n)
    if n == 1 and total >= 1048576 and (total % 2 == 0) and x.is_contiguous():
        n = total
        stride = total
    mantissa, exponent = math.frexp(scale_inv)
    shift = exponent - 1
    if (
        x.dtype != torch.bfloat16
        or abs(mantissa) != 0.5
        or shift < -8
        or (shift > 8)
        or n % 2
        or stride % 2
        or x.storage_offset() % 2
    ):
        return _unpaired(x, scale_inv, fp8_dtype)
    out = torch.empty_like(
        x, dtype=torch.uint8, memory_format=torch.contiguous_format
    )
    total = x.numel() // 2
    source = x
    destination = out
    block = (
        (32768 if total >= 4194304 else 16384)
        if stride == n
        else (
            (8192 if 64 <= n <= 1024 and total % 8192 == 0 else 2048)
            if total >= 524288 and n <= 1024
            else max(512, min(8192, n // 2))
        )
    )
    if total:
        _pair_kernel.run(
            source,
            destination,
            total,
            n // 2,
            stride // 2,
            fp8_dtype == torch.float8_e4m3fn,
            shift,
            scale_inv < 0.0,
            block,
            enable_fp_fusion=False,
            grid=(triton.cdiv(total, block),),
            warmup=False,
        )
    return out.view(fp8_dtype)


def _small(x, scale_inv, fp8_dtype, total):
    out = torch.empty_like(
        x, dtype=fp8_dtype, memory_format=torch.contiguous_format
    )
    if total:
        if (
            scale_inv == 1.0
            and fp8_dtype == torch.float8_e4m3fn
            and x.is_contiguous()
        ):
            _j8.run(
                x,
                out.data_ptr(),
                total,
                grid=((total + 4095) // 4096,),
                warmup=False,
            )
        else:
            n = x.shape[-1]
            stride = _rs(x, n)
            if x.dtype == torch.bfloat16:
                _j16.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total,
                    n,
                    stride,
                    scale_inv,
                    fp8_dtype == torch.float8_e4m3fn,
                    grid=((total + 4095) // 4096,),
                    warmup=False,
                )
            else:
                _j17.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total,
                    n,
                    stride,
                    scale_inv,
                    fp8_dtype == torch.float8_e4m3fn,
                    grid=((total + 4095) // 4096,),
                    warmup=False,
                )
    return out


@triton.jit
def _contig_e4(X, Y, TOTAL: tl.constexpr):
    BLOCK: tl.constexpr = 32768 if TOTAL >= 4194304 else 16384
    _pair_kernel(X, Y, TOTAL, TOTAL, TOTAL, True, 0, False, BLOCK)


@triton.jit
def _contig_e5(X, Y, TOTAL: tl.constexpr):
    BLOCK: tl.constexpr = 32768 if TOTAL >= 4194304 else 16384
    _pair_kernel(X, Y, TOTAL, TOTAL, TOTAL, False, 0, False, BLOCK)


def fp8_quantize(x, scale_inv, fp8_dtype):
    total = x.numel()
    if total <= 262144 and scale_inv != 0.0:
        return _small(x, scale_inv, fp8_dtype, total)
    if (
        total >= 1048576
        and total % 131072 == 0
        and (x.dtype == torch.float16)
        and x.is_contiguous()
        and (x.data_ptr() % 8 == 0)
        and (abs(scale_inv) == 1.0)
    ):
        out = torch.empty_like(
            x, dtype=fp8_dtype, memory_format=torch.contiguous_format
        )
        _half_quad_v1169.run(
            x.data_ptr(),
            out.data_ptr(),
            total // 4,
            fp8_dtype == torch.float8_e4m3fn,
            scale_inv < 0.0,
            (
                8192
                if fp8_dtype == torch.float8_e4m3fn or total < 16777216
                else 32768
            ),
            grid=(
                total
                // (
                    4
                    * (
                        8192
                        if fp8_dtype == torch.float8_e4m3fn or total < 16777216
                        else 32768
                    )
                ),
            ),
            warmup=False,
            enable_fp_fusion=False,
        )
        return out
    if (
        (
            total >= 1048576
            and total % 4 == 0
            and (x.dtype == torch.bfloat16)
            and x.is_contiguous()
            and (x.data_ptr() % 8 == 0)
        )
        and scale_inv != 0.0
    ) and total % (32768 if total < 16777216 else 131072) == 0:
        mantissa, exponent = math.frexp(scale_inv)
        shift = exponent - 1
        if abs(mantissa) == 0.5 and -8 <= shift <= 8:
            out = torch.empty_like(
                x, dtype=fp8_dtype, memory_format=torch.contiguous_format
            )
            _quad_pack_v1154.run(
                x.data_ptr(),
                out.data_ptr(),
                total // 4,
                fp8_dtype == torch.float8_e4m3fn,
                shift,
                scale_inv < 0.0,
                8192 if total < 16777216 else 32768,
                grid=(
                    triton.cdiv(
                        total // 4, 8192 if total < 16777216 else 32768
                    ),
                ),
                warmup=False,
                enable_fp_fusion=False,
            )
            return out
    if (
        total >= 1048576
        and x.dtype == torch.float16
        and (scale_inv == -1.0)
        and x.is_contiguous()
    ):
        e4 = fp8_dtype == torch.float8_e4m3fn
        if total >= 4194304 and (x.data_ptr() % 4 == 2 or total % 65536 != 0):
            prefix = 1 if x.data_ptr() % 4 == 2 else 0
            storage = torch.empty(
                total + prefix, dtype=torch.uint8, device=x.device
            )
            out = storage[prefix:].view(fp8_dtype).view(x.shape)
            full = (total - prefix) // 65536 * 65536
            rem = total - prefix - full
            if prefix:
                _half_negative_scalar_v1127.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    1,
                    e4,
                    1024,
                    grid=(1,),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            _half_negative_pair_v1127.run(
                x.data_ptr() + prefix * 2,
                out.data_ptr() + prefix,
                full // 2,
                e4,
                32768,
                grid=(full // 65536,),
                warmup=False,
                enable_fp_fusion=False,
            )
            if rem:
                _half_negative_scalar_v1127.run(
                    x.data_ptr() + (full + prefix) * 2,
                    out.data_ptr() + full + prefix,
                    rem,
                    e4,
                    1024,
                    grid=(triton.cdiv(rem, 1024),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            return out
        out = torch.empty_like(
            x, dtype=fp8_dtype, memory_format=torch.contiguous_format
        )
        if total >= 2097152 and total % 2 == 0 and (x.data_ptr() % 4 == 0):
            _half_negative_pair_v1127.run(
                x.data_ptr(),
                out.data_ptr(),
                total // 2,
                e4,
                32768,
                grid=(triton.cdiv(total // 2, 32768),),
                warmup=False,
                enable_fp_fusion=False,
            )
        else:
            _half_negative_scalar_v1127.run(
                x.data_ptr(),
                out.data_ptr(),
                total,
                e4,
                32768,
                grid=(triton.cdiv(total, 32768),),
                warmup=False,
                enable_fp_fusion=False,
            )
        return out
    if (
        total >= 16777216
        and scale_inv != 0.0
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 2)
    ):
        fraction, exponent = math.frexp(scale_inv)
        if abs(fraction) != 0.5:
            storage = torch.empty(
                total + 1, dtype=torch.uint8, device=x.device
            )
            out = storage[1:].view(fp8_dtype).view(x.shape)
            full = (total - 1) // 32768 * 32768
            rem = total - 1 - full
            _general_tail_v1107.run(
                x.data_ptr(),
                out.data_ptr(),
                1,
                x.dtype == torch.bfloat16,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                grid=(1,),
                warmup=False,
                enable_fp_fusion=False,
            )
            _general_full_v1107.run(
                x.data_ptr() + 2,
                out.data_ptr() + 1,
                full // 2,
                x.dtype == torch.bfloat16,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                16384,
                grid=(full // 32768,),
                warmup=False,
                enable_fp_fusion=False,
            )
            if rem:
                _general_tail_v1107.run(
                    x.data_ptr() + (full + 1) * 2,
                    out.data_ptr() + full + 1,
                    rem,
                    x.dtype == torch.bfloat16,
                    scale_inv,
                    fp8_dtype == torch.float8_e4m3fn,
                    grid=(triton.cdiv(rem, 1024),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            return out
    if (
        total >= (4194304 if fp8_dtype == torch.float8_e4m3fn else 16777216)
        and scale_inv == 1.0
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 2)
    ):
        storage = torch.empty(total + 1, dtype=torch.uint8, device=x.device)
        out = storage[1:].view(fp8_dtype).view(x.shape)
        full = (total - 1) // 65536 * 65536
        rem = total - 1 - full
        full_fn = (
            _bf16_full_v1103 if x.dtype == torch.bfloat16 else _half_full_v1106
        )
        tail_fn = (
            _bf16_tail_v1103 if x.dtype == torch.bfloat16 else _half_tail_v1106
        )
        tail_fn.run(
            x.data_ptr(),
            out.data_ptr(),
            1,
            fp8_dtype == torch.float8_e4m3fn,
            grid=(1,),
            warmup=False,
            enable_fp_fusion=False,
        )
        full_fn.run(
            x.data_ptr() + 2,
            out.data_ptr() + 1,
            full // 2,
            fp8_dtype == torch.float8_e4m3fn,
            32768,
            grid=(full // 65536,),
            warmup=False,
            enable_fp_fusion=False,
        )
        if rem:
            tail_fn.run(
                x.data_ptr() + 2 + full * 2,
                out.data_ptr() + 1 + full,
                rem,
                fp8_dtype == torch.float8_e4m3fn,
                grid=(triton.cdiv(rem, 1024),),
                warmup=False,
                enable_fp_fusion=False,
            )
        return out
    if (
        total >= 4194304
        and x.dtype == torch.float16
        and (0.00390625 <= abs(scale_inv) <= 0.5)
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 2)
    ):
        fraction, exponent = math.frexp(abs(scale_inv))
        if fraction == 0.5:
            storage = torch.empty(
                total + 1, dtype=torch.uint8, device=x.device
            )
            out = storage[1:].view(fp8_dtype).view(x.shape)
            full = (total - 1) // 65536 * 65536
            rem = total - 1 - full
            _half_down_tail_v1111.run(
                x.data_ptr(),
                out.data_ptr(),
                1,
                fp8_dtype == torch.float8_e4m3fn,
                1 - exponent,
                scale_inv < 0.0,
                grid=(1,),
                warmup=False,
                enable_fp_fusion=False,
            )
            _half_down_full_v1111.run(
                x.data_ptr() + 2,
                out.data_ptr() + 1,
                full // 2,
                fp8_dtype == torch.float8_e4m3fn,
                1 - exponent,
                scale_inv < 0.0,
                grid=(full // 65536,),
                warmup=False,
                enable_fp_fusion=False,
            )
            if rem:
                _half_down_tail_v1111.run(
                    x.data_ptr() + (full + 1) * 2,
                    out.data_ptr() + full + 1,
                    rem,
                    fp8_dtype == torch.float8_e4m3fn,
                    1 - exponent,
                    scale_inv < 0.0,
                    grid=(triton.cdiv(rem, 1024),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            return out
    if (
        total >= 4194304
        and x.dtype == torch.float16
        and (
            2.0
            <= abs(scale_inv)
            <= (256.0 if fp8_dtype == torch.float8_e4m3fn else 8.0)
        )
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 2)
    ):
        fraction, exponent = math.frexp(abs(scale_inv))
        if fraction == 0.5:
            storage = torch.empty(
                total + 1, dtype=torch.uint8, device=x.device
            )
            out = storage[1:].view(fp8_dtype).view(x.shape)
            full = (total - 1) // 65536 * 65536
            rem = total - 1 - full
            _half_up_tail_v1112.run(
                x.data_ptr(),
                out.data_ptr(),
                1,
                fp8_dtype == torch.float8_e4m3fn,
                exponent - 1,
                scale_inv < 0.0,
                grid=(1,),
                warmup=False,
                enable_fp_fusion=False,
            )
            _half_up_full_v1112.run(
                x.data_ptr() + 2,
                out.data_ptr() + 1,
                full // 2,
                fp8_dtype == torch.float8_e4m3fn,
                exponent - 1,
                scale_inv < 0.0,
                grid=(full // 65536,),
                warmup=False,
                enable_fp_fusion=False,
            )
            if rem:
                _half_up_tail_v1112.run(
                    x.data_ptr() + (full + 1) * 2,
                    out.data_ptr() + full + 1,
                    rem,
                    fp8_dtype == torch.float8_e4m3fn,
                    exponent - 1,
                    scale_inv < 0.0,
                    grid=(triton.cdiv(rem, 1024),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            return out
    if (
        total >= 4194304
        and x.dtype == torch.bfloat16
        and (scale_inv != 1.0)
        and (0.00390625 <= abs(scale_inv) <= 256.0)
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 2)
    ):
        fraction, exponent = math.frexp(abs(scale_inv))
        if fraction == 0.5:
            storage = torch.empty(
                total + 1, dtype=torch.uint8, device=x.device
            )
            out = storage[1:].view(fp8_dtype).view(x.shape)
            full = (total - 1) // 65536 * 65536
            rem = total - 1 - full
            _bf16_power_tail_v1113.run(
                x.data_ptr(),
                out.data_ptr(),
                1,
                fp8_dtype == torch.float8_e4m3fn,
                scale_inv,
                grid=(1,),
                warmup=False,
                enable_fp_fusion=False,
            )
            _bf16_power_full_v1113.run(
                x.data_ptr() + 2,
                out.data_ptr() + 1,
                full // 2,
                fp8_dtype == torch.float8_e4m3fn,
                exponent - 1,
                scale_inv < 0.0,
                grid=(full // 65536,),
                warmup=False,
                enable_fp_fusion=False,
            )
            if rem:
                _bf16_power_tail_v1113.run(
                    x.data_ptr() + (full + 1) * 2,
                    out.data_ptr() + full + 1,
                    rem,
                    fp8_dtype == torch.float8_e4m3fn,
                    scale_inv,
                    grid=(triton.cdiv(rem, 1024),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            return out
    if (
        total >= (4194304 if total % 2 else 16777216)
        and x.dtype == torch.float16
        and (0.00390625 <= abs(scale_inv) <= 0.5)
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 0)
        and (total % 65536 != 0)
    ):
        fraction, exponent = math.frexp(abs(scale_inv))
        if fraction == 0.5:
            out = torch.empty_like(
                x, dtype=fp8_dtype, memory_format=torch.contiguous_format
            )
            full = total // 65536 * 65536
            rem = total - full
            _half_down_full_v1111.run(
                x.data_ptr(),
                out.data_ptr(),
                full // 2,
                fp8_dtype == torch.float8_e4m3fn,
                1 - exponent,
                scale_inv < 0.0,
                grid=(full // 65536,),
                warmup=False,
                enable_fp_fusion=False,
            )
            _half_down_tail_v1111.run(
                x.data_ptr() + full * 2,
                out.data_ptr() + full,
                rem,
                fp8_dtype == torch.float8_e4m3fn,
                1 - exponent,
                scale_inv < 0.0,
                grid=(triton.cdiv(rem, 1024),),
                warmup=False,
                enable_fp_fusion=False,
            )
            return out
    if (
        total >= (4194304 if total % 2 else 16777216)
        and x.dtype == torch.float16
        and (
            2.0
            <= abs(scale_inv)
            <= (256.0 if fp8_dtype == torch.float8_e4m3fn else 8.0)
        )
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 0)
        and (total % 65536 != 0)
    ):
        fraction, exponent = math.frexp(abs(scale_inv))
        if fraction == 0.5:
            out = torch.empty_like(
                x, dtype=fp8_dtype, memory_format=torch.contiguous_format
            )
            full = total // 65536 * 65536
            rem = total - full
            _half_up_full_v1112.run(
                x.data_ptr(),
                out.data_ptr(),
                full // 2,
                fp8_dtype == torch.float8_e4m3fn,
                exponent - 1,
                scale_inv < 0.0,
                grid=(full // 65536,),
                warmup=False,
                enable_fp_fusion=False,
            )
            _half_up_tail_v1112.run(
                x.data_ptr() + full * 2,
                out.data_ptr() + full,
                rem,
                fp8_dtype == torch.float8_e4m3fn,
                exponent - 1,
                scale_inv < 0.0,
                grid=(triton.cdiv(rem, 1024),),
                warmup=False,
                enable_fp_fusion=False,
            )
            return out
    if (
        total >= (4194304 if total % 2 else 16777216)
        and x.dtype == torch.bfloat16
        and (scale_inv != 1.0)
        and (0.00390625 <= abs(scale_inv) <= 256.0)
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 0)
        and (total % 65536 != 0)
    ):
        fraction, exponent = math.frexp(abs(scale_inv))
        if fraction == 0.5:
            out = torch.empty_like(
                x, dtype=fp8_dtype, memory_format=torch.contiguous_format
            )
            full = total // 65536 * 65536
            rem = total - full
            _bf16_power_full_v1113.run(
                x.data_ptr(),
                out.data_ptr(),
                full // 2,
                fp8_dtype == torch.float8_e4m3fn,
                exponent - 1,
                scale_inv < 0.0,
                grid=(full // 65536,),
                warmup=False,
                enable_fp_fusion=False,
            )
            _bf16_power_tail_v1113.run(
                x.data_ptr() + full * 2,
                out.data_ptr() + full,
                rem,
                fp8_dtype == torch.float8_e4m3fn,
                scale_inv,
                grid=(triton.cdiv(rem, 1024),),
                warmup=False,
                enable_fp_fusion=False,
            )
            return out
    if (
        (total >= 16777216 and total % 2 != 0)
        and scale_inv != 0.0
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 0)
        and (total % 32768 != 0)
    ):
        fraction, exponent = math.frexp(scale_inv)
        if abs(fraction) != 0.5:
            out = torch.empty_like(
                x, dtype=fp8_dtype, memory_format=torch.contiguous_format
            )
            full = total // 32768 * 32768
            rem = total - full
            _general_full_v1107.run(
                x.data_ptr(),
                out.data_ptr(),
                full // 2,
                x.dtype == torch.bfloat16,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                16384,
                grid=(full // 32768,),
                warmup=False,
                enable_fp_fusion=False,
            )
            _general_tail_v1107.run(
                x.data_ptr() + full * 2,
                out.data_ptr() + full,
                rem,
                x.dtype == torch.bfloat16,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                grid=(triton.cdiv(rem, 1024),),
                warmup=False,
                enable_fp_fusion=False,
            )
            return out
    if (
        total >= 4194304
        and x.dtype == torch.float16
        and (scale_inv == 1.0)
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 0)
        and (total % 65536 != 0)
    ):
        out = torch.empty_like(
            x, dtype=fp8_dtype, memory_format=torch.contiguous_format
        )
        full = total // 65536 * 65536
        rem = total - full
        _half_full_v1106.run(
            x.data_ptr(),
            out.data_ptr(),
            full // 2,
            fp8_dtype == torch.float8_e4m3fn,
            32768,
            grid=(full // 65536,),
            warmup=False,
            enable_fp_fusion=False,
        )
        _half_tail_v1106.run(
            x.data_ptr() + full * 2,
            out.data_ptr() + full,
            rem,
            fp8_dtype == torch.float8_e4m3fn,
            grid=(triton.cdiv(rem, 1024),),
            warmup=False,
            enable_fp_fusion=False,
        )
        return out
    if (
        (total >= 8388608 or (total >= 4194304 and total % 2 != 0))
        and x.dtype == torch.bfloat16
        and (scale_inv == 1.0)
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 0)
        and (total % 65536 != 0)
    ):
        out = torch.empty_like(
            x, dtype=fp8_dtype, memory_format=torch.contiguous_format
        )
        full = total // 65536 * 65536
        rem = total - full
        _bf16_full_v1103.run(
            x.data_ptr(),
            out.data_ptr(),
            full // 2,
            fp8_dtype == torch.float8_e4m3fn,
            32768,
            grid=(full // 65536,),
            warmup=False,
            enable_fp_fusion=False,
        )
        _bf16_tail_v1103.run(
            x.data_ptr() + full * 2,
            out.data_ptr() + full,
            rem,
            fp8_dtype == torch.float8_e4m3fn,
            grid=(triton.cdiv(rem, 1024),),
            warmup=False,
            enable_fp_fusion=False,
        )
        return out
    if (
        total >= 1048576
        and scale_inv != 0.0
        and (not x.is_contiguous())
        and (x.size(-1) >= 2)
        and (x.data_ptr() % 4 == 0)
    ):
        n = x.size(-1)
        pitch = _rs(x, n)
        fraction, exponent = math.frexp(scale_inv)
        if (
            n % 2 == 0
            and pitch % 2 == 0
            and (
                abs(fraction) != 0.5
                or (
                    x.dtype == torch.float16
                    and -8
                    <= exponent - 1
                    <= (8 if fp8_dtype == torch.float8_e4m3fn else 3)
                )
            )
            and (
                n >= 2048 or n == 2 or (64 <= n <= 1024 and total % 16384 == 0)
            )
        ):
            out = torch.empty_like(
                x, dtype=fp8_dtype, memory_format=torch.contiguous_format
            )
            block = (
                min(n // 2, 32768)
                if n >= 2048
                else 8192 if n >= 64 and total >= 8388608 else 2048
            )
            _general_strided_pair_v1098.run(
                x.data_ptr(),
                out.data_ptr(),
                total // 2,
                n // 2,
                pitch // 2,
                x.dtype == torch.bfloat16,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                block,
                grid=(triton.cdiv(total // 2, block),),
                warmup=False,
                enable_fp_fusion=False,
                Q=exponent - 1 if abs(fraction) == 0.5 else 99,
            )
            return out
    if (
        total >= 1048576
        and x.dtype == torch.float16
        and x.is_contiguous()
        and (
            2.0
            <= abs(scale_inv)
            <= (256.0 if fp8_dtype == torch.float8_e4m3fn else 8.0)
        )
    ):
        fraction, exponent = math.frexp(abs(scale_inv))
        if fraction == 0.5:
            out = torch.empty_like(
                x, dtype=fp8_dtype, memory_format=torch.contiguous_format
            )
            if total >= 2097152 and total % 2 == 0 and (x.data_ptr() % 4 == 0):
                _half_up_pair_v901.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total // 2,
                    fp8_dtype == torch.float8_e4m3fn,
                    exponent - 1,
                    scale_inv < 0.0,
                    32768,
                    grid=(triton.cdiv(total // 2, 32768),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            else:
                _half_up_scalar_v1085.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total,
                    fp8_dtype == torch.float8_e4m3fn,
                    exponent - 1,
                    scale_inv < 0.0,
                    32768,
                    grid=(triton.cdiv(total, 32768),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            return out
    if (
        total >= 1048576
        and x.dtype == torch.float16
        and (-0.5 <= scale_inv <= -0.00390625)
        and x.is_contiguous()
    ):
        fraction, exponent = math.frexp(-scale_inv)
        if fraction == 0.5:
            out = torch.empty_like(
                x, dtype=fp8_dtype, memory_format=torch.contiguous_format
            )
            if total >= 2097152 and total % 2 == 0 and (x.data_ptr() % 4 == 0):
                _half_down_negative_pair_v1088.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total // 2,
                    fp8_dtype == torch.float8_e4m3fn,
                    1 - exponent,
                    32768,
                    grid=(triton.cdiv(total // 2, 32768),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            else:
                _half_down_negative_scalar_v1088.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total,
                    fp8_dtype == torch.float8_e4m3fn,
                    1 - exponent,
                    32768,
                    grid=(triton.cdiv(total, 32768),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            return out
    if scale_inv == 0.0:
        out = torch.empty_like(
            x, dtype=fp8_dtype, memory_format=torch.contiguous_format
        )
        if total:
            if (
                total >= 1048576
                and total % 2 == 0
                and x.is_contiguous()
                and (x.data_ptr() % 4 == 0)
            ):
                _zero_pair_v1081.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total // 2,
                    x.dtype == torch.bfloat16,
                    math.copysign(1.0, scale_inv) < 0.0,
                    16384,
                    grid=(triton.cdiv(total // 2, 16384),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            else:
                n = x.size(-1)
                pitch = _rs(x, n)
                block = (
                    32768
                    if x.is_contiguous() and total >= 1048576
                    else (
                        4096
                        if x.is_contiguous() or (total >= 262144 and n <= 1024)
                        else min(1024, n)
                    )
                )
                _zero_scalar_v1082.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total,
                    n,
                    pitch,
                    x.dtype == torch.bfloat16,
                    math.copysign(1.0, scale_inv) < 0.0,
                    block,
                    grid=(triton.cdiv(total, block),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
        return out
    if (
        total >= 1048576
        and x.dtype == torch.float16
        and (0.00390625 <= scale_inv <= 0.5)
        and x.is_contiguous()
    ):
        fraction, exponent = math.frexp(scale_inv)
        if fraction == 0.5:
            out = torch.empty_like(
                x, dtype=fp8_dtype, memory_format=torch.contiguous_format
            )
            if total >= 2097152 and total % 2 == 0 and (x.data_ptr() % 4 == 0):
                _half_pair_v1076.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total // 2,
                    fp8_dtype == torch.float8_e4m3fn,
                    1 - exponent,
                    32768,
                    grid=(triton.cdiv(total // 2, 32768),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            else:
                _half_scalar_v1076.run(
                    x.data_ptr(),
                    out.data_ptr(),
                    total,
                    fp8_dtype == torch.float8_e4m3fn,
                    1 - exponent,
                    32768,
                    grid=(triton.cdiv(total, 32768),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            return out
    if (
        total >= 1048576
        and total % 2 == 0
        and (scale_inv != 1.0)
        and x.is_contiguous()
        and (x.data_ptr() % 4 == 0)
    ):
        fraction, exponent = math.frexp(scale_inv)
        if abs(fraction) != 0.5:
            out = torch.empty_like(
                x, dtype=fp8_dtype, memory_format=torch.contiguous_format
            )
            block = (
                32768
                if total >= 33554432
                and (
                    x.dtype == torch.float16 or fp8_dtype == torch.float8_e5m2
                )
                else 16384
            )
            _general_pair_v1077.run(
                x.data_ptr(),
                out.data_ptr(),
                total // 2,
                x.dtype == torch.bfloat16,
                scale_inv,
                fp8_dtype == torch.float8_e4m3fn,
                block,
                grid=(triton.cdiv(total // 2, block),),
                warmup=False,
                enable_fp_fusion=False,
            )
            return out
    if (
        total >= 1048576
        and x.dtype == torch.float16
        and (scale_inv == 1.0)
        and x.is_contiguous()
    ):
        out = torch.empty_like(
            x, dtype=fp8_dtype, memory_format=torch.contiguous_format
        )
        if total >= 2097152 and total % 2 == 0 and (x.data_ptr() % 4 == 0):
            _half_pair_v471.run(
                x.data_ptr(),
                out.data_ptr(),
                total // 2,
                fp8_dtype == torch.float8_e4m3fn,
                32768,
                grid=(triton.cdiv(total // 2, 32768),),
                warmup=False,
                enable_fp_fusion=False,
            )
        else:
            _half_scalar_v471.run(
                x.data_ptr(),
                out.data_ptr(),
                total,
                fp8_dtype == torch.float8_e4m3fn,
                32768,
                grid=(triton.cdiv(total, 32768),),
                warmup=False,
                enable_fp_fusion=False,
            )
        return out
    if (
        1048576 <= total < 2147483648
        and x.dtype == torch.bfloat16
        and (x.size(-1) == 1)
        and (_rs(x, 1) == 2)
        and (x.data_ptr() % 4 == 0)
    ):
        out = torch.empty_like(
            x, dtype=fp8_dtype, memory_format=torch.contiguous_format
        )
        full = (total - 1) // 16384
        rem = total - full * 16384
        _pitch2_bulk_v1058.run(
            x.data_ptr(),
            out.data_ptr(),
            scale_inv,
            fp8_dtype == torch.float8_e4m3fn,
            16384,
            grid=(full,),
            warmup=False,
            enable_fp_fusion=False,
        )
        _pitch2_tail_v1058.run(
            x.data_ptr() + full * 16384 * 4,
            out.data_ptr() + full * 16384,
            rem,
            scale_inv,
            fp8_dtype == torch.float8_e4m3fn,
            grid=(triton.cdiv(rem, 1024),),
            warmup=False,
            enable_fp_fusion=False,
        )
        return out
    if (
        x.dtype == torch.bfloat16
        and scale_inv == 1.0
        and (total % 2 == 0)
        and x.is_contiguous()
        and (x.storage_offset() % 2 == 0)
    ):
        out = torch.empty_like(
            x, dtype=torch.uint8, memory_format=torch.contiguous_format
        )
        pairs = total // 2
        block = 32768 if pairs >= 4194304 else 16384
        if fp8_dtype == torch.float8_e4m3fn:
            _contig_e4.run(
                x, out, pairs, grid=(triton.cdiv(pairs, block),), warmup=False
            )
        else:
            _contig_e5.run(
                x, out, pairs, grid=(triton.cdiv(pairs, block),), warmup=False
            )
        return out.view(fp8_dtype)
    return _general_quantize(x, scale_inv, fp8_dtype)


@triton.jit
def _j0(raw):
    raw = raw.to(tl.int32)
    mag = raw & 32767
    normal = ((mag << 13) + (112 << 23)).to(tl.float32, bitcast=True)
    sub = (mag & 1023).to(tl.float32) * 5.960464477539063e-08
    value = tl.where(mag < 1024, sub, normal)
    return (value.to(tl.int32, bitcast=True) | (raw & 32768) << 16).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _j16(
    X,
    Y,
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
):
    X = X.to(tl.pointer_type(tl.bfloat16))
    _j7(X, Y, TOTAL, N, STRIDE, SCALE, E4)


@triton.jit
def _j17(
    X,
    Y,
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
):
    X = X.to(tl.pointer_type(tl.float16))
    _j7(X, Y, TOTAL, N, STRIDE, SCALE, E4)


@triton.jit
def _j18(v, M: tl.constexpr, BIAS: tl.constexpr, MAX16: tl.constexpr):
    raw = v.to(tl.float32).to(tl.int32, bitcast=True) >> 16
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
def _j2(v, M: tl.constexpr, BIAS: tl.constexpr, MAX: tl.constexpr):
    raw = v.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    mag = tl.minimum(
        raw & 2147483647,
        tl.full((), MAX, tl.float32).to(tl.uint32, bitcast=True),
    )
    normal = (mag + ((1 << 22 - M) - 1) + (mag >> 23 - M & 1) >> 23 - M) - (
        127 - BIAS << M
    )
    magic = tl.full((), 2.0 ** (24 - BIAS - M), tl.float32)
    sub = (mag.to(tl.float32, bitcast=True) + magic).to(
        tl.uint32, bitcast=True
    ) - magic.to(tl.uint32, bitcast=True)
    code = tl.where(mag < 128 - BIAS << 23, sub, normal)
    return (code | sign).to(tl.uint8)


@triton.jit
def _j7(
    X,
    Y,
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
):
    Y = Y.to(tl.pointer_type(tl.uint8))
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX: tl.constexpr = 448.0 if E4 else 57344.0
    off = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647 or (TOTAL // N - 1) * STRIDE + N - 1 > 2147483647
        else tl.program_id(0)
    ) * 4096 + tl.arange(0, 4096)
    src = off if STRIDE == N else off // N * STRIDE + off % N
    if X.dtype.element_ty == tl.bfloat16 and (SCALE == 1.0 or SCALE == 0.125):
        v = tl.load(X + src, off < TOTAL, other=0)
        SHIFT: tl.constexpr = 0 if SCALE == 1.0 else 3
        code = _j18(
            v, M, BIAS - SHIFT, (17376 if E4 else 18272) + (SHIFT << 7)
        )
    else:
        if X.dtype.element_ty == tl.float16:
            value = _j0(
                tl.load(
                    X.to(tl.pointer_type(tl.int16)) + src, off < TOTAL, other=0
                )
            )
        else:
            value = tl.load(X + src, off < TOTAL, other=0).to(tl.float32)
        original = value.to(tl.int32, bitcast=True)
        if SCALE != 1.0:
            value = value * SCALE
        code = _j2(value, M, BIAS, MAX)
        if SCALE != 1.0:
            sign = (
                original >> 24
                ^ tl.full((), SCALE, tl.float32).to(tl.int32, bitcast=True)
                >> 24
            ) & 128
            code = (code.to(tl.int32) & 127 | sign).to(tl.uint8)
    tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _j8(X, Y, TOTAL: tl.constexpr):
    _j7(X, Y, TOTAL, TOTAL, TOTAL, 1.0, True)


def _rs(x, n):
    for axis in range(x.ndim - 2, -1, -1):
        if x.size(axis) > 1:
            return x.stride(axis)
    return n


@triton.jit
def _pitch2_bulk_v1058(
    X: "*i32",
    Y: "*u8",
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX: tl.constexpr = 448.0 if E4 else 57344.0
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw = tl.load(X + off, offset_state_policy="continuous")
    v = (raw << 16).to(tl.float32, bitcast=True)
    if SCALE != 1.0:
        v = v * SCALE
    code = _j2(v, M, BIAS, MAX)
    if SCALE != 1.0:
        input_sign = raw.to(tl.int32) >> 8 & 128
        scale_sign = (
            tl.full((), SCALE, tl.float32).to(tl.int32, bitcast=True) >> 24
            & 128
        )
        code = (code.to(tl.int32) & 127 | input_sign ^ scale_sign).to(tl.uint8)
    tl.store(Y + off, code, offset_state_policy="continuous")


@triton.jit
def _pitch2_tail_v1058(
    X: "*i16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX: tl.constexpr = 448.0 if E4 else 57344.0
    off = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    raw = tl.load(X + off * 2, off < TOTAL, other=0).to(tl.int32)
    v = (raw << 16).to(tl.float32, bitcast=True)
    if SCALE != 1.0:
        v = v * SCALE
    code = _j2(v, M, BIAS, MAX)
    if SCALE != 1.0:
        input_sign = raw.to(tl.int32) >> 8 & 128
        scale_sign = (
            tl.full((), SCALE, tl.float32).to(tl.int32, bitcast=True) >> 24
            & 128
        )
        code = (code.to(tl.int32) & 127 | input_sign ^ scale_sign).to(tl.uint8)
    tl.store(Y + off, code, off < TOTAL)


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
        raw = tl.load(X + off, offset_state_policy="continuous").to(tl.uint32)
    else:
        raw = tl.load(X + off, off < TOTAL, other=0).to(tl.uint32)
    code = _half_code_v471(raw, E4).to(tl.uint8)
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
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
        raw = tl.load(X + off, offset_state_policy="continuous")
    else:
        raw = tl.load(X + off, off < TOTAL, other=0)
    code = (_half_code_v471(raw, E4) | _half_code_v471(raw >> 16, E4) << 8).to(
        tl.uint16
    )
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
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
def _half_scalar_v1076(
    X: "*u16",
    Y: "*u8",
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
        raw = tl.load(X + off, offset_state_policy="continuous").to(tl.uint32)
    else:
        raw = tl.load(X + off, off < TOTAL, other=0).to(tl.uint32)
    code = _half_code_v490(raw, E4, SHIFT).to(tl.uint8)
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _half_pair_v1076(
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
        raw = tl.load(X + off, offset_state_policy="continuous")
    else:
        raw = tl.load(X + off, off < TOTAL, other=0)
    code = (
        _half_code_v490(raw, E4, SHIFT)
        | _half_code_v490(raw >> 16, E4, SHIFT) << 8
    ).to(tl.uint16)
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _general_pair_v1077(
    X: "*u32",
    Y: "*u16",
    TOTAL: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = (
        tl.program_id(0).to(tl.int64)
        if (TOTAL + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    if TOTAL % BLOCK == 0:
        raw = tl.load(X + off, offset_state_policy="continuous")
    else:
        raw = tl.load(X + off, off < TOTAL, other=0)
    lo = raw & 65535
    hi = raw >> 16
    if BF16:
        a = ((lo & 32767) << 16).to(tl.float32, bitcast=True)
        b = ((hi & 32767) << 16).to(tl.float32, bitcast=True)
    else:
        a = _decode_fp16(lo & 32767)
        b = _decode_fp16(hi & 32767)
    a = a * (SCALE if SCALE >= 0.0 else -SCALE)
    b = b * (SCALE if SCALE >= 0.0 else -SCALE)
    ca = _encode_magnitude32_v1151(
        a, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
    ).to(tl.uint16)
    cb = _encode_magnitude32_v1151(
        b, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
    ).to(tl.uint16)
    scale_sign = (
        tl.full((), SCALE, tl.float32).to(tl.uint32, bitcast=True) >> 24 & 128
    )
    signs = lo >> 8 & 128 ^ scale_sign | (hi >> 8 & 128 ^ scale_sign) << 8
    code = ca & 127 | (cb & 127) << 8 | signs.to(tl.uint16)
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _zero_code_v1081(raw, BF16: tl.constexpr, NEG: tl.constexpr):
    sign = raw >> 8 & 128
    mag = raw & 32767
    inf: tl.constexpr = 32640 if BF16 else 31744
    return tl.where(
        mag > inf,
        sign | 127,
        tl.where(mag == inf, 255, sign ^ (128 if NEG else 0)),
    )


@triton.jit
def _zero_scalar_v1082(
    X: "*u16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    BF16: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = (
        tl.program_id(0).to(tl.int64)
        if (TOTAL + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
        or ((TOTAL + BLOCK - 1) // BLOCK * BLOCK - 1) // N * PITCH + N - 1
        > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    src = off // N * PITCH + off % N
    if TOTAL % BLOCK == 0 and (PITCH == N or BLOCK <= N):
        raw = tl.load(X + src, offset_state_policy="continuous").to(tl.uint32)
    else:
        raw = tl.load(X + src, off < TOTAL, other=0).to(tl.uint32)
    code = _zero_code_v1081(raw, BF16, NEG).to(tl.uint8)
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _zero_pair_v1081(
    X: "*u32",
    Y: "*u16",
    TOTAL: tl.constexpr,
    BF16: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = (
        tl.program_id(0).to(tl.int64)
        if (TOTAL + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    if TOTAL % BLOCK == 0:
        raw = tl.load(X + off, offset_state_policy="continuous")
    else:
        raw = tl.load(X + off, off < TOTAL, other=0)
    code = (
        _zero_code_v1081(raw, BF16, NEG)
        | _zero_code_v1081(raw >> 16, BF16, NEG) << 8
    ).to(tl.uint16)
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


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
    off = (
        tl.program_id(0).to(tl.int64)
        if (PAIRS + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
        else tl.program_id(0)
    ) * BLOCK + tl.arange(0, BLOCK)
    if PAIRS % BLOCK == 0:
        raw = tl.load(X + off, offset_state_policy="continuous")
    else:
        raw = tl.load(X + off, off < PAIRS, other=0)
    code = (
        _half_up_code_v901(raw, E4, Q)
        | _half_up_code_v901(raw >> 16, E4, Q) << 8
    ) ^ (32896 if NEG else 0)
    if PAIRS % BLOCK == 0:
        tl.store(Y + off, code.to(tl.uint16), offset_state_policy="continuous")
    else:
        tl.store(Y + off, code.to(tl.uint16), off < PAIRS)


@triton.jit
def _half_up_scalar_v1085(
    X: "*u16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    Q: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    if TOTAL % BLOCK == 0:
        raw = tl.load(X + off, offset_state_policy="continuous").to(tl.uint32)
    else:
        raw = tl.load(X + off, off < TOTAL, other=0).to(tl.uint32)
    code = (_half_up_code_v901(raw, E4, Q) ^ (128 if NEG else 0)).to(tl.uint8)
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _half_down_negative_scalar_v1088(
    X: "*u16",
    Y: "*u8",
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
        raw = tl.load(X + off, offset_state_policy="continuous").to(tl.uint32)
    else:
        raw = tl.load(X + off, off < TOTAL, other=0).to(tl.uint32)
    code = _half_code_v490(raw, E4, SHIFT).to(tl.uint8) ^ 128
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _half_down_negative_pair_v1088(
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
        raw = tl.load(X + off, offset_state_policy="continuous")
    else:
        raw = tl.load(X + off, off < TOTAL, other=0)
    code = (
        _half_code_v490(raw, E4, SHIFT)
        | _half_code_v490(raw >> 16, E4, SHIFT) << 8
    ).to(tl.uint16) ^ 32896
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _general_strided_pair_v1098(
    X: "*u32",
    Y: "*u16",
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
    Q: tl.constexpr = 99,
):
    wide: tl.constexpr = (
        TOTAL + BLOCK - 1
    ) // BLOCK * BLOCK - 1 > 2147483647 or (
        (TOTAL + BLOCK - 1) // BLOCK * BLOCK - 1
    ) // N * PITCH + N - 1 > 2147483647
    tile = tl.program_id(0).to(tl.int64) if wide else tl.program_id(0)
    off = tile * BLOCK + tl.arange(0, BLOCK)
    src = off // N * PITCH + off % N
    if TOTAL % BLOCK == 0 and BLOCK <= N:
        raw = tl.load(X + src, offset_state_policy="continuous")
    else:
        raw = tl.load(X + src, off < TOTAL, other=0)
    if not BF16 and (Q >= -8 and Q <= (8 if E4 else 3)):
        if Q == 0:
            ca = _half_code_v471(raw, E4)
            cb = _half_code_v471(raw >> 16, E4)
        elif Q < 0:
            ca = _half_code_v490(raw, E4, -Q)
            cb = _half_code_v490(raw >> 16, E4, -Q)
        else:
            ca = _half_up_code_v901(raw, E4, Q)
            cb = _half_up_code_v901(raw >> 16, E4, Q)
        code = (ca | cb << 8).to(tl.uint16) ^ (32896 if SCALE < 0.0 else 0)
    else:
        lo = raw & 65535
        hi = raw >> 16
        if BF16:
            a = ((lo & 32767) << 16).to(tl.float32, bitcast=True)
            b = ((hi & 32767) << 16).to(tl.float32, bitcast=True)
        else:
            a = _decode_fp16(lo & 32767)
            b = _decode_fp16(hi & 32767)
        a = a * (SCALE if SCALE >= 0.0 else -SCALE)
        b = b * (SCALE if SCALE >= 0.0 else -SCALE)
        ca = _encode_magnitude32_v1151(
            a, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
        ).to(tl.uint16)
        cb = _encode_magnitude32_v1151(
            b, 3 if E4 else 2, 7 if E4 else 15, 448.0 if E4 else 57344.0
        ).to(tl.uint16)
        scale_sign = (
            tl.full((), SCALE, tl.float32).to(tl.uint32, bitcast=True) >> 24
            & 128
        )
        signs = lo >> 8 & 128 ^ scale_sign | (hi >> 8 & 128 ^ scale_sign) << 8
        code = ca & 127 | (cb & 127) << 8 | signs.to(tl.uint16)
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _bf16_full_v1103(
    X: "*u32",
    Y: "*u16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _pair_kernel(X, Y, TOTAL, TOTAL, TOTAL, E4, 0, False, BLOCK)


@triton.jit
def _bf16_tail_v1103(
    X: "*u16", Y: "*u8", TOTAL: tl.constexpr, E4: tl.constexpr
):
    _kernel(
        X,
        Y,
        TOTAL,
        TOTAL,
        TOTAL,
        1.0,
        True,
        3 if E4 else 2,
        7 if E4 else 15,
        448.0 if E4 else 57344.0,
        17376 if E4 else 18272,
        1024,
    )


@triton.jit
def _half_full_v1106(
    X: "*u32",
    Y: "*u16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _half_pair_v471(X, Y, TOTAL, E4, BLOCK)


@triton.jit
def _half_tail_v1106(
    X: "*u16", Y: "*u8", TOTAL: tl.constexpr, E4: tl.constexpr
):
    _half_scalar_v471(X, Y, TOTAL, E4, 1024)


@triton.jit
def _general_full_v1107(
    X: "*u32",
    Y: "*u16",
    TOTAL: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _general_pair_v1077(X, Y, TOTAL, BF16, SCALE, E4, BLOCK)


@triton.jit
def _general_tail_v1107(
    X: "*u16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    BF16: tl.constexpr,
    SCALE: tl.constexpr,
    E4: tl.constexpr,
):
    _kernel(
        X,
        Y,
        TOTAL,
        TOTAL,
        TOTAL,
        SCALE,
        BF16,
        3 if E4 else 2,
        7 if E4 else 15,
        448.0 if E4 else 57344.0,
        17376 if E4 else 18272,
        1024,
    )


@triton.jit
def _half_down_full_v1111(
    X: "*u32",
    Y: "*u16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    SHIFT: tl.constexpr,
    NEG: tl.constexpr,
):
    if NEG:
        _half_down_negative_pair_v1088(X, Y, TOTAL, E4, SHIFT, 32768)
    else:
        _half_pair_v1076(X, Y, TOTAL, E4, SHIFT, 32768)


@triton.jit
def _half_down_tail_v1111(
    X: "*u16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    SHIFT: tl.constexpr,
    NEG: tl.constexpr,
):
    if NEG:
        _half_down_negative_scalar_v1088(X, Y, TOTAL, E4, SHIFT, 1024)
    else:
        _half_scalar_v1076(X, Y, TOTAL, E4, SHIFT, 1024)


@triton.jit
def _half_up_full_v1112(
    X: "*u32",
    Y: "*u16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    Q: tl.constexpr,
    NEG: tl.constexpr,
):
    _half_up_pair_v901(X, Y, TOTAL, E4, Q, NEG, 32768)


@triton.jit
def _half_up_tail_v1112(
    X: "*u16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    Q: tl.constexpr,
    NEG: tl.constexpr,
):
    _half_up_scalar_v1085(X, Y, TOTAL, E4, Q, NEG, 1024)


@triton.jit
def _bf16_power_full_v1113(
    X: "*u32",
    Y: "*u16",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    SHIFT: tl.constexpr,
    NEG: tl.constexpr,
):
    _pair_kernel(X, Y, TOTAL, TOTAL, TOTAL, E4, SHIFT, NEG, 32768)


@triton.jit
def _bf16_power_tail_v1113(
    X: "*u16",
    Y: "*u8",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    SCALE: tl.constexpr,
):
    _kernel(
        X,
        Y,
        TOTAL,
        TOTAL,
        TOTAL,
        SCALE,
        True,
        3 if E4 else 2,
        7 if E4 else 15,
        448.0 if E4 else 57344.0,
        17376 if E4 else 18272,
        1024,
    )


@triton.jit
def _half_negative_scalar_v1127(
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
        raw = tl.load(X + off, offset_state_policy="continuous").to(tl.uint32)
    else:
        raw = tl.load(X + off, off < TOTAL, other=0).to(tl.uint32)
    code = _half_code_v471(raw, E4).to(tl.uint8)
    code = code ^ 128
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _half_negative_pair_v1127(
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
        raw = tl.load(X + off, offset_state_policy="continuous")
    else:
        raw = tl.load(X + off, off < TOTAL, other=0)
    code = (_half_code_v471(raw, E4) | _half_code_v471(raw >> 16, E4) << 8).to(
        tl.uint16
    )
    code = code ^ 32896
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _encode_magnitude32_v1151(
    v, M: tl.constexpr, BIAS: tl.constexpr, MAX: tl.constexpr
):
    raw = v.to(tl.uint32, bitcast=True)
    mag = tl.minimum(
        raw, tl.full((), MAX, tl.float32).to(tl.uint32, bitcast=True)
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
    return code.to(tl.uint8)


@triton.jit
def _bf16_word_v1154(
    raw, E4: tl.constexpr, SHIFT: tl.constexpr, NEG: tl.constexpr
):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = (7 if E4 else 15) + SHIFT
    MAX16: tl.constexpr = (17376 if E4 else 18272) - 128 * SHIFT
    a = tl.minimum(raw & 32767, MAX16)
    b = tl.minimum(raw >> 16 & 32767, MAX16)
    addend: tl.constexpr = (
        8388608.0 - (127 - BIAS << M) - 8388608.0 * 2.0 ** (M - 7)
    )
    an = ((a | 143 + M << 23).to(tl.float32, bitcast=True) + addend).to(
        tl.int32, bitcast=True
    )
    bn = ((b | 143 + M << 23).to(tl.float32, bitcast=True) + addend).to(
        tl.int32, bitcast=True
    )
    magic: tl.constexpr = 2.0 ** (24 - BIAS - M)
    sa = ((a << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.int32, bitcast=True
    )
    sb = ((b << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.int32, bitcast=True
    )
    ca = (tl.where(a < 128 - BIAS << 7, sa, an) & 127).to(tl.uint16)
    cb = (tl.where(b < 128 - BIAS << 7, sb, bn) & 127).to(tl.uint16)
    signs = (raw >> 8 & 128 | raw >> 16 & 32768).to(tl.uint16)
    code = (ca | cb << 8 | signs) ^ (32896 if NEG else 0)
    return code.to(tl.uint32)


@triton.jit
def _quad_pack_v1154(
    X: "*u64",
    Y: "*u32",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    SHIFT: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = (
        tl.program_id(0).to(tl.int64)
        if (TOTAL + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    if TOTAL % BLOCK == 0:
        raw = tl.load(X + off, offset_state_policy="continuous")
    else:
        raw = tl.load(X + off, off < TOTAL, other=0)
    lo = raw.to(tl.uint32)
    hi = (raw >> 32).to(tl.uint32)
    a = _bf16_word_v1154(lo, E4, SHIFT, NEG)
    b = _bf16_word_v1154(hi, E4, SHIFT, NEG)
    code = a | b << 16
    if TOTAL % BLOCK == 0:
        tl.store(Y + off, code, offset_state_policy="continuous")
    else:
        tl.store(Y + off, code, off < TOTAL)


@triton.jit
def _half_quad_v1169(
    X: "*u64",
    Y: "*u32",
    TOTAL: tl.constexpr,
    E4: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = (
        tl.program_id(0).to(tl.int64)
        if TOTAL > 2147483647
        else tl.program_id(0)
    )
    off = tile * BLOCK + tl.arange(0, BLOCK)
    raw = tl.load(X + off, offset_state_policy="continuous")
    lo = raw.to(tl.uint32)
    hi = (raw >> 32).to(tl.uint32)
    a = _half_code_v471(lo, E4)
    b = _half_code_v471(lo >> 16, E4)
    c = _half_code_v471(hi, E4)
    d = _half_code_v471(hi >> 16, E4)
    code = (a | b << 8 | c << 16 | d << 24) ^ (2155905152 if NEG else 0)
    tl.store(Y + off, code, offset_state_policy="continuous")


__all__ = ["fp8_quantize"]
