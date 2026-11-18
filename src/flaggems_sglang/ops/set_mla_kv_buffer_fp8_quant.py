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

import torch
import triton
import triton.language as tl


@triton.jit
def _encode(
    value,
    M: tl.constexpr,
    BIAS: tl.constexpr,
    MAX: tl.constexpr,
    FNUZ: tl.constexpr,
):
    raw = value.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    magnitude = raw & 2147483647
    normal = (
        magnitude
        + tl.full(
            (), (1 << 22 - M) - 1 - (127 - BIAS << 23) & 4294967295, tl.uint32
        )
        + (magnitude >> 23 - M & 1)
        >> 23 - M
    )
    magic = tl.full((), 2.0 ** (24 - BIAS - M), tl.float32)
    sub = (magnitude.to(tl.float32, bitcast=True) + magic).to(
        tl.uint32, bitcast=True
    )
    code = tl.where(magnitude < 128 - BIAS << 23, sub, normal)
    if FNUZ:
        result = (code | sign).to(tl.uint8)
        result = tl.where(result & 127 == 0, 0, result)
        limit = 248.0 if M == 3 else 61440.0
        return tl.where(
            raw & 2147483647
            >= tl.full((), limit, tl.float32).to(tl.uint32, bitcast=True),
            128,
            result,
        ).to(tl.uint8)
    elif M == 3:
        code = tl.where(
            raw & 2147483647
            > tl.full((), 464.0, tl.float32).to(tl.uint32, bitcast=True),
            127,
            code,
        )
    else:
        code = tl.where(
            raw & 2147483647
            >= tl.full((), 61440.0, tl.float32).to(tl.uint32, bitcast=True),
            124,
            code,
        )
    code = tl.where(
        raw & 2147483647 > tl.full((), 2139095040, tl.uint32), 127, code
    )
    return (code | sign).to(tl.uint8)


@triton.jit
def _encode_bf16(
    value,
    M: tl.constexpr,
    BIAS: tl.constexpr,
    MAX: tl.constexpr,
    FNUZ: tl.constexpr,
):
    raw = value.to(tl.int32)
    magnitude = raw & 32767
    sign = raw >> 8 & 128
    addend: tl.constexpr = (
        8388608.0 - (127 - BIAS << M) - 8388608.0 * 2.0 ** (M - 7)
    )
    normal = (
        (magnitude | 143 + M << 23).to(tl.float32, bitcast=True) + addend
    ).to(tl.int32, bitcast=True)
    magic: tl.constexpr = 2.0 ** (24 - BIAS - M)
    sub = ((magnitude << 16).to(tl.float32, bitcast=True) + magic).to(
        tl.int32, bitcast=True
    )
    code = tl.where(magnitude < 128 - BIAS << 7, sub, normal)
    if FNUZ:
        result = (code | sign).to(tl.uint8)
        result = tl.where(result & 127 == 0, 0, result)
        return tl.where(
            magnitude >= (17272 if M == 3 else 18288), 128, result
        ).to(tl.uint8)
    elif M == 3:
        code = tl.where(magnitude > 17384, 127, code)
    else:
        code = tl.where(magnitude >= 18288, 124, code)
    code = tl.where(magnitude > 32640, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _encode_cuda(
    value,
    M: tl.constexpr,
    BIAS: tl.constexpr,
    MAX: tl.constexpr,
    FNUZ: tl.constexpr,
    LARGE: tl.constexpr,
):
    if LARGE and value.dtype == tl.bfloat16:
        return _encode_bf16(
            value.to(tl.uint16, bitcast=True).to(tl.uint32), M, BIAS, MAX, FNUZ
        )
    else:
        return _encode(value.to(tl.float32), M, BIAS, MAX, FNUZ)


@triton.jit
def _kernel(KV, LOC, NOPE, ROPE, META: tl.constexpr):
    T: tl.constexpr = META[0]
    KV0: tl.constexpr = META[1]
    L0: tl.constexpr = META[2]
    S: tl.constexpr = META[3]
    N0: tl.constexpr = META[4]
    N1: tl.constexpr = META[5]
    R0: tl.constexpr = META[6]
    R1: tl.constexpr = META[7]
    B: tl.constexpr = META[8]
    E4: tl.constexpr = META[9]
    FNUZ: tl.constexpr = META[10]
    KV = KV.to(tl.pointer_type(tl.uint8))
    if T >= 1024:
        row = tl.program_id(0) * 8 + tl.arange(0, 8)
        block = tl.program_id(1)
        loc = tl.load(LOC + row * L0, row < T, other=0).to(
            tl.int32
            if S < 2147483647 and (S - 1) * KV0 + 575 < 2147483647
            else tl.int64
        )
        loc = tl.where(loc < 0, loc + S, loc)
        cols = tl.arange(0, 128)
        if NOPE.dtype.element_ty == ROPE.dtype.element_ty:
            source = tl.where(block < 4, NOPE, ROPE)
            stride0 = tl.where(block < 4, N0, R0)
            stride1 = tl.where(block < 4, N1, R1)
            start = tl.where(block < 4, block * 128, 0)
            mask = (row[:, None] < T) & ((block < 4) | (cols[None, :] < 64))
            value = tl.load(
                source
                + row[:, None] * stride0
                + (start + cols[None, :]) * stride1,
                mask,
                other=0.0,
                cache_modifier=".cg",
            )
            encoded = _encode_cuda(
                value,
                3 if E4 else 2,
                (7 if E4 else 15) + FNUZ,
                (240.0 if FNUZ else 448.0) if E4 else 57344.0,
                FNUZ,
                T >= 8192,
            )
            tl.store(
                KV + loc[:, None] * KV0 + block * 128 + cols[None, :],
                encoded,
                mask,
                cache_modifier=".cs",
            )
        elif block < 4:
            col = block * 128 + cols
            value = tl.load(
                NOPE + row[:, None] * N0 + col[None, :] * N1,
                row[:, None] < T,
                other=0.0,
                cache_modifier=".cg",
            )
            encoded = _encode_cuda(
                value,
                3 if E4 else 2,
                (7 if E4 else 15) + FNUZ,
                (240.0 if FNUZ else 448.0) if E4 else 57344.0,
                FNUZ,
                T >= 8192,
            )
            tl.store(
                KV + loc[:, None] * KV0 + col[None, :],
                encoded,
                row[:, None] < T,
                cache_modifier=".cs",
            )
        else:
            mask = (row[:, None] < T) & (cols[None, :] < 64)
            value = tl.load(
                ROPE + row[:, None] * R0 + cols[None, :] * R1,
                mask,
                other=0.0,
                cache_modifier=".cg",
            )
            encoded = _encode_cuda(
                value,
                3 if E4 else 2,
                (7 if E4 else 15) + FNUZ,
                (240.0 if FNUZ else 448.0) if E4 else 57344.0,
                FNUZ,
                T >= 8192,
            )
            tl.store(
                KV + loc[:, None] * KV0 + 512 + cols[None, :],
                encoded,
                mask,
                cache_modifier=".cs",
            )
    else:
        pid = tl.program_id(0)
        blocks = tl.cdiv(576, B)
        row = pid // blocks
        block = pid % blocks
        cols = block * B + tl.arange(0, B)
        row_mask = row < T
        loc = tl.load(LOC + row * L0, mask=row_mask, other=0).to(
            tl.int32
            if S < 2147483647 and (S - 1) * KV0 + 575 < 2147483647
            else tl.int64
        )
        loc = tl.where(loc < 0, loc + S, loc)
        if NOPE.dtype.element_ty == ROPE.dtype.element_ty:
            source = tl.where(block < 4, NOPE, ROPE)
            stride0 = tl.where(block < 4, N0, R0)
            stride1 = tl.where(block < 4, N1, R1)
            start = tl.where(block < 4, block * B, 0)
            value = tl.load(
                source + row * stride0 + (start + tl.arange(0, B)) * stride1,
                row_mask & (cols < 576),
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
        else:
            nope_col = cols
            rope_col = cols - 512
            nope_mask = row_mask & (cols < 512)
            rope_mask = row_mask & (cols >= 512) & (cols < 576)
            nope = tl.load(
                NOPE + row * N0 + nope_col * N1,
                mask=nope_mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            rope = tl.load(
                ROPE + row * R0 + rope_col * R1,
                mask=rope_mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            value = tl.where(cols < 512, nope, rope)
        encoded = _encode(
            value,
            3 if E4 else 2,
            (7 if E4 else 15) + FNUZ,
            (240.0 if FNUZ else 448.0) if E4 else 57344.0,
            FNUZ,
        )
        tl.store(
            KV + loc * KV0 + cols,
            encoded,
            mask=row_mask & (cols < 576),
            cache_modifier=".cs",
        )


@triton.jit
def _kernel_hip(KV, LOC, NOPE, ROPE, META: tl.constexpr):
    T: tl.constexpr = META[0]
    KV0: tl.constexpr = META[1]
    L0: tl.constexpr = META[2]
    S: tl.constexpr = META[3]
    N0: tl.constexpr = META[4]
    N1: tl.constexpr = META[5]
    R0: tl.constexpr = META[6]
    R1: tl.constexpr = META[7]
    B: tl.constexpr = META[8]
    E4: tl.constexpr = META[9]
    FNUZ: tl.constexpr = META[10]
    KV = KV.to(tl.pointer_type(tl.uint8))
    if T >= 1024:
        row = tl.program_id(0) * 2 + tl.arange(0, 2)
        loc = tl.load(LOC + row * L0, row < T, other=0).to(
            tl.int32
            if S < 2147483647 and (S - 1) * KV0 + 575 < 2147483647
            else tl.int64
        )
        loc = tl.where(loc < 0, loc + S, loc)
        cols0 = 0 * 128 + tl.arange(0, 128)
        value0 = tl.load(
            NOPE + row[:, None] * N0 + cols0[None, :] * N1,
            row[:, None] < T,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        encoded0 = _encode(
            value0,
            3 if E4 else 2,
            (7 if E4 else 15) + FNUZ,
            (240.0 if FNUZ else 448.0) if E4 else 57344.0,
            FNUZ,
        )
        cols1 = 1 * 128 + tl.arange(0, 128)
        value1 = tl.load(
            NOPE + row[:, None] * N0 + cols1[None, :] * N1,
            row[:, None] < T,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        encoded1 = _encode(
            value1,
            3 if E4 else 2,
            (7 if E4 else 15) + FNUZ,
            (240.0 if FNUZ else 448.0) if E4 else 57344.0,
            FNUZ,
        )
        cols2 = 2 * 128 + tl.arange(0, 128)
        value2 = tl.load(
            NOPE + row[:, None] * N0 + cols2[None, :] * N1,
            row[:, None] < T,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        encoded2 = _encode(
            value2,
            3 if E4 else 2,
            (7 if E4 else 15) + FNUZ,
            (240.0 if FNUZ else 448.0) if E4 else 57344.0,
            FNUZ,
        )
        cols3 = 3 * 128 + tl.arange(0, 128)
        value3 = tl.load(
            NOPE + row[:, None] * N0 + cols3[None, :] * N1,
            row[:, None] < T,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        encoded3 = _encode(
            value3,
            3 if E4 else 2,
            (7 if E4 else 15) + FNUZ,
            (240.0 if FNUZ else 448.0) if E4 else 57344.0,
            FNUZ,
        )
        tl.store(
            KV + loc[:, None] * KV0 + cols0[None, :],
            encoded0,
            row[:, None] < T,
            cache_modifier=".cs",
        )
        tl.store(
            KV + loc[:, None] * KV0 + cols1[None, :],
            encoded1,
            row[:, None] < T,
            cache_modifier=".cs",
        )
        tl.store(
            KV + loc[:, None] * KV0 + cols2[None, :],
            encoded2,
            row[:, None] < T,
            cache_modifier=".cs",
        )
        tl.store(
            KV + loc[:, None] * KV0 + cols3[None, :],
            encoded3,
            row[:, None] < T,
            cache_modifier=".cs",
        )
        cols = tl.arange(0, 128)
        value = tl.load(
            ROPE + row[:, None] * R0 + cols[None, :] * R1,
            (row[:, None] < T) & (cols[None, :] < 64),
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        encoded = _encode(
            value,
            3 if E4 else 2,
            (7 if E4 else 15) + FNUZ,
            (240.0 if FNUZ else 448.0) if E4 else 57344.0,
            FNUZ,
        )
        tl.store(
            KV + loc[:, None] * KV0 + 512 + cols[None, :],
            encoded,
            (row[:, None] < T) & (cols[None, :] < 64),
            cache_modifier=".cs",
        )
    else:
        pid = tl.program_id(0)
        blocks = tl.cdiv(576, B)
        row = pid // blocks
        block = pid % blocks
        cols = block * B + tl.arange(0, B)
        row_mask = row < T
        loc = tl.load(LOC + row * L0, mask=row_mask, other=0).to(
            tl.int32
            if S < 2147483647 and (S - 1) * KV0 + 575 < 2147483647
            else tl.int64
        )
        loc = tl.where(loc < 0, loc + S, loc)
        nope_col = cols
        rope_col = cols - 512
        nope_mask = row_mask & (cols < 512)
        rope_mask = row_mask & (cols >= 512) & (cols < 576)
        nope = tl.load(
            NOPE + row * N0 + nope_col * N1,
            mask=nope_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        rope = tl.load(
            ROPE + row * R0 + rope_col * R1,
            mask=rope_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        value = tl.where(cols < 512, nope, rope)
        encoded = _encode(
            value,
            3 if E4 else 2,
            (7 if E4 else 15) + FNUZ,
            (240.0 if FNUZ else 448.0) if E4 else 57344.0,
            FNUZ,
        )
        tl.store(
            KV + loc * KV0 + cols,
            encoded,
            mask=row_mask & (cols < 576),
            cache_modifier=".cs",
        )


def set_mla_kv_buffer_fp8_quant(
    kv_buffer, loc, cache_k_nope, cache_k_rope, fp8_dtype
):
    rows = loc.shape[0]
    if rows:
        ns = cache_k_nope.stride()
        rs = cache_k_rope.stride()
        meta = (
            rows,
            kv_buffer.stride(0) * kv_buffer.element_size(),
            loc.stride(0),
            kv_buffer.shape[0],
            ns[0],
            ns[1],
            rs[0],
            rs[1],
            128,
            fp8_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz),
            fp8_dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz),
        )
        if torch.version.hip is None:
            _kernel.run(
                kv_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
                meta,
                grid=((rows + 7) // 8, 5) if rows >= 1024 else (rows * 5,),
                warmup=False,
            )
        else:
            _kernel_hip.run(
                kv_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
                meta,
                grid=((rows + 1) // 2 if rows >= 1024 else rows * 5,),
                warmup=False,
                num_warps=2 if rows >= 1024 else 4,
            )
    return kv_buffer


__all__ = ["set_mla_kv_buffer_fp8_quant"]
