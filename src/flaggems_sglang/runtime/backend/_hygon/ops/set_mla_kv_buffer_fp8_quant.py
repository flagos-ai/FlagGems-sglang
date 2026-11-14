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
        row = tl.program_id(0) * 4 + tl.arange(0, 4)
        loc = tl.load(LOC + row * L0, row < T, other=0).to(
            tl.int32
            if S < 2147483647 and (S - 1) * KV0 + 575 < 2147483647
            else tl.int64
        )
        loc = tl.where(loc < 0, loc + S, loc)
        for block in tl.static_range(0, 4):
            cols = block * 128 + tl.arange(0, 128)
            value = tl.load(
                NOPE + row[:, None] * N0 + cols[None, :] * N1,
                row[:, None] < T,
                other=0.0,
            ).to(tl.float32)
            encoded = _encode(
                value,
                3 if E4 else 2,
                (7 if E4 else 15) + FNUZ,
                (240.0 if FNUZ else 448.0) if E4 else 57344.0,
                FNUZ,
            )
            tl.store(
                KV + loc[:, None] * KV0 + cols[None, :],
                encoded,
                row[:, None] < T,
            )
        cols = tl.arange(0, 128)
        value = tl.load(
            ROPE + row[:, None] * R0 + cols[None, :] * R1,
            (row[:, None] < T) & (cols[None, :] < 64),
            other=0.0,
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
            NOPE + row * N0 + nope_col * N1, mask=nope_mask, other=0.0
        ).to(tl.float32)
        rope = tl.load(
            ROPE + row * R0 + rope_col * R1, mask=rope_mask, other=0.0
        ).to(tl.float32)
        value = tl.where(cols < 512, nope, rope)
        encoded = _encode(
            value,
            3 if E4 else 2,
            (7 if E4 else 15) + FNUZ,
            (240.0 if FNUZ else 448.0) if E4 else 57344.0,
            FNUZ,
        )
        tl.store(KV + loc * KV0 + cols, encoded, mask=row_mask & (cols < 576))


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
        _kernel.run(
            kv_buffer,
            loc,
            cache_k_nope,
            cache_k_rope,
            meta,
            grid=((rows + 3) // 4 if rows >= 1024 else rows * 5,),
            warmup=False,
        )
    return kv_buffer


__all__ = ["set_mla_kv_buffer_fp8_quant"]
