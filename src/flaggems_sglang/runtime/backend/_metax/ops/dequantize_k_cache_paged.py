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
def _kernel(Q, P, Dest, META: tl.constexpr):
    N: tl.constexpr = META[0]
    M: tl.constexpr = META[1]
    S0: tl.constexpr = META[2]
    S1: tl.constexpr = META[3]
    SP: tl.constexpr = META[4]
    BR: tl.constexpr = META[5]
    ALIGNED: tl.constexpr = META[6]
    Q = Q.to(tl.pointer_type(tl.uint8))
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    src = tl.load(P + rows * SP, (N % BR == 0) | (rows < N), other=0).to(
        tl.int64
    )
    src = tl.where(src < 0, src + M, src)
    col = tl.arange(0, 512)
    base = Q + src[:, None] * S0
    raw = tl.load(
        base + col[None, :] * S1, (N % BR == 0) | (rows[:, None] < N), other=0
    ).to(tl.uint32)
    magnitude = raw & 127
    exponent = magnitude >> 3
    mantissa = magnitude & 7
    normal = ((magnitude << 20) + 1006632960).to(tl.float32, bitcast=True)
    decoded = tl.where(
        exponent == 0, mantissa.to(tl.float32) * 0.001953125, normal
    )
    decoded_bits = decoded.to(tl.uint32, bitcast=True) | (raw & 128) << 24
    decoded_bits = tl.where(magnitude == 127, 2143289344, decoded_bits)
    decoded = decoded_bits.to(tl.float32, bitcast=True)
    if ALIGNED:
        scales = (base + 512).to(tl.pointer_type(tl.float32))
        s0 = tl.load(scales, (N % BR == 0) | (rows[:, None] < N), other=0)
        s1 = tl.load(scales + 1, (N % BR == 0) | (rows[:, None] < N), other=0)
        s2 = tl.load(scales + 2, (N % BR == 0) | (rows[:, None] < N), other=0)
        s3 = tl.load(scales + 3, (N % BR == 0) | (rows[:, None] < N), other=0)
        scale = tl.where(
            col[None, :] < 256,
            tl.where(col[None, :] < 128, s0, s1),
            tl.where(col[None, :] < 384, s2, s3),
        )
    else:
        sc = 512 + col // 128 * 4
        b0 = tl.load(
            base + sc[None, :] * S1,
            (N % BR == 0) | (rows[:, None] < N),
            other=0,
        ).to(tl.uint32)
        b1 = tl.load(
            base + (sc[None, :] + 1) * S1,
            (N % BR == 0) | (rows[:, None] < N),
            other=0,
        ).to(tl.uint32)
        b2 = tl.load(
            base + (sc[None, :] + 2) * S1,
            (N % BR == 0) | (rows[:, None] < N),
            other=0,
        ).to(tl.uint32)
        b3 = tl.load(
            base + (sc[None, :] + 3) * S1,
            (N % BR == 0) | (rows[:, None] < N),
            other=0,
        ).to(tl.uint32)
        scale = (b0 | b1 << 8 | b2 << 16 | b3 << 24).to(
            tl.float32, bitcast=True
        )
    product = decoded * scale
    tl.store(
        Dest
        + (
            rows[:, None].to(tl.int64)
            if N * 576 >= 2147483648
            else rows[:, None]
        )
        * 576
        + col[None, :],
        product,
        (N % BR == 0) | (rows[:, None] < N),
    )
    rc = tl.arange(0, 64)
    if ALIGNED:
        packed = tl.load(
            (base + 528).to(tl.pointer_type(tl.uint16)) + rc[None, :],
            (N % BR == 0) | (rows[:, None] < N),
            other=0,
        )
    else:
        lo = tl.load(
            base + (528 + rc[None, :] * 2) * S1,
            (N % BR == 0) | (rows[:, None] < N),
            other=0,
        ).to(tl.uint32)
        hi = tl.load(
            base + (529 + rc[None, :] * 2) * S1,
            (N % BR == 0) | (rows[:, None] < N),
            other=0,
        ).to(tl.uint32)
        packed = (lo | hi << 8).to(tl.uint16)
    tl.store(
        Dest
        + (
            rows[:, None].to(tl.int64)
            if N * 576 >= 2147483648
            else rows[:, None]
        )
        * 576
        + 512
        + rc[None, :],
        packed.to(tl.bfloat16, bitcast=True),
        (N % BR == 0) | (rows[:, None] < N),
    )


@triton.jit
def _packed(Q, P, Dest, META: tl.constexpr):
    N: tl.constexpr = META[3]
    M: tl.constexpr = META[0]
    S0: tl.constexpr = META[1]
    SP: tl.constexpr = META[2]
    row = tl.program_id(0)
    src = tl.load(P + row * SP)
    if M * S0 < 2147483648:
        src = src.to(tl.int32)
    else:
        src = src.to(tl.int64)
    src = tl.where(src < 0, src + M, src)
    base = Q.to(tl.pointer_type(tl.uint8)) + src * S0
    groups = tl.arange(0, 4)
    words = tl.arange(0, 32)
    packed = tl.load(
        base.to(tl.pointer_type(tl.uint32))
        + groups[:, None] * 32
        + words[None, :]
    )
    nan_flags = ~((packed & 2139062143 ^ 2139062143) + 2139062143) & 2155905152
    pair0 = packed & 16711935
    pair1 = packed >> 8 & 16711935
    pair0 = pair0 + (pair0 & 8388736) << 7 | (nan_flags & 8388736) << 7
    pair1 = pair1 + (pair1 & 8388736) << 7 | (nan_flags >> 8 & 8388736) << 7
    carrier = tl.join(pair0, pair1)
    hs = tl.arange(0, 2) * 16
    half_words = carrier[:, :, :, None] >> hs[None, None, None, :] & 65535
    half_bits = (
        tl.permute(half_words, (0, 1, 3, 2)).reshape(4, 128).to(tl.uint16)
    )
    decoded = half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    scales = tl.load((base + 512).to(tl.pointer_type(tl.float32)) + groups)
    value = decoded * scales[:, None]
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(
        Dest
        + (row.to(tl.int64) if N * 576 >= 2147483648 else row) * 576
        + cols,
        value,
    )
    rc = tl.arange(0, 32)
    rope = tl.load((base + 528).to(tl.pointer_type(tl.uint32)) + rc)
    tl.store(
        (
            Dest
            + (row.to(tl.int64) if N * 576 >= 2147483648 else row) * 576
            + 512
        ).to(tl.pointer_type(tl.uint32))
        + rc,
        rope,
    )


@triton.jit
def _packed_loop(Q, P, Dest, META: tl.constexpr):
    M: tl.constexpr = META[0]
    S0: tl.constexpr = META[1]
    SP: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    groups = tl.arange(0, 4)
    words = tl.arange(0, 32)
    rc = tl.arange(0, 32)
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    row0 = tl.program_id(0) * 2 + 0
    live0 = (N % 2 == 0) | (row0 < N)
    src0 = tl.load(P + row0 * SP, live0, other=0)
    if M * S0 < 2147483648:
        src0 = src0.to(tl.int32)
    else:
        src0 = src0.to(tl.int64)
    src0 = tl.where(src0 < 0, src0 + M, src0)
    base0 = Q.to(tl.pointer_type(tl.uint8)) + src0 * S0
    packed0 = tl.load(
        base0.to(tl.pointer_type(tl.uint32))
        + groups[:, None] * 32
        + words[None, :],
        live0,
        other=0,
        cache_modifier=".cg",
    )
    scales0 = tl.load(
        (base0 + 512).to(tl.pointer_type(tl.float32)) + groups,
        live0,
        other=0,
        cache_modifier=".cg",
    )
    rope0 = tl.load(
        (base0 + 528).to(tl.pointer_type(tl.uint32)) + rc,
        live0,
        other=0,
        cache_modifier=".cg",
    )
    row1 = tl.program_id(0) * 2 + 1
    live1 = (N % 2 == 0) | (row1 < N)
    src1 = tl.load(P + row1 * SP, live1, other=0)
    if M * S0 < 2147483648:
        src1 = src1.to(tl.int32)
    else:
        src1 = src1.to(tl.int64)
    src1 = tl.where(src1 < 0, src1 + M, src1)
    base1 = Q.to(tl.pointer_type(tl.uint8)) + src1 * S0
    packed1 = tl.load(
        base1.to(tl.pointer_type(tl.uint32))
        + groups[:, None] * 32
        + words[None, :],
        live1,
        other=0,
        cache_modifier=".cg",
    )
    scales1 = tl.load(
        (base1 + 512).to(tl.pointer_type(tl.float32)) + groups,
        live1,
        other=0,
        cache_modifier=".cg",
    )
    rope1 = tl.load(
        (base1 + 528).to(tl.pointer_type(tl.uint32)) + rc,
        live1,
        other=0,
        cache_modifier=".cg",
    )
    nan_flags = (
        ~((packed0 & 2139062143 ^ 2139062143) + 2139062143) & 2155905152
    )
    pair0 = packed0 & 16711935
    pair1 = packed0 >> 8 & 16711935
    pair0 = pair0 + (pair0 & 8388736) << 7 | (nan_flags & 8388736) << 7
    pair1 = pair1 + (pair1 & 8388736) << 7 | (nan_flags >> 8 & 8388736) << 7
    carrier = tl.join(pair0, pair1)
    hs = tl.arange(0, 2) * 16
    half_words = carrier[:, :, :, None] >> hs[None, None, None, :] & 65535
    half_bits = (
        tl.permute(half_words, (0, 1, 3, 2)).reshape(4, 128).to(tl.uint16)
    )
    decoded0 = half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    value0 = decoded0 * scales0[:, None]
    offset0 = (row0.to(tl.int64) if N * 576 >= 2147483648 else row0) * 576
    tl.store(Dest + offset0 + cols, value0, live0)
    tl.store(
        (Dest + offset0 + 512).to(tl.pointer_type(tl.uint32)) + rc,
        rope0,
        live0,
    )
    nan_flags = (
        ~((packed1 & 2139062143 ^ 2139062143) + 2139062143) & 2155905152
    )
    pair0 = packed1 & 16711935
    pair1 = packed1 >> 8 & 16711935
    pair0 = pair0 + (pair0 & 8388736) << 7 | (nan_flags & 8388736) << 7
    pair1 = pair1 + (pair1 & 8388736) << 7 | (nan_flags >> 8 & 8388736) << 7
    carrier = tl.join(pair0, pair1)
    hs = tl.arange(0, 2) * 16
    half_words = carrier[:, :, :, None] >> hs[None, None, None, :] & 65535
    half_bits = (
        tl.permute(half_words, (0, 1, 3, 2)).reshape(4, 128).to(tl.uint16)
    )
    decoded1 = half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    value1 = decoded1 * scales1[:, None]
    offset1 = (row1.to(tl.int64) if N * 576 >= 2147483648 else row1) * 576
    tl.store(Dest + offset1 + cols, value1, live1)
    tl.store(
        (Dest + offset1 + 512).to(tl.pointer_type(tl.uint32)) + rc,
        rope1,
        live1,
    )


def dequantize_k_cache_paged(
    quant_k_cache, page_table_1_flattened, group_size=128
):
    if quant_k_cache.ndim == 3 and quant_k_cache.shape[1] == 1:
        cache = quant_k_cache
    else:
        cache = quant_k_cache.view(-1, quant_k_cache.shape[-1])
    n = page_table_1_flattened.numel()
    out = cache.new_empty((n, 1, 576), dtype=torch.bfloat16)
    if n:
        strides = cache.stride()
        br = 1
        if (
            strides[-1] == 1
            and strides[0] % 4 == 0
            and (cache.storage_offset() % 4 == 0)
        ):
            if n >= 4096:
                _packed_loop.run(
                    cache.view(torch.uint8),
                    page_table_1_flattened,
                    out,
                    (
                        cache.shape[0],
                        strides[0],
                        page_table_1_flattened.stride(0),
                        n,
                    ),
                    grid=(triton.cdiv(n, 2),),
                    num_warps=1,
                    warmup=False,
                )
            else:
                _packed.run(
                    cache.view(torch.uint8),
                    page_table_1_flattened,
                    out,
                    (
                        cache.shape[0],
                        strides[0],
                        page_table_1_flattened.stride(0),
                        n,
                    ),
                    grid=(n,),
                    num_warps=1,
                    warmup=False,
                )
            return out
        _kernel.run(
            cache.view(torch.uint8),
            page_table_1_flattened,
            out,
            (
                n,
                cache.shape[0],
                strides[0],
                strides[-1],
                page_table_1_flattened.stride(0),
                br,
                strides[-1] == 1
                and strides[0] % 4 == 0
                and (cache.storage_offset() % 4 == 0),
            ),
            num_warps=1,
            grid=(triton.cdiv(n, br),),
            warmup=False,
        )
    return out


__all__ = ["dequantize_k_cache_paged"]
