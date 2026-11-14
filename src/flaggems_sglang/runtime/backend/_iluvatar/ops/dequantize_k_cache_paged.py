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
def _separate_rope(Q, P, Dest, META: tl.constexpr, PARTS: tl.constexpr):
    M: tl.constexpr = META[0]
    S0: tl.constexpr = META[1]
    SP: tl.constexpr = META[2]
    Q = Q.to(tl.pointer_type(tl.uint8))
    row = tl.program_id(0) // (PARTS + 1)
    part = tl.program_id(0) % (PARTS + 1)
    src = tl.load(P + row * SP)
    if M * S0 < 2147483648:
        src = src.to(tl.int32)
    else:
        src = src.to(tl.int64)
    src = tl.where(src < 0, src + M, src)
    base = Q + src * S0
    if part < PARTS:
        words = part * (128 // PARTS) + tl.arange(0, 128 // PARTS)
        packed = tl.load(base.to(tl.pointer_type(tl.uint32)) + words)
        shifts = tl.arange(0, 4) * 8
        raw = (packed[:, None] >> shifts[None, :] & 255).reshape(512 // PARTS)
        scale = tl.load(
            (base + 512).to(tl.pointer_type(tl.float32)) + part // (PARTS // 4)
        )
        magnitude = raw & 127
        normal = ((magnitude << 20) + 1006632960).to(tl.float32, bitcast=True)
        decoded = tl.where(
            magnitude < 8, tl.fma(normal, 2.0, -0.015625), normal
        )
        bits = decoded.to(tl.uint32, bitcast=True) | (raw & 128) << 24
        decoded = tl.where(magnitude == 127, 2143289344, bits).to(
            tl.float32, bitcast=True
        )
        col = part * (512 // PARTS) + tl.arange(0, 512 // PARTS)
        tl.store(Dest + row * 576 + col, decoded * scale)
    else:
        rc = tl.arange(0, 32)
        rope = tl.load((base + 528).to(tl.pointer_type(tl.uint32)) + rc)
        tl.store(
            Dest.to(tl.pointer_type(tl.uint32)) + row * 288 + 256 + rc, rope
        )


@triton.jit
def _kernel(Q, P, Dest, META: tl.constexpr):
    Q = Q.to(tl.pointer_type(tl.uint8))
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
    Q = Q.to(tl.pointer_type(tl.uint8))
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
    shifts = tl.arange(0, 4) * 8
    packed = tl.load(
        base.to(tl.pointer_type(tl.uint32))
        + groups[:, None] * 32
        + words[None, :]
    )
    raw = (packed[:, :, None] >> shifts[None, None, :] & 255).reshape(4, 128)
    magnitude = raw & 127
    half_bits = (raw.to(tl.uint16) + (raw & 128).to(tl.uint16) << 7).to(
        tl.uint16
    )
    decoded = half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    decoded = tl.where(magnitude == 127, float("nan"), decoded)
    scales = tl.load((base + 512).to(tl.pointer_type(tl.float32)) + groups)
    value = decoded * scales[:, None]
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(Dest + row * 576 + cols, value)
    rc = tl.arange(0, 32)
    rope = tl.load((base + 528).to(tl.pointer_type(tl.uint32)) + rc)
    tl.store(
        (Dest + row * 576 + 512).to(tl.pointer_type(tl.uint32)) + rc, rope
    )


@triton.jit
def _packed_loop(Q, P, Dest, META: tl.constexpr):
    Q = Q.to(tl.pointer_type(tl.uint8))
    M: tl.constexpr = META[0]
    S0: tl.constexpr = META[1]
    SP: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    groups = tl.arange(0, 4)
    words = tl.arange(0, 32)
    shifts = tl.arange(0, 4) * 8
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
    )
    scales0 = tl.load(
        (base0 + 512).to(tl.pointer_type(tl.float32)) + groups, live0, other=0
    )
    rope0 = tl.load(
        (base0 + 528).to(tl.pointer_type(tl.uint32)) + rc, live0, other=0
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
    )
    scales1 = tl.load(
        (base1 + 512).to(tl.pointer_type(tl.float32)) + groups, live1, other=0
    )
    rope1 = tl.load(
        (base1 + 528).to(tl.pointer_type(tl.uint32)) + rc, live1, other=0
    )
    raw0 = (packed0[:, :, None] >> shifts[None, None, :] & 255).reshape(4, 128)
    magnitude0 = raw0 & 127
    half_bits0 = (raw0.to(tl.uint16) + (raw0 & 128).to(tl.uint16) << 7).to(
        tl.uint16
    )
    decoded0 = half_bits0.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    decoded0 = tl.where(magnitude0 == 127, float("nan"), decoded0)
    value0 = decoded0 * scales0[:, None]
    offset0 = (row0.to(tl.int64) if N * 576 >= 2147483648 else row0) * 576
    tl.store(Dest + offset0 + cols, value0, live0)
    tl.store(
        (Dest + offset0 + 512).to(tl.pointer_type(tl.uint32)) + rc,
        rope0,
        live0,
    )
    raw1 = (packed1[:, :, None] >> shifts[None, None, :] & 255).reshape(4, 128)
    magnitude1 = raw1 & 127
    half_bits1 = (raw1.to(tl.uint16) + (raw1 & 128).to(tl.uint16) << 7).to(
        tl.uint16
    )
    decoded1 = half_bits1.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    decoded1 = tl.where(magnitude1 == 127, float("nan"), decoded1)
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
    if (
        0 < n < 256
        and cache.stride(-1) == 1
        and (cache.stride(0) % 4 == 0)
        and (cache.storage_offset() % 4 == 0)
    ):
        _separate_rope.run(
            cache.view(torch.uint8),
            page_table_1_flattened,
            out,
            (
                cache.shape[0],
                cache.stride(0),
                page_table_1_flattened.stride(0),
            ),
            4,
            grid=(n * 5,),
            num_warps=1,
            warmup=False,
        )
        return out
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
                        2,
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
