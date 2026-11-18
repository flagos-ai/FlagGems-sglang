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


@triton.jit(do_not_specialize=["Q", "Dest"])
def _grouped(
    Q,
    P,
    Dest,
    META: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    N: tl.constexpr = META[5]
    M: tl.constexpr = META[0]
    S0: tl.constexpr = META[1]
    S1: tl.constexpr = META[2]
    SP: tl.constexpr = META[3]
    ALIGNED: tl.constexpr = META[4]
    row = tl.program_id(0)
    src = tl.load(P + row * SP)
    if M * S0 < 2147483648:
        src = src.to(tl.int32)
    else:
        src = src.to(tl.int64)
    if M & M - 1 == 0:
        src = src & M - 1
    else:
        src = tl.where(src < 0, src + M, src)
    base = Q + src * S0
    groups = tl.arange(0, 4)
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    raw = tl.load(base + cols * S1).to(tl.uint32)
    half_bits = (raw.to(tl.uint16) + (raw & 128).to(tl.uint16) << 7).to(
        tl.uint16
    )
    decoded = half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    decoded = tl.where(tl.abs(decoded) == 480.0, float("nan"), decoded)
    if ALIGNED:
        scale = tl.load((base + 512).to(tl.pointer_type(tl.float32)) + groups)
    else:
        sc = 512 + groups * 4
        b0 = tl.load(base + sc * S1).to(tl.uint32)
        b1 = tl.load(base + (sc + 1) * S1).to(tl.uint32)
        b2 = tl.load(base + (sc + 2) * S1).to(tl.uint32)
        b3 = tl.load(base + (sc + 3) * S1).to(tl.uint32)
        scale = (b0 | b1 << 8 | b2 << 16 | b3 << 24).to(
            tl.float32, bitcast=True
        )
    tl.store(
        tl.multiple_of(
            Dest + (row.to(tl.int64) if N * 576 >= 2147483648 else row) * 576,
            [16],
        )
        + cols,
        decoded * scale[:, None],
    )
    rc = tl.arange(0, 64)
    if ALIGNED:
        rope = tl.load((base + 528).to(tl.pointer_type(tl.uint16)) + rc)
    else:
        lo = tl.load(base + (528 + 2 * rc) * S1).to(tl.uint32)
        hi = tl.load(base + (529 + 2 * rc) * S1).to(tl.uint32)
        rope = (lo | hi << 8).to(tl.uint16)
    tl.store(
        tl.multiple_of(
            Dest + (row.to(tl.int64) if N * 576 >= 2147483648 else row) * 576,
            [16],
        )
        + 512
        + rc,
        rope.to(tl.bfloat16, bitcast=True),
    )


@triton.jit(do_not_specialize=["Q", "Dest"])
def _eager(
    Q,
    P,
    Dest,
    META: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    M: tl.constexpr = META[0]
    S0: tl.constexpr = META[1]
    S1: tl.constexpr = META[2]
    SP: tl.constexpr = META[3]
    ALIGNED: tl.constexpr = META[4]
    row = tl.program_id(0)
    src = tl.load(P + row * SP)
    if M * S0 < 2147483648:
        src = src.to(tl.int32)
    else:
        src = src.to(tl.int64)
    if M & M - 1 == 0:
        src = src & M - 1
    else:
        src = tl.where(src < 0, src + M, src)
    base = Q + src * S0
    groups = tl.arange(0, 4)
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    raw = tl.load(base + cols * S1).to(tl.uint32)
    if ALIGNED:
        scale = tl.load((base + 512).to(tl.pointer_type(tl.float32)) + groups)
    else:
        sc = 512 + groups * 4
        b0 = tl.load(base + sc * S1).to(tl.uint32)
        b1 = tl.load(base + (sc + 1) * S1).to(tl.uint32)
        b2 = tl.load(base + (sc + 2) * S1).to(tl.uint32)
        b3 = tl.load(base + (sc + 3) * S1).to(tl.uint32)
        scale = (b0 | b1 << 8 | b2 << 16 | b3 << 24).to(
            tl.float32, bitcast=True
        )
    rc = tl.arange(0, 64)
    if ALIGNED:
        rope = tl.load((base + 528).to(tl.pointer_type(tl.uint16)) + rc)
    else:
        lo = tl.load(base + (528 + 2 * rc) * S1).to(tl.uint32)
        hi = tl.load(base + (529 + 2 * rc) * S1).to(tl.uint32)
        rope = (lo | hi << 8).to(tl.uint16)
    half_bits = (raw.to(tl.uint16) + (raw & 128).to(tl.uint16) << 7).to(
        tl.uint16
    )
    decoded = half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    decoded = tl.where(tl.abs(decoded) == 480.0, float("nan"), decoded)
    tl.store(
        tl.multiple_of(Dest + row * 576, [16]) + cols, decoded * scale[:, None]
    )
    tl.store(
        tl.multiple_of(Dest + row * 576, [16]) + 512 + rc,
        rope.to(tl.bfloat16, bitcast=True),
    )


@triton.jit(do_not_specialize=["Q", "Dest"])
def _split(
    Q,
    P,
    Dest,
    META: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    M: tl.constexpr = META[0]
    S0: tl.constexpr = META[1]
    S1: tl.constexpr = META[2]
    SP: tl.constexpr = META[3]
    ALIGNED: tl.constexpr = META[4]
    row = tl.program_id(0)
    part = tl.program_id(1)
    src = tl.load(P + row * SP)
    if M * S0 < 2147483648:
        src = src.to(tl.int32)
    else:
        src = src.to(tl.int64)
    if M & M - 1 == 0:
        src = src & M - 1
    else:
        src = tl.where(src < 0, src + M, src)
    base = Q + src * S0
    groups = part * 2 + tl.arange(0, 2)
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    raw = tl.load(base + cols * S1).to(tl.uint32)
    if ALIGNED:
        scale = tl.load((base + 512).to(tl.pointer_type(tl.float32)) + groups)
    else:
        sc = 512 + groups * 4
        b0 = tl.load(base + sc * S1).to(tl.uint32)
        b1 = tl.load(base + (sc + 1) * S1).to(tl.uint32)
        b2 = tl.load(base + (sc + 2) * S1).to(tl.uint32)
        b3 = tl.load(base + (sc + 3) * S1).to(tl.uint32)
        scale = (b0 | b1 << 8 | b2 << 16 | b3 << 24).to(
            tl.float32, bitcast=True
        )
    rc = part * 32 + tl.arange(0, 32)
    if ALIGNED:
        rope = tl.load((base + 528).to(tl.pointer_type(tl.uint16)) + rc)
    else:
        lo = tl.load(base + (528 + 2 * rc) * S1).to(tl.uint32)
        hi = tl.load(base + (529 + 2 * rc) * S1).to(tl.uint32)
        rope = (lo | hi << 8).to(tl.uint16)
    half_bits = ((raw & 127) << 7 | (raw & 128) << 8).to(tl.uint16)
    decoded = (
        half_bits.to(tl.float16, bitcast=True) * tl.full((), 256.0, tl.float16)
    ).to(tl.float32)
    decoded = tl.where(tl.abs(decoded) == 480.0, float("nan"), decoded)
    tl.store(
        tl.multiple_of(Dest + row * 576, [16]) + cols, decoded * scale[:, None]
    )
    tl.store(
        tl.multiple_of(Dest + row * 576, [16]) + 512 + rc,
        rope.to(tl.bfloat16, bitcast=True),
    )


def dequantize_k_cache_paged(
    quant_k_cache, page_table_1_flattened, group_size=128
):
    quant_k_cache = quant_k_cache.view(torch.uint8)
    n = page_table_1_flattened.numel()
    out = quant_k_cache.new_empty((n, 1, 576), dtype=torch.bfloat16)
    if n:
        s0, _, s1 = quant_k_cache.stride()
        if n < 256:
            _split.run(
                quant_k_cache,
                page_table_1_flattened,
                out,
                (
                    quant_k_cache.shape[0],
                    s0,
                    s1,
                    page_table_1_flattened.stride(0),
                    s1 == 1
                    and s0 % 4 == 0
                    and (quant_k_cache.storage_offset() % 4 == 0),
                ),
                grid=(n, 2),
                num_warps=4,
                warmup=False,
            )
        elif n < 4096:
            _eager.run(
                quant_k_cache,
                page_table_1_flattened,
                out,
                (
                    quant_k_cache.shape[0],
                    s0,
                    s1,
                    page_table_1_flattened.stride(0),
                    s1 == 1
                    and s0 % 4 == 0
                    and (quant_k_cache.storage_offset() % 4 == 0),
                ),
                grid=(n,),
                num_warps=1,
                warmup=False,
            )
        elif (
            s1 == 1
            and s0 % 4 == 0
            and (quant_k_cache.storage_offset() % 4 == 0)
        ):
            _vector_rows.run(
                quant_k_cache,
                page_table_1_flattened,
                out,
                (
                    quant_k_cache.shape[0],
                    s0,
                    page_table_1_flattened.stride(0),
                    n,
                    2,
                ),
                grid=(triton.cdiv(n, 2),),
                num_warps=1,
                warmup=False,
            )
        else:
            _grouped.run(
                quant_k_cache,
                page_table_1_flattened,
                out,
                (
                    quant_k_cache.shape[0],
                    s0,
                    s1,
                    page_table_1_flattened.stride(0),
                    s1 == 1
                    and s0 % 4 == 0
                    and (quant_k_cache.storage_offset() % 4 == 0),
                    n,
                ),
                grid=(n,),
                num_warps=1,
                warmup=False,
            )
    return out


@triton.jit(do_not_specialize=["Q", "Dest"])
def _vector_rows(
    Q,
    P,
    Dest,
    META: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    M: tl.constexpr = META[0]
    S0: tl.constexpr = META[1]
    SP: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    ROWS: tl.constexpr = META[4]
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    live = (N % ROWS == 0) | (rows < N)
    src = tl.load(P + rows * SP, live, other=0)
    if M * S0 < 2147483648:
        src = src.to(tl.int32)
    else:
        src = src.to(tl.int64)
    if M & M - 1 == 0:
        src = src & M - 1
    else:
        src = tl.where(src < 0, src + M, src)
    base = Q.to(tl.pointer_type(tl.uint8)) + src * S0
    rc = tl.arange(0, 32)
    rope = tl.load(
        (base[:, None] + 528).to(tl.pointer_type(tl.uint32)) + rc[None, :],
        live[:, None],
        other=0,
    )
    groups = tl.arange(0, 4)
    words = tl.arange(0, 32)
    shifts = tl.arange(0, 4) * 8
    packed = tl.load(
        base[:, None, None].to(tl.pointer_type(tl.uint32))
        + groups[None, :, None] * 32
        + words[None, None, :],
        live[:, None, None],
        other=0,
    )
    raw = (packed[:, :, :, None] >> shifts[None, None, None, :] & 255).reshape(
        ROWS, 4, 128
    )
    magnitude = raw & 127
    normal = ((magnitude << 20) + 1006632960).to(tl.float32, bitcast=True)
    decoded = tl.where(magnitude < 8, tl.fma(normal, 2.0, -0.015625), normal)
    bits = decoded.to(tl.uint32, bitcast=True) | (raw & 128) << 24
    decoded = bits.to(tl.float32, bitcast=True)
    decoded = tl.where(tl.abs(decoded) == 480.0, float("nan"), decoded)
    scales = tl.load(
        (base[:, None] + 512).to(tl.pointer_type(tl.float32))
        + groups[None, :],
        live[:, None],
        other=0,
    )
    value = decoded * scales[:, :, None]
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(
        tl.multiple_of(
            Dest
            + (
                rows[:, None, None].to(tl.int64)
                if N * 576 >= 2147483648
                else rows[:, None, None]
            )
            * 576,
            [16, 16, 16],
        )
        + cols[None, :, :],
        value,
        live[:, None, None],
        cache_modifier=".cs",
    )
    tl.store(
        tl.multiple_of(
            Dest.to(tl.pointer_type(tl.uint32))
            + (
                rows[:, None].to(tl.int64)
                if N * 576 >= 2147483648
                else rows[:, None]
            )
            * 288,
            [16, 16],
        )
        + 256
        + rc[None, :],
        rope,
        live[:, None],
        cache_modifier=".cs",
    )


__all__ = ["dequantize_k_cache_paged"]
