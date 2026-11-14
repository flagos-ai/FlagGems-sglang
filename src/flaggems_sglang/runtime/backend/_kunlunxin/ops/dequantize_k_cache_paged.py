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
def _kernel(
    Q,
    P,
    O,
    N: tl.constexpr,
    M: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    SP: tl.constexpr,
    BR: tl.constexpr,
    ALIGNED: tl.constexpr,
    DIRECT: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    if DIRECT:
        src = rows.to(tl.int64)
    else:
        src = tl.load(P + rows * SP, (N % BR == 0) | (rows < N), other=0).to(
            tl.int64
        )
        src = tl.where(src < 0, src + M, src)
    col = tl.arange(0, 512)
    base = Q + src[:, None] * S0
    raw = tl.load(
        base + col[None, :] * S1,
        (N % BR == 0) | (rows[:, None] < N),
        other=0,
        offset_state_policy="continuous" if ALIGNED else "",
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
        scale = tl.load(
            (base + 512).to(tl.pointer_type(tl.float32)) + col[None, :] // 128,
            (N % BR == 0) | (rows[:, None] < N),
            other=0,
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
        O
        + (
            rows[:, None].to(tl.int64)
            if N * 576 >= 2147483648
            else rows[:, None]
        )
        * (512 if DIRECT else 576)
        + col[None, :],
        product,
        (N % BR == 0) | (rows[:, None] < N),
        offset_state_policy="continuous",
    )
    if not DIRECT:
        rc = tl.arange(0, 64)
        if ALIGNED:
            packed = tl.load(
                (base + 528).to(tl.pointer_type(tl.uint16)) + rc[None, :],
                (N % BR == 0) | (rows[:, None] < N),
                other=0,
                offset_state_policy="continuous" if ALIGNED else "",
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
            O.to(tl.pointer_type(tl.uint16))
            + (
                rows[:, None].to(tl.int64)
                if N * 576 >= 2147483648
                else rows[:, None]
            )
            * (512 if DIRECT else 576)
            + 512
            + rc[None, :],
            packed,
            (N % BR == 0) | (rows[:, None] < N),
            offset_state_policy="continuous",
        )


@triton.jit
def _row(
    Q,
    P,
    O,
    N: tl.constexpr,
    M: tl.constexpr,
    QS: tl.constexpr,
    PS: tl.constexpr,
):
    row = tl.program_id(0)
    src = tl.load(P + row * PS).to(tl.int64)
    src = tl.where(src < 0, src + M, src)
    base = Q + src * QS
    col = tl.arange(0, 512)
    raw = tl.load(base + col).to(tl.uint32)
    a = raw & 127
    exponent = a >> 3
    mantissa = a & 7
    normal = (exponent + 120 << 23 | mantissa << 20).to(
        tl.float32, bitcast=True
    )
    value = tl.where(
        exponent == 0, mantissa.to(tl.float32) * 0.001953125, normal
    )
    bits = value.to(tl.uint32, bitcast=True) | (raw & 128) << 24
    value = tl.where(a == 127, 2143289344, bits).to(tl.float32, bitcast=True)
    scales = (base + 512).to(tl.pointer_type(tl.float32))
    s0 = tl.load(scales)
    s1 = tl.load(scales + 1)
    s2 = tl.load(scales + 2)
    s3 = tl.load(scales + 3)
    scale = tl.where(
        col < 256, tl.where(col < 128, s0, s1), tl.where(col < 384, s2, s3)
    )
    product = value * scale
    bits = product.to(tl.uint32, bitcast=True)
    rounded = bits + 32767 + (bits >> 16 & 1) >> 16
    rounded = tl.where(bits & 2139095040 == 2139095040, bits >> 16, rounded)
    dest = (
        O.to(tl.pointer_type(tl.uint16))
        + (row.to(tl.int64) if N * 576 >= 2147483648 else row) * 576
    )
    tl.store(dest + col, rounded.to(tl.uint16))
    rc = tl.arange(0, 64)
    rope = tl.load((base + 528).to(tl.pointer_type(tl.uint16)) + rc)
    tl.store(dest + 512 + rc, rope)


@triton.jit
def _gather(
    T,
    Q,
    P,
    Dest,
    N: tl.constexpr,
    M: tl.constexpr,
    QS: tl.constexpr,
    S1: tl.constexpr,
    PS: tl.constexpr,
    BR: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (N % BR == 0) | (rows < N)
    src = tl.load(P + rows * PS, live, other=0).to(tl.int64)
    src = tl.where(src < 0, src + M, src)
    T = T.to(tl.pointer_type(tl.uint16))
    Q = Q.to(tl.pointer_type(tl.uint8))
    Dest = Dest.to(tl.pointer_type(tl.uint16))
    col = tl.arange(0, 512)
    value = tl.load(
        T + src[:, None] * 512 + col[None, :],
        live[:, None],
        other=0,
        offset_state_policy="continuous",
    )
    tl.store(
        Dest
        + (
            rows[:, None].to(tl.int64)
            if N * 576 >= 2147483648
            else rows[:, None]
        )
        * 576
        + col[None, :],
        value,
        live[:, None],
        offset_state_policy="continuous",
    )
    base = Q + src[:, None] * QS
    rc = tl.arange(0, 64)
    if ALIGNED:
        rope = tl.load(
            (base + 528).to(tl.pointer_type(tl.uint16)) + rc[None, :],
            live[:, None],
            other=0,
            offset_state_policy="continuous",
        )
    else:
        lo = tl.load(
            base + (528 + rc[None, :] * 2) * S1, live[:, None], other=0
        ).to(tl.uint32)
        hi = tl.load(
            base + (529 + rc[None, :] * 2) * S1, live[:, None], other=0
        ).to(tl.uint32)
        rope = (lo | hi << 8).to(tl.uint16)
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
        rope,
        live[:, None],
        offset_state_policy="continuous",
    )


@triton.jit
def _small(
    Q,
    P,
    O,
    M: tl.constexpr,
    QS: tl.constexpr,
    PS: tl.constexpr,
    HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // 3
    part = pid % 3
    src = tl.load(P + row * PS).to(tl.int64)
    src = tl.where(src < 0, src + M, src)
    base = Q + src * QS
    if part < 2:
        col = part * 256 + tl.arange(0, 256)
        q = tl.load(base + col).to(tl.uint32)
        a = q & 127
        if HALF:
            bits = (a << 7 | (q & 128) << 8).to(tl.uint16)
            value = bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
            value = tl.where(a == 127, float("nan"), value)
        else:
            e = a >> 3
            m = a & 7
            normal = ((a << 20) + 1006632960).to(tl.float32, bitcast=True)
            value = tl.where(e == 0, m.to(tl.float32) * 0.001953125, normal)
            bits = value.to(tl.uint32, bitcast=True) | (q & 128) << 24
            value = tl.where(a == 127, 2143289344, bits).to(
                tl.float32, bitcast=True
            )
        scale = tl.load(
            (base + 512).to(tl.pointer_type(tl.float32)) + col // 128
        )
        tl.store(O + row * 576 + col, value * scale)
    else:
        rc = tl.arange(0, 64)
        rope = tl.load((base + 528).to(tl.pointer_type(tl.uint16)) + rc)
        tl.store(O.to(tl.pointer_type(tl.uint16)) + row * 576 + 512 + rc, rope)


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
        aligned = (
            strides[-1] == 1
            and strides[0] % 4 == 0
            and (cache.storage_offset() % 4 == 0)
        )
        q = cache.view(torch.uint8)
        if n < 16 and aligned:
            _small.run(
                q,
                page_table_1_flattened,
                out,
                cache.shape[0],
                strides[0],
                page_table_1_flattened.stride(0),
                False,
                grid=(n * 3,),
                num_warps=4,
                warmup=False,
            )
        elif n % 16 != 0 and aligned:
            _row.run(
                q,
                page_table_1_flattened,
                out,
                n,
                cache.shape[0],
                strides[0],
                page_table_1_flattened.stride(0),
                grid=(n,),
                num_warps=4,
                warmup=False,
            )
        else:
            m = cache.shape[0]
            direct = n >= 2 * m and n >= 1024
            count = m if direct else n
            target = (
                cache.new_empty((m, 1, 512), dtype=torch.bfloat16)
                if direct
                else out
            )
            br = (
                64
                if count >= 1024 and count % 64 == 0
                else 16 if count % 16 == 0 else 1
            )
            _kernel.run(
                q,
                page_table_1_flattened,
                target,
                count,
                cache.shape[0],
                strides[0],
                strides[-1],
                page_table_1_flattened.stride(0),
                br,
                aligned,
                direct,
                groups_per_cluster=br,
                isCloseUnrollControl=False,
                num_warps=4,
                grid=(triton.cdiv(count, br),),
                warmup=False,
                unroll_num=16 if direct and count >= 1024 else 8,
                isCloseMemoryAsync=False,
            )
            if direct:
                gb = (
                    64
                    if n >= 1024 and n % 64 == 0
                    else 16 if n % 16 == 0 else 1
                )
                _gather.run(
                    target,
                    q,
                    page_table_1_flattened,
                    out,
                    n,
                    m,
                    cache.stride(0),
                    cache.stride(-1),
                    page_table_1_flattened.stride(0),
                    gb,
                    cache.stride(-1) == 1
                    and cache.stride(0) % 4 == 0
                    and (cache.storage_offset() % 4 == 0),
                    grid=(n // gb,),
                    groups_per_cluster=gb,
                    isCloseUnrollControl=False,
                    num_warps=4,
                    warmup=False,
                    unroll_num=8,
                )
    return out


__all__ = ["dequantize_k_cache_paged"]
