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


@triton.jit(do_not_specialize_on_alignment=["P"])
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
    src = tl.load(P + rows * SP, (N % BR == 0) | (rows < N), other=0)
    if M * S0 < 2147483648:
        src = src.to(tl.int32)
    else:
        src = src.to(tl.int64)
    src = tl.where(src < 0, src + M, src)
    groups = tl.arange(0, 4)
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    valid = (N % BR == 0) | (rows < N)
    base = Q + src[:, None] * S0
    if ALIGNED:
        word = tl.arange(0, 32)
        packed_q = tl.load(
            base[:, None, :].to(tl.pointer_type(tl.uint32))
            + groups[None, :, None] * 32
            + word[None, None, :],
            valid[:, None, None],
            other=0,
        )
    else:
        raw = tl.load(
            Q + src[:, None, None] * S0 + cols[None, :, :] * S1,
            valid[:, None, None],
            other=0,
        ).to(tl.uint32)
    if ALIGNED:
        scale = tl.load(
            (base + 512).to(tl.pointer_type(tl.float32)) + groups[None, :],
            valid[:, None],
            other=0,
        )
    else:
        sc = 512 + groups * 4
        b0 = tl.load(base + sc[None, :] * S1, valid[:, None], other=0).to(
            tl.uint32
        )
        b1 = tl.load(
            base + (sc[None, :] + 1) * S1, valid[:, None], other=0
        ).to(tl.uint32)
        b2 = tl.load(
            base + (sc[None, :] + 2) * S1, valid[:, None], other=0
        ).to(tl.uint32)
        b3 = tl.load(
            base + (sc[None, :] + 3) * S1, valid[:, None], other=0
        ).to(tl.uint32)
        scale = (b0 | b1 << 8 | b2 << 16 | b3 << 24).to(
            tl.float32, bitcast=True
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
    if N >= 4096:
        if ALIGNED:
            shifts = tl.arange(0, 4) * 8
            raw = (
                packed_q[:, :, :, None] >> shifts[None, None, None, :] & 255
            ).reshape(BR, 4, 128)
        byte = tl.arange(0, 256)
        half_bits = (byte + (byte & 128) + ((byte & 127) + 1 & 128) << 7).to(
            tl.uint16
        )
        table = half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
        decoded = tl.gather(
            table, raw.to(tl.int32).reshape(BR * 512), axis=0
        ).reshape(BR, 4, 128)
    else:
        if ALIGNED:
            nan_flags = (
                ~((packed_q & 2139062143 ^ 2139062143) + 2139062143)
                & 2155905152
            )
            pair0 = packed_q & 16711935
            pair1 = packed_q >> 8 & 16711935
            pair0 = pair0 + (pair0 & 8388736) << 7 | (nan_flags & 8388736) << 7
            pair1 = (
                pair1 + (pair1 & 8388736) << 7
                | (nan_flags >> 8 & 8388736) << 7
            )
            carrier = tl.join(pair0, pair1)
            hs = tl.arange(0, 2) * 16
            half_words = (
                carrier[:, :, :, :, None] >> hs[None, None, None, None, :]
                & 65535
            )
            half_bits = (
                tl.permute(half_words, (0, 1, 2, 4, 3))
                .reshape(BR, 4, 128)
                .to(tl.uint16)
            )
        else:
            half_bits = (
                raw.to(tl.uint8).to(tl.int8, bitcast=True).to(tl.int16) << 7
            ).to(tl.uint16) & 49151
        decoded = half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
        if not ALIGNED:
            decoded = tl.where(tl.abs(decoded) == 480.0, float("nan"), decoded)
    product = decoded * scale[:, :, None]
    tl.store(
        Dest
        + (
            rows[:, None, None].to(tl.int64)
            if N * 576 >= 2147483648
            else rows[:, None, None]
        )
        * 576
        + cols[None, :, :],
        product,
        valid[:, None, None],
    )
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


def dequantize_k_cache_paged(
    quant_k_cache, page_table_1_flattened, group_size=128
):
    cache = quant_k_cache
    n = page_table_1_flattened.numel()
    out = torch.empty(
        (n, 1, 576), dtype=torch.bfloat16, device=quant_k_cache.device
    )
    if n:
        br = 8 if n >= 4096 else 4
        _kernel.run(
            cache,
            page_table_1_flattened,
            out,
            (
                n,
                cache.shape[0],
                cache.stride(0),
                cache.stride(-1),
                page_table_1_flattened.stride(0),
                br,
                cache.stride(-1) == 1
                and cache.stride(0) % 4 == 0
                and (cache.storage_offset() % 4 == 0),
            ),
            num_warps=4 if n >= 4096 else 4,
            grid=(triton.cdiv(n, br),),
            warmup=False,
        )
    return out


__all__ = ["dequantize_k_cache_paged"]
