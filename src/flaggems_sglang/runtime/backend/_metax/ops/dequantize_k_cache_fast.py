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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl

_DIM_NOPE = 512
_OUT_COLS = 576
_OUT_WORDS = tl.constexpr(288)
_HALF_WORDS = tl.constexpr(64)


@triton.jit
def _dq_row_body(u32_ptr, out32_ptr, base, obase, NT: tl.constexpr):
    k = tl.arange(0, _HALF_WORDS)
    sth = tl.arange(0, NT // 2)
    rw = tl.arange(0, 32)
    NTW: tl.constexpr = 128 // NT

    u = tl.load(u32_ptr + base + k)
    u2 = tl.load(u32_ptr + base + 64 + k)
    scwA = tl.load(u32_ptr + base + 128 + sth)
    scwA2 = tl.load(u32_ptr + base + 128 + NT // 2 + sth)
    rope = tl.load(u32_ptr + base + 128 + NT + rw)

    s32A = ((scwA + 0x7FFF + ((scwA >> 16) & 1)) >> 16) << 16
    sbA = (s32A >> 16) | s32A
    sa = sbA.reshape(NT // 2, 1).to(tl.float32, bitcast=True)

    a0 = u & tl.cast(0x00FF00FF, tl.uint32)
    b0 = (u >> 8) & tl.cast(0x00FF00FF, tl.uint32)
    a1 = (u >> 16) & tl.cast(0x00FF00FF, tl.uint32)
    b1 = (u >> 24) & tl.cast(0x00FF00FF, tl.uint32)
    q0 = ((a0 << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
        0x3C000000, tl.uint32
    ) | ((a0 & tl.cast(0x00800080, tl.uint32)) << 24)
    q1 = ((b0 << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
        0x3C000000, tl.uint32
    ) | ((b0 & tl.cast(0x00800080, tl.uint32)) << 24)
    q2 = ((a1 << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
        0x3C000000, tl.uint32
    ) | ((a1 & tl.cast(0x00800080, tl.uint32)) << 24)
    q3 = ((b1 << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
        0x3C000000, tl.uint32
    ) | ((b1 & tl.cast(0x00800080, tl.uint32)) << 24)
    f0 = (
        (q0.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa)
        .to(tl.uint32, bitcast=True)
        .reshape(_HALF_WORDS)
    )
    f2 = (
        (q2.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa)
        .to(tl.uint32, bitcast=True)
        .reshape(_HALF_WORDS)
    )
    f1 = (
        (q1.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa)
        .to(tl.uint32, bitcast=True)
        .reshape(_HALF_WORDS)
    )
    f3 = (
        (q3.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa)
        .to(tl.uint32, bitcast=True)
        .reshape(_HALF_WORDS)
    )
    w_lo = (f0 >> 16) | (f1 & tl.cast(0xFFFF0000, tl.uint32))
    w_hi = (f2 >> 16) | (f3 & tl.cast(0xFFFF0000, tl.uint32))
    tl.store(out32_ptr + obase + tl.arange(0, 128), tl.interleave(w_lo, w_hi))

    s32A2 = ((scwA2 + 0x7FFF + ((scwA2 >> 16) & 1)) >> 16) << 16
    sbA2 = (s32A2 >> 16) | s32A2
    sa2 = sbA2.reshape(NT // 2, 1).to(tl.float32, bitcast=True)
    a0h = u2 & tl.cast(0x00FF00FF, tl.uint32)
    b0h = (u2 >> 8) & tl.cast(0x00FF00FF, tl.uint32)
    a1h = (u2 >> 16) & tl.cast(0x00FF00FF, tl.uint32)
    b1h = (u2 >> 24) & tl.cast(0x00FF00FF, tl.uint32)
    q0h = ((a0h << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
        0x3C000000, tl.uint32
    ) | ((a0h & tl.cast(0x00800080, tl.uint32)) << 24)
    q1h = ((b0h << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
        0x3C000000, tl.uint32
    ) | ((b0h & tl.cast(0x00800080, tl.uint32)) << 24)
    q2h = ((a1h << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
        0x3C000000, tl.uint32
    ) | ((a1h & tl.cast(0x00800080, tl.uint32)) << 24)
    q3h = ((b1h << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
        0x3C000000, tl.uint32
    ) | ((b1h & tl.cast(0x00800080, tl.uint32)) << 24)
    f0h = (
        (q0h.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa2)
        .to(tl.uint32, bitcast=True)
        .reshape(_HALF_WORDS)
    )
    f2h = (
        (q2h.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa2)
        .to(tl.uint32, bitcast=True)
        .reshape(_HALF_WORDS)
    )
    f1h = (
        (q1h.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa2)
        .to(tl.uint32, bitcast=True)
        .reshape(_HALF_WORDS)
    )
    f3h = (
        (q3h.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa2)
        .to(tl.uint32, bitcast=True)
        .reshape(_HALF_WORDS)
    )
    w_loh = (f0h >> 16) | (f1h & tl.cast(0xFFFF0000, tl.uint32))
    w_hih = (f2h >> 16) | (f3h & tl.cast(0xFFFF0000, tl.uint32))
    tl.store(
        out32_ptr + obase + 128 + tl.arange(0, 128),
        tl.interleave(w_loh, w_hih),
    )
    tl.store(out32_ptr + obase + 256 + rw, rope)


@triton.jit
def _dequant_k_cache_fast_kernel_1row(
    q_ptr,
    out_ptr,
    in_row_words,
    NT: tl.constexpr,
):
    _dq_row_body(
        q_ptr.to(tl.pointer_type(tl.uint32)),
        out_ptr.to(tl.pointer_type(tl.uint32)),
        tl.program_id(0) * in_row_words,
        tl.program_id(0) * _OUT_WORDS,
        NT,
    )


@triton.jit
def _dequant_k_cache_fast_kernel_2row(
    q_ptr,
    out_ptr,
    in_row_words,
    NT: tl.constexpr,
):
    row0 = tl.program_id(0) * 2
    u32_ptr = q_ptr.to(tl.pointer_type(tl.uint32))
    out32_ptr = out_ptr.to(tl.pointer_type(tl.uint32))
    k = tl.arange(0, _HALF_WORDS)
    sth = tl.arange(0, NT // 2)
    rw = tl.arange(0, 32)
    NTW: tl.constexpr = 128 // NT

    base0 = row0 * in_row_words
    base1 = base0 + in_row_words
    uA = tl.load(u32_ptr + base0 + k)
    uA2 = tl.load(u32_ptr + base0 + 64 + k)
    scwAA = tl.load(u32_ptr + base0 + 128 + sth)
    scwAA2 = tl.load(u32_ptr + base0 + 128 + NT // 2 + sth)
    ropeA = tl.load(u32_ptr + base0 + 128 + NT + rw)
    uB = tl.load(u32_ptr + base1 + k)
    uB2 = tl.load(u32_ptr + base1 + 64 + k)
    scwAB = tl.load(u32_ptr + base1 + 128 + sth)
    scwAB2 = tl.load(u32_ptr + base1 + 128 + NT // 2 + sth)
    ropeB = tl.load(u32_ptr + base1 + 128 + NT + rw)

    for i in tl.static_range(2):
        if i == 0:
            u = uA
            u2 = uA2
            scwA = scwAA
            scwA2 = scwAA2
            rope = ropeA
        else:
            u = uB
            u2 = uB2
            scwA = scwAB
            scwA2 = scwAB2
            rope = ropeB
        obase = (row0 + i) * _OUT_WORDS

        s32A = ((scwA + 0x7FFF + ((scwA >> 16) & 1)) >> 16) << 16
        sbA = (s32A >> 16) | s32A
        sa = sbA.reshape(NT // 2, 1).to(tl.float32, bitcast=True)

        a0 = u & tl.cast(0x00FF00FF, tl.uint32)
        b0 = (u >> 8) & tl.cast(0x00FF00FF, tl.uint32)
        a1 = (u >> 16) & tl.cast(0x00FF00FF, tl.uint32)
        b1 = (u >> 24) & tl.cast(0x00FF00FF, tl.uint32)
        q0 = ((a0 << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
            0x3C000000, tl.uint32
        ) | ((a0 & tl.cast(0x00800080, tl.uint32)) << 24)
        q1 = ((b0 << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
            0x3C000000, tl.uint32
        ) | ((b0 & tl.cast(0x00800080, tl.uint32)) << 24)
        q2 = ((a1 << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
            0x3C000000, tl.uint32
        ) | ((a1 & tl.cast(0x00800080, tl.uint32)) << 24)
        q3 = ((b1 << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
            0x3C000000, tl.uint32
        ) | ((b1 & tl.cast(0x00800080, tl.uint32)) << 24)
        f0 = (
            (q0.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa)
            .to(tl.uint32, bitcast=True)
            .reshape(_HALF_WORDS)
        )
        f2 = (
            (q2.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa)
            .to(tl.uint32, bitcast=True)
            .reshape(_HALF_WORDS)
        )
        f1 = (
            (q1.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa)
            .to(tl.uint32, bitcast=True)
            .reshape(_HALF_WORDS)
        )
        f3 = (
            (q3.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa)
            .to(tl.uint32, bitcast=True)
            .reshape(_HALF_WORDS)
        )
        w_lo = (f0 >> 16) | (f1 & tl.cast(0xFFFF0000, tl.uint32))
        w_hi = (f2 >> 16) | (f3 & tl.cast(0xFFFF0000, tl.uint32))
        tl.store(
            out32_ptr + obase + tl.arange(0, 128), tl.interleave(w_lo, w_hi)
        )

        s32A2 = ((scwA2 + 0x7FFF + ((scwA2 >> 16) & 1)) >> 16) << 16
        sbA2 = (s32A2 >> 16) | s32A2
        sa2 = sbA2.reshape(NT // 2, 1).to(tl.float32, bitcast=True)
        a0h = u2 & tl.cast(0x00FF00FF, tl.uint32)
        b0h = (u2 >> 8) & tl.cast(0x00FF00FF, tl.uint32)
        a1h = (u2 >> 16) & tl.cast(0x00FF00FF, tl.uint32)
        b1h = (u2 >> 24) & tl.cast(0x00FF00FF, tl.uint32)
        q0h = ((a0h << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
            0x3C000000, tl.uint32
        ) | ((a0h & tl.cast(0x00800080, tl.uint32)) << 24)
        q1h = ((b0h << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
            0x3C000000, tl.uint32
        ) | ((b0h & tl.cast(0x00800080, tl.uint32)) << 24)
        q2h = ((a1h << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
            0x3C000000, tl.uint32
        ) | ((a1h & tl.cast(0x00800080, tl.uint32)) << 24)
        q3h = ((b1h << 20) & tl.cast(0x07F00000, tl.uint32)) + tl.cast(
            0x3C000000, tl.uint32
        ) | ((b1h & tl.cast(0x00800080, tl.uint32)) << 24)
        f0h = (
            (q0h.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa2)
            .to(tl.uint32, bitcast=True)
            .reshape(_HALF_WORDS)
        )
        f2h = (
            (q2h.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa2)
            .to(tl.uint32, bitcast=True)
            .reshape(_HALF_WORDS)
        )
        f1h = (
            (q1h.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa2)
            .to(tl.uint32, bitcast=True)
            .reshape(_HALF_WORDS)
        )
        f3h = (
            (q3h.reshape(NT // 2, NTW).to(tl.float32, bitcast=True) * sa2)
            .to(tl.uint32, bitcast=True)
            .reshape(_HALF_WORDS)
        )
        w_loh = (f0h >> 16) | (f1h & tl.cast(0xFFFF0000, tl.uint32))
        w_hih = (f2h >> 16) | (f3h & tl.cast(0xFFFF0000, tl.uint32))
        tl.store(
            out32_ptr + obase + 128 + tl.arange(0, 128),
            tl.interleave(w_loh, w_hih),
        )
        tl.store(out32_ptr + obase + 256 + rw, rope)


def dequantize_k_cache_fast(quant_k_cache, group_size=128):
    num_tokens = quant_k_cache.shape[0]
    out = torch.empty(
        (num_tokens, _OUT_COLS),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    if num_tokens == 0:
        return out
    in_row_words = quant_k_cache.stride(0) // 4
    NT = _DIM_NOPE // group_size
    if num_tokens % 2 == 0:
        _dequant_k_cache_fast_kernel_2row[(num_tokens // 2,)](
            quant_k_cache,
            out,
            in_row_words,
            NT=NT,
            num_warps=1,
        )
    else:
        _dequant_k_cache_fast_kernel_1row[(num_tokens,)](
            quant_k_cache,
            out,
            in_row_words,
            NT=NT,
            num_warps=1,
        )
    return out


__all__ = ["dequantize_k_cache_fast"]
