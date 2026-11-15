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

_INV = 1.0 / 448.0


@triton.jit
def _encode_S(quot, value):
    af = tl.minimum(tl.abs(quot), 448.0)
    ab = af.to(tl.uint32, bitcast=True)
    normal = ((ab + 0x7FFFF + ((ab >> 20) & 1)) >> 20) - 960
    sub = (af + 16384.0).to(tl.uint32, bitcast=True) & 15
    code = tl.where(af >= 0.015625, normal, sub)
    code = tl.where(quot != quot, 127, code)
    return (code | ((value.to(tl.uint32, bitcast=True) >> 24) & 128)).to(
        tl.uint8
    )


@triton.jit
def _quot_S(value, scale):
    bump = tl.where(
        scale.to(tl.uint32, bitcast=True) < 0x0C000000,
        18446744073709551616.0,
        1.0,
    )
    s = scale * bump
    v = value * bump
    r = tl.div_rn(1.0, s)
    u = v * r
    return tl.math.fma(tl.math.fma(-u, s, v), r, u)


@triton.jit
def _tight_S(
    X,
    P,
    Q,
    O,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    RDIM: tl.constexpr,
    QROW: tl.constexpr,
    DIM: tl.constexpr,
    INV: tl.constexpr,
):
    g = tl.arange(0, GROUP)
    gc = tl.arange(0, GROUPS)
    rc = tl.arange(0, RDIM)
    b = tl.arange(0, 4)
    h = tl.arange(0, 2)
    row = tl.program_id(0)
    value = tl.load(X + row * DIM + gc[:, None] * GROUP + g[None, :]).to(
        tl.float32
    )
    scale = tl.max(tl.abs(value), 1) * INV
    tl.store(
        Q + row * QROW + gc[:, None] * GROUP + g[None, :],
        _encode_S(_quot_S(value, scale[:, None]), value),
    )
    bits = scale.to(tl.uint32, bitcast=True)
    tl.store(
        Q + row * QROW + DIM + gc[:, None] * 4 + b[None, :],
        ((bits[:, None] >> (b * 8)[None, :]) & 255).to(tl.uint8),
    )
    word = (
        tl.load(P + row * RDIM + rc).to(tl.uint16, bitcast=True).to(tl.uint32)
    )
    tl.store(
        O + row * (RDIM * 2) + rc[:, None] * 2 + h[None, :],
        ((word[:, None] >> (h * 8)[None, :]) & 255).to(tl.uint8),
    )


@triton.jit
def _pair_S(
    X,
    P,
    Q,
    O,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    RDIM: tl.constexpr,
    QROW: tl.constexpr,
    DIM: tl.constexpr,
    INV: tl.constexpr,
    BR: tl.constexpr,
):
    TG: tl.constexpr = BR * GROUPS
    g = tl.arange(0, GROUP)
    gi = tl.arange(0, TG)
    wi = tl.arange(0, BR * RDIM)
    b = tl.arange(0, 4)
    h = tl.arange(0, 2)
    base = tl.program_id(0) * BR
    row = base + gi // GROUPS
    grp = gi % GROUPS
    value = tl.load(
        X + row[:, None] * DIM + grp[:, None] * GROUP + g[None, :],
        cache_modifier=".cg",
    ).to(tl.float32)
    word = (
        tl.load(P + base * RDIM + wi, cache_modifier=".cg")
        .to(tl.uint16, bitcast=True)
        .to(tl.uint32)
    )
    scale = tl.max(tl.abs(value), 1) * INV
    tl.store(
        Q + row[:, None] * QROW + grp[:, None] * GROUP + g[None, :],
        _encode_S(_quot_S(value, scale[:, None]), value),
        cache_modifier=".cs",
    )
    bits = scale.to(tl.uint32, bitcast=True)
    tl.store(
        Q + row[:, None] * QROW + DIM + grp[:, None] * 4 + b[None, :],
        ((bits[:, None] >> (b * 8)[None, :]) & 255).to(tl.uint8),
        cache_modifier=".cs",
    )
    tl.store(
        O + base * RDIM * 2 + wi[:, None] * 2 + h[None, :],
        ((word[:, None] >> (h * 8)[None, :]) & 255).to(tl.uint8),
        cache_modifier=".cs",
    )


@triton.jit
def _strided_S(
    X,
    P,
    Q,
    O,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    RDIM: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    QROW: tl.constexpr,
    DIM: tl.constexpr,
    INV: tl.constexpr,
):
    g = tl.arange(0, GROUP)
    gc = tl.arange(0, GROUPS)
    rc = tl.arange(0, RDIM)
    b = tl.arange(0, 4)
    h = tl.arange(0, 2)
    row = tl.program_id(0)
    value = tl.load(X + row * X0 + (gc[:, None] * GROUP + g[None, :]) * X1).to(
        tl.float32
    )
    scale = tl.max(tl.abs(value), 1) * INV
    tl.store(
        Q + row * QROW + gc[:, None] * GROUP + g[None, :],
        _encode_S(tl.div_rn(value, scale[:, None]), value),
    )
    bits = scale.to(tl.uint32, bitcast=True)
    tl.store(
        Q + row * QROW + DIM + gc[:, None] * 4 + b[None, :],
        ((bits[:, None] >> (b * 8)[None, :]) & 255).to(tl.uint8),
    )
    word = (
        tl.load(P + row * P0 + rc * P1)
        .to(tl.uint16, bitcast=True)
        .to(tl.uint32)
    )
    tl.store(
        O + row * (RDIM * 2) + rc[:, None] * 2 + h[None, :],
        ((word[:, None] >> (h * 8)[None, :]) & 255).to(tl.uint8),
    )


def quantize_k_cache_fast_separate_small(k_nope, k_rope, group_size=128):
    tokens, dim_nope = k_nope.shape
    dim_rope = k_rope.shape[1]
    groups = dim_nope // group_size
    width = dim_nope + groups * 4
    nope_part = torch.empty(
        (tokens, 1, width), dtype=torch.uint8, device=k_nope.device
    )
    rope_part = torch.empty(
        (tokens, 1, dim_rope * 2), dtype=torch.uint8, device=k_rope.device
    )
    if tokens:
        if not (k_nope.is_contiguous() and k_rope.is_contiguous()):
            _strided_S.run(
                k_nope,
                k_rope,
                nope_part,
                rope_part,
                group_size,
                groups,
                dim_rope,
                *k_nope.stride(),
                *k_rope.stride(),
                width,
                dim_nope,
                _INV,
                grid=(tokens,),
                warmup=False,
                num_warps=2,
                enable_fp_fusion=False,
            )
        elif tokens >= 2048 and tokens % 4 == 0:
            _pair_S.run(
                k_nope,
                k_rope,
                nope_part,
                rope_part,
                group_size,
                groups,
                dim_rope,
                width,
                dim_nope,
                _INV,
                4,
                grid=(tokens // 4,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _tight_S.run(
                k_nope,
                k_rope,
                nope_part,
                rope_part,
                group_size,
                groups,
                dim_rope,
                width,
                dim_nope,
                _INV,
                grid=(tokens,),
                warmup=False,
                num_warps=2,
                enable_fp_fusion=False,
            )
    return nope_part, rope_part


@triton.jit
def _encode_L(quot, value):
    af = tl.minimum(tl.abs(quot), 448.0)
    raw = af.to(tl.int32, bitcast=True)
    exponent = tl.maximum(raw >> 23, 121)
    bias = exponent * 8388616 + 167771192
    code = (af + bias.to(tl.float32, bitcast=True)).to(tl.int32, bitcast=True)
    code = tl.where(quot != quot, 127, code)
    sign = (value.to(tl.uint32, bitcast=True) >> 24) & 128
    return ((code & 127) | sign).to(tl.uint8)


@triton.jit
def _quot_L(value, scale):
    bump = tl.where(
        scale.to(tl.uint32, bitcast=True) < 0x0C000000,
        18446744073709551616.0,
        1.0,
    )
    s = scale * bump
    v = value * bump
    r = tl.div_rn(1.0, s)
    u = v * r
    return tl.math.fma(tl.math.fma(-u, s, v), r, u)


@triton.jit
def _strided_L(
    X,
    P,
    Q,
    O,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    RDIM: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    QROW: tl.constexpr,
    DIM: tl.constexpr,
    INV: tl.constexpr,
):
    g = tl.arange(0, GROUP)
    gc = tl.arange(0, GROUPS)
    rc = tl.arange(0, RDIM)
    b = tl.arange(0, 4)
    h = tl.arange(0, 2)
    row = tl.program_id(0)
    value = tl.load(X + row * X0 + (gc[:, None] * GROUP + g[None, :]) * X1).to(
        tl.float32
    )
    scale = tl.max(tl.abs(value), 1) * INV
    tl.store(
        Q + row * QROW + gc[:, None] * GROUP + g[None, :],
        _encode_L(tl.div_rn(value, scale[:, None]), value),
    )
    bits = scale.to(tl.uint32, bitcast=True)
    tl.store(
        Q + row * QROW + DIM + gc[:, None] * 4 + b[None, :],
        ((bits[:, None] >> (b * 8)[None, :]) & 255).to(tl.uint8),
    )
    word = (
        tl.load(P + row * P0 + rc * P1)
        .to(tl.uint16, bitcast=True)
        .to(tl.uint32)
    )
    tl.store(
        O + row * (RDIM * 2) + rc[:, None] * 2 + h[None, :],
        ((word[:, None] >> (h * 8)[None, :]) & 255).to(tl.uint8),
    )


@triton.jit
def _encode_group_validity_L(quot, value, valid):
    af = quot
    raw = af.to(tl.int32, bitcast=True)
    exponent = tl.maximum(raw >> 23, 121)
    bias = exponent * 8388616 + 167771192
    code = (af + bias.to(tl.float32, bitcast=True)).to(tl.int32, bitcast=True)
    code = tl.where(valid, code, 127)
    sign = (value.to(tl.uint32, bitcast=True) >> 24) & 128
    return ((code & 127) | sign).to(tl.uint8)


@triton.jit
def _quot_negscale_L(value, scale):
    bump = tl.where(
        scale.to(tl.uint32, bitcast=True) < 0x0C000000,
        18446744073709551616.0,
        1.0,
    )
    s = scale * bump
    v = value * bump
    r = tl.div_rn(1.0, s)
    u = v * r
    ns = -s
    return tl.math.fma(tl.math.fma(u, ns, v), r, u)


@triton.jit
def _encode_without_validity_L(quot, value):
    af = quot
    raw = af.to(tl.int32, bitcast=True)
    exponent = tl.maximum(raw >> 23, 121)
    bias = exponent * 8388616 + 167771192
    code = (af + bias.to(tl.float32, bitcast=True)).to(tl.int32, bitcast=True)
    sign = (value.to(tl.uint32, bitcast=True) >> 24) & 128
    return ((code & 127) | sign).to(tl.uint8)


@triton.jit
def _tight_group_bits_L(
    X,
    P,
    Q,
    O,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    RDIM: tl.constexpr,
    QROW: tl.constexpr,
    DIM: tl.constexpr,
    INV: tl.constexpr,
):
    g = tl.arange(0, GROUP)
    gc = tl.arange(0, GROUPS)
    rc = tl.arange(0, RDIM)
    b = tl.arange(0, 4)
    h = tl.arange(0, 2)
    row = tl.program_id(0)
    value = tl.load(X + row * DIM + gc[:, None] * GROUP + g[None, :]).to(
        tl.float32
    )
    magnitude = tl.abs(value)
    scale = tl.max(magnitude, 1) * INV
    tl.store(
        Q + row * QROW + gc[:, None] * GROUP + g[None, :],
        _encode_group_validity_L(
            _quot_L(magnitude, scale[:, None]),
            value,
            ((scale.to(tl.uint32, bitcast=True) - 1) < 0x7F7FFFFF)[:, None],
        ),
    )
    bits = scale.to(tl.uint32, bitcast=True)
    tl.store(
        Q + row * QROW + DIM + gc[:, None] * 4 + b[None, :],
        ((bits[:, None] >> (b * 8)[None, :]) & 255).to(tl.uint8),
    )
    word = (
        tl.load(P + row * RDIM + rc).to(tl.uint16, bitcast=True).to(tl.uint32)
    )
    tl.store(
        O + row * (RDIM * 2) + rc[:, None] * 2 + h[None, :],
        ((word[:, None] >> (h * 8)[None, :]) & 255).to(tl.uint8),
    )


@triton.jit
def _partition_memory_L(
    X,
    P,
    Q,
    O,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    RDIM: tl.constexpr,
    QROW: tl.constexpr,
    DIM: tl.constexpr,
    INV: tl.constexpr,
    BR: tl.constexpr,
    ROWS: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < ROWS // BR:
        for tile in tl.range(
            pid, tl.minimum(pid + 1, ROWS // BR), num_stages=1
        ):
            g = tl.arange(0, GROUP)
            gi = tl.arange(0, BR * GROUPS)
            b = tl.arange(0, 4)
            row = tile * BR + gi // GROUPS
            grp = gi % GROUPS
            value = tl.load(
                X + row[:, None] * DIM + grp[:, None] * GROUP + g[None, :],
                cache_modifier=".cg",
            ).to(tl.float32)
            magnitude = tl.abs(value)
            scale = tl.max(magnitude, 1) * INV
            valid = scale.to(tl.uint32, bitcast=True) - 1 < 2139095039
            code = tl.reshape(
                _encode_without_validity_L(
                    _quot_negscale_L(magnitude, scale[:, None]), value
                ),
                (BR * GROUPS, GROUP // 4, 4),
            )
            words = tl.sum(code.to(tl.uint32) << (b * 8)[None, None, :], 2)
            words = tl.where(
                valid[:, None], words, words & 2155905152 | 2139062143
            )
            c = tl.max_contiguous(tl.arange(0, GROUP // 4), 2)
            tl.store(
                Q.to(tl.pointer_type(tl.uint32))
                + row[:, None] * (QROW // 4)
                + grp[:, None] * (GROUP // 4)
                + c[None, :],
                words,
                cache_modifier=".cs",
            )
            tl.store(
                Q.to(tl.pointer_type(tl.float32))
                + row * (QROW // 4)
                + DIM // 4
                + grp,
                scale,
                cache_modifier=".cs",
            )
    else:
        i = (pid - ROWS // BR) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(
            P.to(tl.pointer_type(tl.uint32)) + i,
            i < ROWS * RDIM // 2,
            other=0,
            cache_modifier=".cg",
        )
        tl.store(
            O.to(tl.pointer_type(tl.uint32)) + i,
            v,
            i < ROWS * RDIM // 2,
            cache_modifier=".cs",
        )


@triton.jit
def _b2_L(
    X,
    P,
    Q,
    O,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    RDIM: tl.constexpr,
    QROW: tl.constexpr,
    DIM: tl.constexpr,
    INV: tl.constexpr,
    BR: tl.constexpr,
    ROWS: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < ROWS // BR:
        for tile in tl.range(
            pid, tl.minimum(pid + 1, ROWS // BR), num_stages=1
        ):
            g = tl.arange(0, GROUP)
            gi = tl.arange(0, BR * GROUPS)
            b = tl.arange(0, 4)
            row = tile * BR + gi // GROUPS
            grp = gi % GROUPS
            value = tl.load(
                X + row[:, None] * DIM + grp[:, None] * GROUP + g[None, :],
                cache_modifier=".cg",
            ).to(tl.float32)
            magnitude = tl.abs(value)
            scale = tl.max(magnitude, 1) * INV
            valid = scale.to(tl.uint32, bitcast=True) - 1 < 2139095039
            code = tl.reshape(
                _encode_without_validity_L(
                    _quot_negscale_L(magnitude, scale[:, None]), value
                ),
                (BR * GROUPS, GROUP // 4, 4),
            )
            words = tl.sum(code.to(tl.uint32) << (b * 8)[None, None, :], 2)
            words = tl.where(
                valid[:, None], words, words & 2155905152 | 2139062143
            )
            c = tl.max_contiguous(tl.arange(0, GROUP // 4), 2)
            tl.store(
                Q.to(tl.pointer_type(tl.uint32))
                + row[:, None] * (QROW // 4)
                + grp[:, None] * (GROUP // 4)
                + c[None, :],
                words,
                cache_modifier=".cs",
            )
            tl.store(
                Q.to(tl.pointer_type(tl.float32))
                + row * (QROW // 4)
                + DIM // 4
                + grp,
                scale,
                cache_modifier=".cs",
            )
    else:
        i = (pid - ROWS // BR) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(
            P.to(tl.pointer_type(tl.uint32)) + i,
            i < ROWS * RDIM // 2,
            other=0,
            cache_modifier=".cg",
        )
        tl.store(
            O.to(tl.pointer_type(tl.uint32)) + i,
            v,
            i < ROWS * RDIM // 2,
            cache_modifier=".cs",
        )


def quantize_k_cache_fast_separate_large(k_nope, k_rope, group_size=128):
    tokens, dim_nope = k_nope.shape
    dim_rope = k_rope.shape[1]
    groups = dim_nope // group_size
    width = dim_nope + groups * 4
    nope_part = torch.empty(
        (tokens, 1, width), dtype=torch.uint8, device=k_nope.device
    )
    rope_part = torch.empty(
        (tokens, 1, dim_rope * 2), dtype=torch.uint8, device=k_rope.device
    )
    if tokens:
        if not (k_nope.is_contiguous() and k_rope.is_contiguous()):
            _strided_L.run(
                k_nope,
                k_rope,
                nope_part,
                rope_part,
                group_size,
                groups,
                dim_rope,
                *k_nope.stride(),
                *k_rope.stride(),
                width,
                dim_nope,
                _INV,
                grid=(tokens,),
                warmup=False,
                num_warps=2,
                enable_fp_fusion=False,
            )
        elif tokens == 2048 and dim_rope % 2 == 0:
            _b2_L.run(
                k_nope,
                k_rope,
                nope_part,
                rope_part,
                group_size,
                groups,
                dim_rope,
                width,
                dim_nope,
                _INV,
                2,
                tokens,
                1,
                1024,
                grid=(
                    tokens // 2 + triton.cdiv(tokens * dim_rope // 2, 1024),
                ),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        elif tokens >= 2048 and tokens % 4 == 0 and dim_rope % 2 == 0:
            _partition_memory_L.run(
                k_nope,
                k_rope,
                nope_part,
                rope_part,
                group_size,
                groups,
                dim_rope,
                width,
                dim_nope,
                _INV,
                4,
                tokens,
                1,
                1024,
                grid=(
                    tokens // 4 + triton.cdiv(tokens * dim_rope // 2, 1024),
                ),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _tight_group_bits_L.run(
                k_nope,
                k_rope,
                nope_part,
                rope_part,
                group_size,
                groups,
                dim_rope,
                width,
                dim_nope,
                _INV,
                grid=(tokens,),
                warmup=False,
                num_warps=2,
                enable_fp_fusion=False,
            )
    return nope_part, rope_part


def quantize_k_cache_fast_separate(k_nope, k_rope, group_size=128):
    if k_nope.shape[0] <= 128:
        return quantize_k_cache_fast_separate_small(k_nope, k_rope, group_size)
    return quantize_k_cache_fast_separate_large(k_nope, k_rope, group_size)


__all__ = ["quantize_k_cache_fast_separate"]
