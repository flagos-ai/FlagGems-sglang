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

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)

_SPLIT_TOKENS = 512
_CHANNEL_SPLIT_TOKENS = 2048


@triton.jit
def _quantize_body(
    nope_out_ptr,
    rope_out_ptr,
    nope_ptr,
    rope_ptr,
    num_tokens,
    n_s0,
    n_s1,
    r_s0,
    r_s1,
    TOK: tl.constexpr,
    MASKED: tl.constexpr,
    CG: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    G: tl.constexpr,
    NOPE: tl.constexpr,
    RPE: tl.constexpr,
    CONTIG: tl.constexpr,
):
    NRB: tl.constexpr = DN + (DN // G) * 4
    RRB: tl.constexpr = DR * 2
    pid = tl.program_id(0)
    cgrp = tl.program_id(1)
    rows = pid * TOK + tl.arange(0, TOK)
    if MASKED:
        rmask = rows < num_tokens
    else:
        rmask = tl.full((TOK,), 1, tl.int1)

    if NOPE:
        NTI: tl.constexpr = CG // G
        ti = tl.arange(0, NTI)
        ci = tl.arange(0, G)
        cbase = cgrp * CG + ti[None, :, None] * G + ci[None, None, :]
        if CONTIG:
            x = tl.load(
                nope_ptr + rows[:, None, None] * DN + cbase,
                mask=rmask[:, None, None],
                other=0.0,
            ).to(tl.float32)
        else:
            x = tl.load(
                nope_ptr + rows[:, None, None] * n_s0 + cbase * n_s1,
                mask=rmask[:, None, None],
                other=0.0,
            ).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=2)
        scale = amax / 448.0
        q = x * (448.0 / amax)[:, :, None]

        u = q.to(tl.int32, bitcast=True)
        mag = u & 0x7FFFFFFF
        r = mag + 0x7FFFF + ((mag >> 20) & 1)
        v = (r >> 20) - 960
        b = (v + ((u >> 24) & 0x80)).to(tl.uint8)

        tl.store(
            nope_out_ptr.to(tl.pointer_type(tl.uint8))
            + rows[:, None, None] * NRB
            + cbase,
            b,
            mask=rmask[:, None, None],
        )

        if (NRB % 4 == 0) and (DN % 4 == 0):
            fptr = nope_out_ptr.to(tl.pointer_type(tl.float32))
            tl.store(
                fptr
                + rows[:, None] * (NRB // 4)
                + DN // 4
                + (cgrp * CG) // G
                + ti[None, :],
                scale,
                mask=rmask[:, None],
            )
        else:
            sv = scale.to(tl.int32, bitcast=True)
            sbase = (
                nope_out_ptr.to(tl.pointer_type(tl.uint8))
                + rows[:, None] * NRB
                + DN
                + ((cgrp * CG) // G + ti[None, :]) * 4
            )
            for kk in tl.static_range(4):
                tl.store(
                    sbase + kk,
                    ((sv >> (8 * kk)) & 0xFF).to(tl.uint8),
                    mask=rmask[:, None],
                )

    if RPE and (cgrp == 0):
        RP: tl.constexpr = triton.next_power_of_2(DR)
        rcofs = tl.arange(0, RP)
        rcm = rcofs < DR
        rmask2 = rmask[:, None] & rcm[None, :]
        if CONTIG:
            rv = tl.load(
                rope_ptr + rows[:, None] * DR + rcofs[None, :],
                mask=rmask2,
                other=0.0,
            )
        else:
            rv = tl.load(
                rope_ptr + rows[:, None] * r_s0 + rcofs[None, :] * r_s1,
                mask=rmask2,
                other=0.0,
            )
        if RRB % 2 == 0:
            bptr = rope_out_ptr.to(tl.pointer_type(tl.bfloat16))
            tl.store(
                bptr + rows[:, None] * (RRB // 2) + rcofs[None, :],
                rv,
                mask=rmask2,
            )
        else:
            ri = rv.to(tl.int16, bitcast=True)
            rbase = (
                rope_out_ptr.to(tl.pointer_type(tl.uint8))
                + rows[:, None] * RRB
                + 2 * rcofs[None, :]
            )
            tl.store(rbase, (ri & 0xFF).to(tl.uint8), mask=rmask2)
            tl.store(rbase + 1, ((ri >> 8) & 0xFF).to(tl.uint8), mask=rmask2)


@triton.jit
def _quantize_contig(
    nope_out_ptr,
    rope_out_ptr,
    nope_ptr,
    rope_ptr,
    num_tokens,
    TOK: tl.constexpr,
    MASKED: tl.constexpr,
    CG: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    G: tl.constexpr,
    NOPE: tl.constexpr,
    RPE: tl.constexpr,
):
    _quantize_body(
        nope_out_ptr,
        rope_out_ptr,
        nope_ptr,
        rope_ptr,
        num_tokens,
        0,
        0,
        0,
        0,
        TOK,
        MASKED,
        CG,
        DN,
        DR,
        G,
        NOPE,
        RPE,
        True,
    )


@triton.jit
def _quantize_strided(
    nope_out_ptr,
    rope_out_ptr,
    nope_ptr,
    rope_ptr,
    num_tokens,
    n_s0,
    n_s1,
    r_s0,
    r_s1,
    TOK: tl.constexpr,
    MASKED: tl.constexpr,
    CG: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    G: tl.constexpr,
    NOPE: tl.constexpr,
    RPE: tl.constexpr,
):
    _quantize_body(
        nope_out_ptr,
        rope_out_ptr,
        nope_ptr,
        rope_ptr,
        num_tokens,
        n_s0,
        n_s1,
        r_s0,
        r_s1,
        TOK,
        MASKED,
        CG,
        DN,
        DR,
        G,
        NOPE,
        RPE,
        False,
    )


def quantize_k_cache_fast_separate(k_nope, k_rope, group_size=128):
    num_tokens, dim_nope = k_nope.shape
    dim_rope = k_rope.shape[1]
    num_tiles = dim_nope // group_size

    nope_row_bytes = dim_nope + num_tiles * 4
    rope_row_bytes = dim_rope * k_rope.element_size()

    nope_part = torch.empty(
        (num_tokens, 1, nope_row_bytes),
        dtype=torch.uint8,
        device=k_nope.device,
    )
    rope_part = torch.empty(
        (num_tokens, 1, rope_row_bytes),
        dtype=torch.uint8,
        device=k_rope.device,
    )
    if num_tokens == 0:
        return nope_part, rope_part

    if (
        num_tokens <= _CHANNEL_SPLIT_TOKENS
        and dim_nope > 0
        and dim_nope % (2 * group_size) == 0
    ):
        ncg = 2
    else:
        ncg = 1
    cg = dim_nope // ncg if dim_nope > 0 else 0

    if num_tokens <= _SPLIT_TOKENS:
        tok = 1
    else:
        tok = 2

    masked = num_tokens % tok != 0
    cexprs = dict(
        TOK=tok,
        MASKED=masked,
        CG=cg,
        DN=dim_nope,
        DR=dim_rope,
        G=group_size,
        NOPE=(dim_nope > 0),
        RPE=(dim_rope > 0),
    )
    grid = ((num_tokens + tok - 1) // tok, ncg)
    if (
        k_nope.stride(1) == 1
        and k_nope.stride(0) == dim_nope
        and k_rope.stride(1) == 1
        and k_rope.stride(0) == dim_rope
    ):
        _quantize_contig[grid](
            nope_part,
            rope_part,
            k_nope,
            k_rope,
            num_tokens,
            num_warps=1,
            **cexprs,
        )
    else:
        _quantize_strided[grid](
            nope_part,
            rope_part,
            k_nope,
            k_rope,
            num_tokens,
            k_nope.stride(0),
            k_nope.stride(1),
            k_rope.stride(0),
            k_rope.stride(1),
            num_warps=1,
            **cexprs,
        )
    return nope_part, rope_part


__all__ = ["quantize_k_cache_fast_separate"]
