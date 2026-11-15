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
    raw = af.to(tl.int32, bitcast=True)
    exponent = tl.maximum(raw >> 23, 121)
    bias = exponent * 8388616 + 167771192
    code = (af + bias.to(tl.float32, bitcast=True)).to(tl.int32, bitcast=True)
    code = tl.where(quot != quot, 127, code)
    sign = (value.to(tl.uint32, bitcast=True) >> 24) & 128
    return ((code & 127) | sign).to(tl.uint8)


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


@triton.jit
def _partition_memory_S(
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
        g = tl.arange(0, GROUP)
        gi = tl.arange(0, BR * GROUPS)
        b = tl.arange(0, 4)
        row = pid * BR + gi // GROUPS
        grp = gi % GROUPS
        value = tl.load(
            X + row[:, None] * DIM + grp[:, None] * GROUP + g[None, :],
            cache_modifier=".cg",
        ).to(tl.float32)
        scale = tl.max(tl.abs(value), 1) * INV
        if MODE == 0:
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
        elif MODE == 1:
            tl.store(
                Q + row[:, None] * QROW + grp[:, None] * GROUP + g[None, :],
                _encode_S(_quot_S(value, scale[:, None]), value),
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
        elif MODE == 2:
            codes = tl.reshape(
                _encode_S(_quot_S(value, scale[:, None]), value), (BR, DIM)
            )
            bits = scale.to(tl.uint32, bitcast=True)
            sbits = tl.reshape(bits, (BR, GROUPS))
            sbytes = ((sbits[:, :, None] >> (b * 8)[None, None, :]) & 255).to(
                tl.uint8
            )
            sbytes = tl.reshape(
                tl.broadcast_to(
                    sbytes[:, None, :, :], (BR, DIM // (GROUPS * 4), GROUPS, 4)
                ),
                (BR, DIM),
            )
            out = tl.reshape(
                tl.permute(tl.join(codes, sbytes), (0, 2, 1)), (BR, DIM * 2)
            )
            off = tl.arange(0, DIM * 2)
            tl.store(
                Q
                + (pid * BR + tl.arange(0, BR)[:, None]) * QROW
                + off[None, :],
                out,
                off[None, :] < QROW,
                cache_modifier=".cs",
            )
        else:
            codes = tl.reshape(
                _encode_S(_quot_S(value, scale[:, None]), value),
                (BR, DIM // 4, 4),
            )
            packed = tl.sum(codes.to(tl.uint32) << (b * 8)[None, None, :], 2)
            bits = scale.to(tl.uint32, bitcast=True)
            rr = tl.arange(0, BR)
            sbits = tl.reshape(
                tl.broadcast_to(
                    tl.reshape(bits, (BR, GROUPS))[:, None, :],
                    (BR, DIM // (4 * GROUPS), GROUPS),
                ),
                (BR, DIM // 4),
            )
            out = tl.reshape(
                tl.permute(tl.join(packed, sbits), (0, 2, 1)), (BR, DIM // 2)
            )
            off = tl.arange(0, DIM // 2)
            tl.store(
                Q.to(tl.pointer_type(tl.uint32))
                + (pid * BR + rr[:, None]) * (QROW // 4)
                + off[None, :],
                out,
                off[None, :] < QROW // 4,
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
        elif tokens >= 2048 and tokens % 4 == 0 and dim_rope % 2 == 0:
            _partition_memory_S.run(
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
def _e4m3_code_L(q):
    u = q.to(tl.uint32, bitcast=True) + 0x80000
    code = (u >> 20).to(tl.int32) - 960
    return tl.minimum(tl.maximum(code, 0), 126)


@triton.jit
def _mag_sign_L(v):
    if v.dtype == tl.bfloat16:
        u = v.to(tl.uint16, bitcast=True).to(tl.uint32)
        sgn = (u >> 8) & 0x80
        mag = ((u & 0x7FFF) << 16).to(tl.float32, bitcast=True)
        return mag, sgn
    else:
        u = v.to(tl.float32).to(tl.uint32, bitcast=True)
        sgn = (u >> 24) & 0x80
        mag = (u & 0x7FFFFFFF).to(tl.float32, bitcast=True)
        return mag, sgn


@triton.jit
def _qsep_fused_kernel_L(
    kn_ptr,
    rope_in,
    nope_out,
    rope_out,
    T: tl.constexpr,
    NT: tl.constexpr,
    GS: tl.constexpr,
    DR: tl.constexpr,
    SROW: tl.constexpr,
    DN: tl.constexpr,
    RB: tl.constexpr,
    NTP: tl.constexpr,
    GSP: tl.constexpr,
    GS_EQ: tl.constexpr,
    NRP: tl.constexpr,
    ROW: tl.constexpr,
    WIDE: tl.constexpr,
    EVICT: tl.constexpr,
    FULLROW: tl.constexpr,
):
    nope8 = nope_out.to(tl.pointer_type(tl.uint8))
    s32 = nope_out.to(tl.pointer_type(tl.float32))
    r64_out = rope_out.to(tl.pointer_type(tl.uint64))
    r64_in = rope_in.to(tl.pointer_type(tl.uint64))
    if WIDE:
        r = (
            tl.program_id(0).to(tl.int64) * ROW
            + tl.arange(0, ROW).to(tl.int64)[:, None, None]
        )
    else:
        r = tl.program_id(0) * ROW + tl.arange(0, ROW)[:, None, None]
    t = tl.arange(0, NTP)[None, :, None]
    lane = tl.arange(0, GSP)[None, None, :]
    tm = t < NT
    if FULLROW:
        if GS_EQ:
            lm = tm
        else:
            lm = tm & (lane < GS)
        srm = tm
    else:
        rm = r < T
        if GS_EQ:
            lm = rm & tm
        else:
            lm = rm & tm & (lane < GS)
        srm = rm & tm
    xoff = r * DN + t * GS + lane
    if EVICT:
        v = tl.load(
            kn_ptr + xoff, mask=lm, other=0.0, eviction_policy="evict_first"
        )
    else:
        v = tl.load(kn_ptr + xoff, mask=lm, other=0.0)
    mag, sgn = _mag_sign_L(v)
    amax = tl.max(mag, axis=2)
    scale2 = amax * 0.002232142857142857
    scale = scale2[:, :, None]
    rcp = tl.inline_asm_elementwise(
        "v_rcp_f32_e32 $0, $1;",
        "=v,v",
        [scale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    q = mag * rcp
    code = _e4m3_code_L(q)
    y8 = (code | sgn).to(tl.uint8)
    ooff = r * SROW + t * GS + lane
    if EVICT:
        tl.store(nope8 + ooff, y8, mask=lm, eviction_policy="evict_first")
    else:
        tl.store(nope8 + ooff, y8, mask=lm)
    soff = r * (SROW // 4) + (DN // 4) + t
    tl.store(s32 + soff, scale, mask=srm)
    nr = tl.arange(0, NRP)[None, None, :]
    if FULLROW:
        nm = nr < (DR // 4)
    else:
        nm = rm & (nr < (DR // 4))
    roff_in = r * (DR // 4) + nr
    roff_out = r * (RB // 8) + nr
    if EVICT:
        rv = tl.load(
            r64_in + roff_in, mask=nm, other=0.0, eviction_policy="evict_first"
        )
    else:
        rv = tl.load(r64_in + roff_in, mask=nm, other=0.0)
    if EVICT:
        tl.store(
            r64_out + roff_out, rv, mask=nm, eviction_policy="evict_first"
        )
    else:
        tl.store(r64_out + roff_out, rv, mask=nm)


@triton.jit
def _qsep_split_kernel_L(
    kn_ptr,
    rope_in,
    nope_out,
    rope_out,
    T: tl.constexpr,
    GS: tl.constexpr,
    DR: tl.constexpr,
    SROW: tl.constexpr,
    DN: tl.constexpr,
    RB: tl.constexpr,
    GSP: tl.constexpr,
    GS_EQ: tl.constexpr,
    NRP: tl.constexpr,
    ROW: tl.constexpr,
    WIDE: tl.constexpr,
    FULLROW: tl.constexpr,
):
    nope8 = nope_out.to(tl.pointer_type(tl.uint8))
    s32 = nope_out.to(tl.pointer_type(tl.float32))
    r64_out = rope_out.to(tl.pointer_type(tl.uint64))
    r64_in = rope_in.to(tl.pointer_type(tl.uint64))
    t = tl.program_id(1)
    if WIDE:
        r = (
            tl.program_id(0).to(tl.int64) * ROW
            + tl.arange(0, ROW).to(tl.int64)[:, None]
        )
    else:
        r = tl.program_id(0) * ROW + tl.arange(0, ROW)[:, None]
    lane = tl.arange(0, GSP)[None, :]
    if FULLROW:
        rm = True
        lm = True if GS_EQ else (lane < GS)
    else:
        rm = r < T
        lm = rm if GS_EQ else (rm & (lane < GS))
    v = tl.load(kn_ptr + r * DN + t * GS + lane, mask=lm, other=0.0)
    mag, sgn = _mag_sign_L(v)
    amax = tl.max(mag, axis=1)
    scale2 = amax * 0.002232142857142857
    scale = scale2[:, None]
    rcp = tl.inline_asm_elementwise(
        "v_rcp_f32_e32 $0, $1;",
        "=v,v",
        [scale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    q = mag * rcp
    code = _e4m3_code_L(q)
    y8 = (code | sgn).to(tl.uint8)
    tl.store(nope8 + r * SROW + t * GS + lane, y8, mask=lm)
    tl.store(s32 + r * (SROW // 4) + (DN // 4) + t, scale, mask=rm)
    if t == 0:
        nr = tl.arange(0, NRP)[None, :]
        nm = nr < (DR // 4)
        if not FULLROW:
            nm = rm & nm
        rv = tl.load(r64_in + r * (DR // 4) + nr, mask=nm, other=0.0)
        tl.store(r64_out + r * (RB // 8) + nr, rv, mask=nm)


@triton.jit
def _qsep_generic_kernel_L(
    kn_ptr,
    rope_in,
    rope_dst,
    nope8_ptr,
    T,
    nt,
    gs,
    dr,
    srow,
    dn,
    snr,
    snc,
    rnr,
    rnc,
    rdr,
    HAS_NT: tl.constexpr,
    GSP: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    r = pid // nt
    t = pid % nt
    row_ok = r < T
    if HAS_NT:
        tile_ok = row_ok & (t < nt)
        lane = tl.arange(0, GSP).to(tl.int64)
        lm = tile_ok & (lane < gs)
        v = tl.load(
            kn_ptr + r * snr + (t * gs + lane) * snc, mask=lm, other=0.0
        )
        mag, sgn = _mag_sign_L(v)
        amax = tl.max(mag, axis=0)
        scale = amax * 0.002232142857142857
        rcp = 1.0 / scale
        q = mag * rcp
        code = _e4m3_code_L(q)
        y8 = (code | sgn).to(tl.uint8)
        ooff = r * srow + t * gs + lane
        tl.store(nope8_ptr + ooff, y8, mask=lm)
        sb = scale.to(tl.uint32, bitcast=True)
        k4 = tl.arange(0, 4).to(tl.int64)
        sbyte = ((sb >> (8 * k4)) & 0xFF).to(tl.uint8)
        soff = r * srow + (dn + t * 4) + k4
        tl.store(nope8_ptr + soff, sbyte, mask=tile_ok)
    if t == 0:
        c = tl.arange(0, CHUNK).to(tl.int64)
        base_out = r * rdr
        for off in range(0, dr, CHUNK):
            cm = row_ok & ((off + c) < dr)
            rv = tl.load(
                rope_in + r * rnr + (off + c) * rnc, mask=cm, other=0.0
            )
            tl.store(rope_dst + base_out + off + c, rv, mask=cm)


def quantize_k_cache_fast_separate_large(k_nope, k_rope, group_size=128):
    num_tokens, dim_nope = k_nope.shape
    dim_rope = k_rope.shape[1]
    num_tiles = dim_nope // group_size
    esize = k_rope.element_size()
    nope_bytes = dim_nope + num_tiles * 4
    rope_bytes = dim_rope * esize

    nope_part_u8 = torch.empty(
        num_tokens, 1, nope_bytes, dtype=torch.uint8, device=k_nope.device
    )
    rope_part_u8 = torch.empty(
        num_tokens, 1, rope_bytes, dtype=torch.uint8, device=k_rope.device
    )
    if num_tokens == 0:
        return nope_part_u8, rope_part_u8

    if (
        k_nope.is_contiguous()
        and k_rope.is_contiguous()
        and esize == 2
        and dim_nope % 4 == 0
        and dim_rope % 4 == 0
        and group_size <= 2048
        and dim_rope <= 16384
        and k_rope.data_ptr() % 8 == 0
    ):
        gsp = 1 << (group_size - 1).bit_length()
        if gsp < 2:
            gsp = 2
        ntp = 1 << (num_tiles - 1).bit_length() if num_tiles > 1 else 2
        nrp = 1 << ((dim_rope // 4 - 1).bit_length())
        SPLIT = False
        if num_tokens >= 5568:
            ROW, W, EV = 8, 2, False
        elif num_tokens >= 3072:
            SPLIT, ROW, W, EV = True, 32, 4, False
        elif num_tokens >= 1024:
            SPLIT, ROW, W, EV = True, 16, 4, False
        elif num_tokens >= 512:
            ROW, W, EV = 4, 2, False
        else:
            ROW, W, EV = 2, 1, False
        grid = ((num_tokens + ROW - 1) // ROW,)
        if SPLIT:
            _qsep_split_kernel_L[(grid[0], num_tiles)](
                k_nope,
                k_rope,
                nope_part_u8,
                rope_part_u8,
                num_tokens,
                group_size,
                dim_rope,
                nope_bytes,
                dim_nope,
                rope_bytes,
                gsp,
                group_size == gsp,
                nrp,
                ROW,
                num_tokens % ROW == 0,
                num_tokens * nope_bytes >= 2147483648,
                num_warps=W,
                num_stages=2,
            )
            return nope_part_u8, rope_part_u8
        _qsep_fused_kernel_L[grid](
            k_nope,
            k_rope,
            nope_part_u8,
            rope_part_u8,
            num_tokens,
            num_tiles,
            group_size,
            dim_rope,
            nope_bytes,
            dim_nope,
            rope_bytes,
            ntp,
            gsp,
            group_size == gsp,
            nrp,
            ROW,
            num_tokens % ROW == 0,
            num_tokens * nope_bytes >= 2147483648,
            EV,
            num_warps=W,
            num_stages=2,
        )
        return nope_part_u8, rope_part_u8

    nt_work = max(num_tiles, 1)
    grid = (num_tokens * nt_work,)
    if esize == 2:
        rope_view, dst_view = k_rope.view(torch.uint16), rope_part_u8.view(
            torch.uint16
        )
        rdr = rope_bytes // 2
    elif esize == 4:
        rope_view, dst_view = k_rope.view(torch.uint32), rope_part_u8.view(
            torch.uint32
        )
        rdr = rope_bytes // 4
    else:
        rope_view, dst_view = k_rope, rope_part_u8
        rdr = rope_bytes
    _qsep_generic_kernel_L[grid](
        k_nope,
        rope_view,
        dst_view,
        nope_part_u8,
        num_tokens,
        num_tiles,
        group_size,
        dim_rope,
        nope_bytes,
        dim_nope,
        k_nope.stride(0),
        k_nope.stride(1),
        k_rope.stride(0),
        k_rope.stride(1),
        rdr,
        HAS_NT=num_tiles >= 1,
        GSP=max(triton.next_power_of_2(group_size), 2),
        CHUNK=1024,
        num_warps=4,
        num_stages=2,
    )
    return nope_part_u8, rope_part_u8


def quantize_k_cache_fast_separate(k_nope, k_rope, group_size=128):
    if k_nope.shape[0] <= 128:
        return quantize_k_cache_fast_separate_small(k_nope, k_rope, group_size)
    return quantize_k_cache_fast_separate_large(k_nope, k_rope, group_size)


__all__ = ["quantize_k_cache_fast_separate"]
