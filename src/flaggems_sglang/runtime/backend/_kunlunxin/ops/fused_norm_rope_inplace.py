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
import triton.experimental.tle as tle
import triton.language as tl


def _requires_wide(
    kv, weight, freqs_cis, positions, packed=False, freq64=False
):
    m, d = kv.shape
    divisor = 2 if packed else 1
    xspan = max(0, m - 1) * (kv.stride(0) // divisor) + max(
        0, d // divisor - 1
    )
    ospan = max(0, m * d // divisor - 1)
    wspan = (
        max(0, d - 1) * weight.stride(0) // divisor
        if weight is not None
        else 0
    )
    fspan = max(0, freqs_cis.shape[0] - 1) * freqs_cis.stride(0) + max(
        0, freqs_cis.shape[1] - 1
    ) * freqs_cis.stride(1)
    fspan = fspan if freq64 else 2 * fspan + 1
    pspan = max(0, m - 1) * positions.stride(0) if positions is not None else 0
    return max(xspan, ospan, wspan, fspan, pspan) > 2147483647


@triton.jit
def _wide_mul(index, stride, WIDE: tl.constexpr):
    if WIDE:
        return index.to(tl.int64) * stride
    else:
        return index * stride


@triton.jit
def _norm(
    X,
    W,
    T,
    D: tl.constexpr,
    SX: tl.constexpr,
    SW: tl.constexpr,
    EPS: tl.constexpr,
    HAS_W: tl.constexpr,
    B: tl.constexpr,
    WIDE: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    if D == B:
        x = tl.load(X + _wide_mul(row, SX, WIDE) + c).to(tl.float32)
    else:
        x = tl.load(X + _wide_mul(row, SX, WIDE) + tl.minimum(c, D - 1)).to(
            tl.float32
        )
        x = tl.where(c < D, x, 0.0)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    y = x * inv
    if HAS_W:
        w = tl.load(W + _wide_mul(tl.minimum(c, D - 1), SW, WIDE)).to(
            tl.float32
        )
        y = y * w
    tl.store(T + _wide_mul(row, D, WIDE) + c, y, c < D)


@triton.jit
def _rope(
    T,
    F,
    P,
    O,
    D: tl.constexpr,
    R: tl.constexpr,
    SF0: tl.constexpr,
    SF1: tl.constexpr,
    SP: tl.constexpr,
    HAS_P: tl.constexpr,
    B: tl.constexpr,
    WIDE: tl.constexpr,
    NF: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    cc = tl.minimum(c, D - 1)
    y = tl.load(T + _wide_mul(row, D, WIDE) + cc)
    if R > 0:
        j = tl.maximum(cc - (D - R), 0)
        other = D - R + (j ^ 1)
        z = tl.load(T + _wide_mul(row, D, WIDE) + other)
        pos = row
        if HAS_P:
            pos = tl.load(P + _wide_mul(row, SP, WIDE))
            pos = tl.where(pos < 0, pos + NF, pos)
        fr = tl.load(
            F + _wide_mul(pos, SF0, WIDE) + _wide_mul(j // 2, SF1, WIDE)
        )
        fi = tl.load(
            F + _wide_mul(pos, SF0, WIDE) + _wide_mul(j // 2, SF1, WIDE) + 1
        )
        signed = tl.where(j % 2 == 0, -z, z)
        rotated = y * fr + signed * fi
        y = tl.where(c >= D - R, rotated, y)
    tl.store(O + _wide_mul(row, D, WIDE) + c, y, c < D)


def _baseline(kv, weight, eps, freqs_cis, positions):
    WIDE = _requires_wide(kv, weight, freqs_cis, positions, False, False)
    m, d = kv.shape
    r = 2 * freqs_cis.shape[-1]
    f = torch.view_as_real(freqs_cis)
    out = torch.empty((m, d), dtype=kv.dtype, device=kv.device)
    if m == 0 or d == 0:
        return out
    temp = torch.empty((m, d), dtype=torch.float32, device=kv.device)
    b = triton.next_power_of_2(d)
    _norm[m,](
        kv,
        weight,
        temp,
        d,
        kv.stride(0),
        weight.stride(0) if weight is not None else 0,
        eps,
        weight is not None,
        b,
        WIDE,
        enable_fp_fusion=False,
    )
    _rope[m,](
        temp,
        f,
        positions,
        out,
        d,
        r,
        f.stride(0),
        f.stride(1),
        positions.stride(0) if positions is not None else 0,
        positions is not None,
        b,
        WIDE,
        freqs_cis.shape[0],
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _encode(value):
    return (
        value.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
    )


@triton.jit
def _packed_norm(
    X,
    W,
    F,
    P,
    O,
    M: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    SX: tl.constexpr,
    SF0: tl.constexpr,
    SF1: tl.constexpr,
    SP: tl.constexpr,
    EPS: tl.constexpr,
    HW: tl.constexpr,
    HP: tl.constexpr,
    BP: tl.constexpr,
    BR: tl.constexpr,
    GRID: tl.constexpr,
    F64: tl.constexpr,
    WIDE: tl.constexpr,
    NF: tl.constexpr,
):
    for block in range(tl.program_id(0), tl.cdiv(M, BR), GRID):
        rows = block * BR + tl.arange(0, BR)
        source = tl.minimum(rows, M - 1)
        p = tl.arange(0, BP)
        pp = p
        bits = tl.load(
            X.to(tl.pointer_type(tl.uint32))
            + _wide_mul(source[:, None], SX // 2, WIDE)
            + pp[None, :],
            pp[None, :] < D // 2,
            0,
        )
        xr = ((bits & 65535) << 16).to(tl.float32, bitcast=True)
        xi = (bits & 4294901760).to(tl.float32, bitcast=True)
        xr = tl.where(p[None, :] < D // 2, xr, 0.0)
        xi = tl.where(p[None, :] < D // 2, xi, 0.0)
        ss = tl.sum(xr * xr + xi * xi, 1)
        inv = tl.rsqrt(ss / D + EPS)
        yr = xr * inv[:, None]
        yi = xi * inv[:, None]
        if HW:
            wb = tl.load(
                W.to(tl.pointer_type(tl.uint32))
                + tl.broadcast_to(pp[None, :], (BR, BP)),
                pp[None, :] < D // 2,
                0,
            )
            wr = ((wb & 65535) << 16).to(tl.float32, bitcast=True)
            wi = (wb & 4294901760).to(tl.float32, bitcast=True)
            yr *= wr
            yi *= wi
        if R > 0:
            pos = source
            if HP:
                pos = tl.load(P + _wide_mul(source, SP, WIDE))
                pos = tl.where(pos < 0, pos + NF, pos)
            j = p - (D - R) // 2
            fm = (j[None, :] >= 0) & (j[None, :] < R // 2)
            if F64:
                fb = tl.load(
                    F.to(tl.pointer_type(tl.uint64))
                    + _wide_mul(pos[:, None], SF0 // 2, WIDE)
                    + _wide_mul(j[None, :], SF1 // 2, WIDE),
                    fm,
                    0,
                )
                fr = fb.to(tl.uint32).to(tl.float32, bitcast=True)
                fi = (fb >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
            else:
                fr = tl.load(
                    F
                    + _wide_mul(pos[:, None], SF0, WIDE)
                    + _wide_mul(j[None, :], SF1, WIDE),
                    fm,
                    0,
                )
                fi = tl.load(
                    F
                    + _wide_mul(pos[:, None], SF0, WIDE)
                    + _wide_mul(j[None, :], SF1, WIDE)
                    + 1,
                    fm,
                    0,
                )
            rr = yr * fr - yi * fi
            ri = yr * fi + yi * fr
            yr = tl.where(j[None, :] >= 0, rr, yr)
            yi = tl.where(j[None, :] >= 0, ri, yi)
        out = _encode(yr) | _encode(yi) << 16
        tl.store(
            O.to(tl.pointer_type(tl.uint32))
            + _wide_mul(rows[:, None], D // 2, WIDE)
            + p[None, :],
            out,
            (rows[:, None] < M) & (p[None, :] < D // 2),
        )


def _combined(kv, weight, eps, freqs_cis, positions):
    WIDE = _requires_wide(
        kv, weight, freqs_cis, positions, True, kv.shape[0] >= 64
    )
    m, d = kv.shape
    packed = (
        kv.dtype == torch.bfloat16
        and d > 0
        and (d % 2 == 0)
        and (kv.stride(0) % 2 == 0)
        and (kv.storage_offset() % 2 == 0)
    )
    if weight is not None:
        packed = (
            packed
            and weight.dtype == torch.bfloat16
            and (weight.stride(0) == 1)
            and (weight.storage_offset() % 2 == 0)
        )
    if not packed or m == 0:
        return _baseline(kv, weight, eps, freqs_cis, positions)
    r = 2 * freqs_cis.shape[-1]
    f = torch.view_as_real(freqs_cis)
    out = torch.empty((m, d), dtype=kv.dtype, device=kv.device)
    br = 1 if m <= 8 and d == 576 and (r == 64) else 4
    grid = min(triton.cdiv(m, br), 12)
    _packed_norm.run(
        kv,
        weight,
        f,
        positions,
        out,
        m,
        d,
        r,
        kv.stride(0),
        f.stride(0),
        f.stride(1),
        positions.stride(0) if positions is not None else 0,
        eps,
        weight is not None,
        positions is not None,
        triton.next_power_of_2(d // 2),
        br,
        grid,
        m >= 64,
        WIDE,
        freqs_cis.shape[0],
        grid=(grid,),
        warmup=False,
        num_warps=4,
        groups_per_cluster=1,
        enable_fp_fusion=False,
        isCloseDtypeConvert=True,
    )
    return out


@triton.jit
def _emit_part(
    bits,
    W,
    F,
    P,
    O,
    row,
    inv,
    D: tl.constexpr,
    R: tl.constexpr,
    SX: tl.constexpr,
    SF0: tl.constexpr,
    SF1: tl.constexpr,
    SP: tl.constexpr,
    HW: tl.constexpr,
    HP: tl.constexpr,
    BR: tl.constexpr,
    BP: tl.constexpr,
    PART: tl.constexpr,
    WIDE: tl.constexpr,
    NF: tl.constexpr,
):
    p = PART * BP + tl.arange(0, BP)
    xr = ((bits & 65535) << 16).to(tl.float32, bitcast=True)
    xi = (bits & 4294901760).to(tl.float32, bitcast=True)
    yr = xr * inv[:, None]
    yi = xi * inv[:, None]
    if HW:
        wb = tl.load(
            W.to(tl.pointer_type(tl.uint32))
            + tl.broadcast_to(p[None, :], (BR, BP)),
            offset_state_policy="continuous",
        )
        wr = ((wb & 65535) << 16).to(tl.float32, bitcast=True)
        wi = (wb & 4294901760).to(tl.float32, bitcast=True)
        yr *= wr
        yi *= wi
    if PART == 8:
        pos = row
        if HP:
            pos = tl.load(P + _wide_mul(row, SP, WIDE))
            pos = tl.where(pos < 0, pos + NF, pos)
        j = p - (D - R) // 2
        fb = tl.load(
            F.to(tl.pointer_type(tl.uint64))
            + _wide_mul(pos[:, None], SF0 // 2, WIDE)
            + _wide_mul(j[None, :], SF1 // 2, WIDE),
            offset_state_policy="continuous",
        )
        fr = fb.to(tl.uint32).to(tl.float32, bitcast=True)
        fi = (fb >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
        rr = yr * fr - yi * fi
        ri = yr * fi + yi * fr
        value = _encode(rr) | _encode(ri) << 16
    else:
        value = _encode(yr) | _encode(yi) << 16
    tl.store(
        O.to(tl.pointer_type(tl.uint32))
        + _wide_mul(row[:, None], D // 2, WIDE)
        + p[None, :],
        value,
        offset_state_policy="continuous",
    )


@triton.jit
def _nine_parts(
    X,
    W,
    F,
    P,
    O,
    M: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    SX: tl.constexpr,
    SF0: tl.constexpr,
    SF1: tl.constexpr,
    SP: tl.constexpr,
    EPS: tl.constexpr,
    HW: tl.constexpr,
    HP: tl.constexpr,
    BP: tl.constexpr,
    BR: tl.constexpr,
    GRID: tl.constexpr,
    WIDE: tl.constexpr,
    NF: tl.constexpr,
):
    p = tl.arange(0, BP)
    for block in range(tl.program_id(0), M // BR, GRID):
        row = block * BR + tl.arange(0, BR)
        ss = tl.full((BR, BP), 0.0, tl.float32)
        bits0 = tl.load(
            X.to(tl.pointer_type(tl.uint32))
            + _wide_mul(row[:, None], SX // 2, WIDE)
            + 0 * BP
            + p[None, :],
            offset_state_policy="continuous",
        )
        xr = ((bits0 & 65535) << 16).to(tl.float32, bitcast=True)
        xi = (bits0 & 4294901760).to(tl.float32, bitcast=True)
        ss += xr * xr + xi * xi
        bits1 = tl.load(
            X.to(tl.pointer_type(tl.uint32))
            + _wide_mul(row[:, None], SX // 2, WIDE)
            + 1 * BP
            + p[None, :],
            offset_state_policy="continuous",
        )
        xr = ((bits1 & 65535) << 16).to(tl.float32, bitcast=True)
        xi = (bits1 & 4294901760).to(tl.float32, bitcast=True)
        ss += xr * xr + xi * xi
        bits2 = tl.load(
            X.to(tl.pointer_type(tl.uint32))
            + _wide_mul(row[:, None], SX // 2, WIDE)
            + 2 * BP
            + p[None, :],
            offset_state_policy="continuous",
        )
        xr = ((bits2 & 65535) << 16).to(tl.float32, bitcast=True)
        xi = (bits2 & 4294901760).to(tl.float32, bitcast=True)
        ss += xr * xr + xi * xi
        bits3 = tl.load(
            X.to(tl.pointer_type(tl.uint32))
            + _wide_mul(row[:, None], SX // 2, WIDE)
            + 3 * BP
            + p[None, :],
            offset_state_policy="continuous",
        )
        xr = ((bits3 & 65535) << 16).to(tl.float32, bitcast=True)
        xi = (bits3 & 4294901760).to(tl.float32, bitcast=True)
        ss += xr * xr + xi * xi
        bits4 = tl.load(
            X.to(tl.pointer_type(tl.uint32))
            + _wide_mul(row[:, None], SX // 2, WIDE)
            + 4 * BP
            + p[None, :],
            offset_state_policy="continuous",
        )
        xr = ((bits4 & 65535) << 16).to(tl.float32, bitcast=True)
        xi = (bits4 & 4294901760).to(tl.float32, bitcast=True)
        ss += xr * xr + xi * xi
        bits5 = tl.load(
            X.to(tl.pointer_type(tl.uint32))
            + _wide_mul(row[:, None], SX // 2, WIDE)
            + 5 * BP
            + p[None, :],
            offset_state_policy="continuous",
        )
        xr = ((bits5 & 65535) << 16).to(tl.float32, bitcast=True)
        xi = (bits5 & 4294901760).to(tl.float32, bitcast=True)
        ss += xr * xr + xi * xi
        bits6 = tl.load(
            X.to(tl.pointer_type(tl.uint32))
            + _wide_mul(row[:, None], SX // 2, WIDE)
            + 6 * BP
            + p[None, :],
            offset_state_policy="continuous",
        )
        xr = ((bits6 & 65535) << 16).to(tl.float32, bitcast=True)
        xi = (bits6 & 4294901760).to(tl.float32, bitcast=True)
        ss += xr * xr + xi * xi
        bits7 = tl.load(
            X.to(tl.pointer_type(tl.uint32))
            + _wide_mul(row[:, None], SX // 2, WIDE)
            + 7 * BP
            + p[None, :],
            offset_state_policy="continuous",
        )
        xr = ((bits7 & 65535) << 16).to(tl.float32, bitcast=True)
        xi = (bits7 & 4294901760).to(tl.float32, bitcast=True)
        ss += xr * xr + xi * xi
        bits8 = tl.load(
            X.to(tl.pointer_type(tl.uint32))
            + _wide_mul(row[:, None], SX // 2, WIDE)
            + 8 * BP
            + p[None, :],
            offset_state_policy="continuous",
        )
        xr = ((bits8 & 65535) << 16).to(tl.float32, bitcast=True)
        xi = (bits8 & 4294901760).to(tl.float32, bitcast=True)
        ss += xr * xr + xi * xi
        inv = tl.rsqrt(tl.sum(ss, 1) / D + EPS)
        _emit_part(
            bits0,
            W,
            F,
            P,
            O,
            row,
            inv,
            D,
            R,
            SX,
            SF0,
            SF1,
            SP,
            HW,
            HP,
            BR,
            BP,
            0,
            WIDE,
            NF,
        )
        _emit_part(
            bits1,
            W,
            F,
            P,
            O,
            row,
            inv,
            D,
            R,
            SX,
            SF0,
            SF1,
            SP,
            HW,
            HP,
            BR,
            BP,
            1,
            WIDE,
            NF,
        )
        _emit_part(
            bits2,
            W,
            F,
            P,
            O,
            row,
            inv,
            D,
            R,
            SX,
            SF0,
            SF1,
            SP,
            HW,
            HP,
            BR,
            BP,
            2,
            WIDE,
            NF,
        )
        _emit_part(
            bits3,
            W,
            F,
            P,
            O,
            row,
            inv,
            D,
            R,
            SX,
            SF0,
            SF1,
            SP,
            HW,
            HP,
            BR,
            BP,
            3,
            WIDE,
            NF,
        )
        _emit_part(
            bits4,
            W,
            F,
            P,
            O,
            row,
            inv,
            D,
            R,
            SX,
            SF0,
            SF1,
            SP,
            HW,
            HP,
            BR,
            BP,
            4,
            WIDE,
            NF,
        )
        _emit_part(
            bits5,
            W,
            F,
            P,
            O,
            row,
            inv,
            D,
            R,
            SX,
            SF0,
            SF1,
            SP,
            HW,
            HP,
            BR,
            BP,
            5,
            WIDE,
            NF,
        )
        _emit_part(
            bits6,
            W,
            F,
            P,
            O,
            row,
            inv,
            D,
            R,
            SX,
            SF0,
            SF1,
            SP,
            HW,
            HP,
            BR,
            BP,
            6,
            WIDE,
            NF,
        )
        _emit_part(
            bits7,
            W,
            F,
            P,
            O,
            row,
            inv,
            D,
            R,
            SX,
            SF0,
            SF1,
            SP,
            HW,
            HP,
            BR,
            BP,
            7,
            WIDE,
            NF,
        )
        _emit_part(
            bits8,
            W,
            F,
            P,
            O,
            row,
            inv,
            D,
            R,
            SX,
            SF0,
            SF1,
            SP,
            HW,
            HP,
            BR,
            BP,
            8,
            WIDE,
            NF,
        )


def _nine(kv, weight, eps, freqs_cis, positions):
    WIDE = _requires_wide(kv, weight, freqs_cis, positions, True, True)
    m, d = kv.shape
    r = 2 * freqs_cis.shape[-1]
    f = torch.view_as_real(freqs_cis)
    out = torch.empty((m, d), dtype=kv.dtype, device=kv.device)
    br = 64
    grid = min(m // br, 12)
    _nine_parts.run(
        kv,
        weight,
        f,
        positions,
        out,
        m,
        d,
        r,
        kv.stride(0),
        f.stride(0),
        f.stride(1),
        positions.stride(0) if positions is not None else 0,
        eps,
        weight is not None,
        positions is not None,
        32,
        br,
        grid,
        WIDE,
        freqs_cis.shape[0],
        grid=(grid,),
        warmup=False,
        groups_per_cluster=64,
        enable_fp_fusion=False,
        isCloseUnrollControl=True,
        isCloseDtypeConvert=True,
        isCloseMemoryAsync=m > grid * br,
    )
    return out


def _recommended199(kv, weight, eps, freqs_cis, positions):
    m, d = kv.shape
    fast = (
        d == 576
        and 2 * freqs_cis.shape[-1] == 64
        and (m >= 512)
        and (m % 64 == 0)
    )
    fast = (
        fast
        and kv.dtype == torch.bfloat16
        and (kv.stride(1) == 1)
        and (kv.stride(0) % 32 == 0)
        and (kv.data_ptr() % 64 == 0)
    )
    fast = (
        fast
        and freqs_cis.dtype == torch.complex64
        and (freqs_cis.stride(1) == 1)
        and (freqs_cis.stride(0) % 8 == 0)
        and (freqs_cis.data_ptr() % 64 == 0)
    )
    if weight is not None:
        fast = (
            fast
            and weight.dtype == torch.bfloat16
            and (weight.stride(0) == 1)
            and (weight.data_ptr() % 64 == 0)
        )
    if fast:
        return _nine(kv, weight, eps, freqs_cis, positions)
    return _combined(kv, weight, eps, freqs_cis, positions)


RAW199 = (
    "typedef unsigned short __attribute__((address_space(1))) gu16;\n"
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "union bits199 { f32v f; u32v u; };\n"
    "namespace xpu { __attribute__((device)) float rsqrtf(float); }\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode199(f32v value) "
    "{\n"
    "    bits199 bits;\n"
    "    bits.f = value;\n"
    "    u32v original = bits.u;\n"
    "    u32v word = (original + 32768u) >> 16;\n"
    "    return word;\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _device_row199(\n'
    "    gu16* X, gu16* W, gf32* F, gu16* O,\n"
    "    int M, float EPS, int HW, int MODE, int pid, int np) {\n"
    "    unsigned int xb[288] __attribute__((aligned(64)));\n"
    "    unsigned int wb[288] __attribute__((aligned(64)));\n"
    "    float fb[64] __attribute__((aligned(64)));\n"
    "    float fc[32] __attribute__((aligned(64)));\n"
    "    float fs[32] __attribute__((aligned(64)));\n"
    "    float sums[16] __attribute__((aligned(64)));\n"
    "    const int core = __builtin_xpu_core_id();\n"
    "    int row = pid * 64 + core;\n"
    "    if (row >= M) return;\n"
    "    if (HW && MODE >= 2) {\n"
    "        __builtin_xpu_gm2lm_v3(wb, W, 0, 1152);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "    for (; row < M; row += np * 64) {\n"
    "        __builtin_xpu_gm2lm_v3(xb, X + row * 576, 0, 1152);\n"
    "        if (MODE == 3)\n"
    "            __builtin_xpu_gm2lm_v3(fb, F + row * 64, 0, 256);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        if (MODE != 0) {\n"
    "            f32v acc = (f32v)(0.0f);\n"
    "            for (int pair = 0; pair < 288; pair += 16) {\n"
    "                u32v packed = *((u32v*)(xb + pair));\n"
    "                bits199 real, imag;\n"
    "                real.u = packed << 16;\n"
    "                imag.u = packed & 0xffff0000u;\n"
    "                acc = acc + (real.f * real.f + imag.f * imag.f);\n"
    "            }\n"
    "            *((f32v*)sums) = acc;\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            const float total = ((((sums[0] + sums[1]) + (sums[2]"
    " + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]))) + ((("
    "sums[8] + sums[9]) + (sums[10] + sums[11])) + ((sums[12] + sums[1"
    "3]) + (sums[14] + sums[15]))));\n"
    "            const float inv = xpu::rsqrtf(total / 576.0f + EPS);\n"
    "            if (MODE == 3) {\n"
    "                #pragma clang loop vectorize(disable)\n"
    "                for (int pair = 0; pair < 32; ++pair) {\n"
    "                    fc[pair] = fb[2 * pair];\n"
    "                    fs[pair] = fb[2 * pair + 1];\n"
    "                }\n"
    "            }\n"
    "            for (int pair = 0; pair < 256; pair += 16) {\n"
    "                u32v packed = *((u32v*)(xb + pair));\n"
    "                bits199 real, imag;\n"
    "                real.u = packed << 16;\n"
    "                imag.u = packed & 0xffff0000u;\n"
    "                f32v yr = real.f * inv;\n"
    "                f32v yi = imag.f * inv;\n"
    "                if (HW && MODE >= 2) {\n"
    "                    u32v weight = *((u32v*)(wb + pair));\n"
    "                    bits199 wr, wi;\n"
    "                    wr.u = weight << 16;\n"
    "                    wi.u = weight & 0xffff0000u;\n"
    "                    yr = yr * wr.f;\n"
    "                    yi = yi * wi.f;\n"
    "                }\n"
    "                *((u32v*)(xb + pair)) = encode199(yr) | (encode19"
    "9(yi) << 16);\n"
    "            }\n"
    "            for (int pair = 256; pair < 288; pair += 16) {\n"
    "                u32v packed = *((u32v*)(xb + pair));\n"
    "                bits199 real, imag;\n"
    "                real.u = packed << 16;\n"
    "                imag.u = packed & 0xffff0000u;\n"
    "                f32v yr = real.f * inv;\n"
    "                f32v yi = imag.f * inv;\n"
    "                if (HW && MODE >= 2) {\n"
    "                    u32v weight = *((u32v*)(wb + pair));\n"
    "                    bits199 wr, wi;\n"
    "                    wr.u = weight << 16;\n"
    "                    wi.u = weight & 0xffff0000u;\n"
    "                    yr = yr * wr.f;\n"
    "                    yi = yi * wi.f;\n"
    "                }\n"
    "                if (MODE == 3) {\n"
    "                    f32v c = *((f32v*)(fc + pair - 256));\n"
    "                    f32v s = *((f32v*)(fs + pair - 256));\n"
    "                    f32v rr = yr * c - yi * s;\n"
    "                    f32v ri = yr * s + yi * c;\n"
    "                    yr = rr;\n"
    "                    yi = ri;\n"
    "                }\n"
    "                *((u32v*)(xb + pair)) = encode199(yr) | (encode19"
    "9(yi) << 16);\n"
    "            }\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(O + row * 576, xb, 0, 1152);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW199, flags=("-ffp-contract=off",))
def _device_row199(X, W, F, Out, M, EPS, HW, MODE, pid, np):
    pass


@triton.jit
def _entry_row199(X, W, F, Out, META: tl.constexpr):
    tle.raw.call(
        _device_row199,
        (
            X,
            W,
            F,
            Out,
            tl.full((), META[0], tl.int32),
            tl.full((), META[1], tl.float32),
            tl.full((), META[2], tl.int32),
            tl.full((), META[3], tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


def _raw_launch199(kv, weight, eps, freqs_cis, positions, mode=3):
    m, d = kv.shape
    out = torch.empty((m, d), dtype=kv.dtype, device=kv.device)
    f = torch.view_as_real(freqs_cis)
    grid = min(12, (m + 63) // 64)
    kernel = _entry_pipeline199 if m > grid * 64 else _entry_row199
    kernel.run(
        kv,
        weight if weight is not None else kv,
        f,
        out,
        (m, eps, weight is not None, mode),
        grid=(grid,),
        warmup=False,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out


def _raw_supported199(kv, weight, freqs_cis, positions):
    m, d = kv.shape
    supported = (
        m >= 512
        and m * d <= 2147483647
        and (d == 576)
        and (kv.dtype == torch.bfloat16)
        and kv.is_contiguous()
        and (kv.data_ptr() % 64 == 0)
    )
    supported = (
        supported
        and positions is None
        and (freqs_cis.shape == (m, 32))
        and (freqs_cis.dtype == torch.complex64)
        and freqs_cis.is_contiguous()
        and (freqs_cis.data_ptr() % 64 == 0)
    )
    if weight is not None:
        supported = (
            supported
            and weight.dtype == torch.bfloat16
            and (weight.shape == (d,))
            and weight.is_contiguous()
            and (weight.data_ptr() % 64 == 0)
        )
    return supported


RAW_PIPELINE199 = (
    "typedef unsigned short __attribute__((address_space(1))) gu16;\n"
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "union bits199 { f32v f; u32v u; };\n"
    "namespace xpu { __attribute__((device)) float rsqrtf(float); }\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode199(f32v value) "
    "{\n"
    "    bits199 bits;\n"
    "    bits.f = value;\n"
    "    u32v original = bits.u;\n"
    "    u32v word = (original + 32768u) >> 16;\n"
    "    return word;\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _device_pipeline199(\n'
    "    gu16* X, gu16* W, gf32* F, gu16* O,\n"
    "    int M, float EPS, int HW, int MODE, int pid, int np) {\n"
    "    unsigned int storage_x[576] __attribute__((aligned(64)));\n"
    "    unsigned int output[288] __attribute__((aligned(64)));\n"
    "    unsigned int wb[288] __attribute__((aligned(64)));\n"
    "    float storage_f[128] __attribute__((aligned(64)));\n"
    "    float fc[32] __attribute__((aligned(64)));\n"
    "    float fs[32] __attribute__((aligned(64)));\n"
    "    float sums[16] __attribute__((aligned(64)));\n"
    "    const int core = __builtin_xpu_core_id();\n"
    "    int row = pid * 64 + core;\n"
    "    if (row >= M) return;\n"
    "    if (HW && MODE >= 2) {\n"
    "        __builtin_xpu_gm2lm_v3(wb, W, 0, 1152);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "    int current = 0;\n"
    "    const int step = np * 64;\n"
    "    __builtin_xpu_gm2lm_v3(storage_x, X + row * 576, 0, 1152);\n"
    "    if (MODE == 3)\n"
    "        __builtin_xpu_gm2lm_v3(storage_f, F + row * 64, 0, 256);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    for (; row < M; row += step) {\n"
    "        unsigned int* xb = storage_x + current * 288;\n"
    "        float* fb = storage_f + current * 64;\n"
    "        const int next = row + step;\n"
    "        if (next < M) {\n"
    "            __builtin_xpu_gm2lm_v3(storage_x + (1 - current) * 28"
    "8,\n"
    "                                  X + next * 576, 0, 1152);\n"
    "            if (MODE == 3)\n"
    "                __builtin_xpu_gm2lm_v3(storage_f + (1 - current) "
    "* 64,\n"
    "                                      F + next * 64, 0, 256);\n"
    "        }\n"
    "        if (MODE != 0) {\n"
    "            f32v acc = (f32v)(0.0f);\n"
    "            for (int pair = 0; pair < 288; pair += 16) {\n"
    "                u32v packed = *((u32v*)(xb + pair));\n"
    "                bits199 real, imag;\n"
    "                real.u = packed << 16;\n"
    "                imag.u = packed & 0xffff0000u;\n"
    "                acc = acc + (real.f * real.f + imag.f * imag.f);\n"
    "            }\n"
    "            *((f32v*)sums) = acc;\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            const float total = ((((sums[0] + sums[1]) + (sums[2]"
    " + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]))) + ((("
    "sums[8] + sums[9]) + (sums[10] + sums[11])) + ((sums[12] + sums[1"
    "3]) + (sums[14] + sums[15]))));\n"
    "            const float inv = xpu::rsqrtf(total / 576.0f + EPS);\n"
    "            if (MODE == 3) {\n"
    "                #pragma clang loop vectorize(disable)\n"
    "                for (int pair = 0; pair < 32; ++pair) {\n"
    "                    fc[pair] = fb[2 * pair];\n"
    "                    fs[pair] = fb[2 * pair + 1];\n"
    "                }\n"
    "            }\n"
    "            for (int pair = 0; pair < 256; pair += 16) {\n"
    "                u32v packed = *((u32v*)(xb + pair));\n"
    "                bits199 real, imag;\n"
    "                real.u = packed << 16;\n"
    "                imag.u = packed & 0xffff0000u;\n"
    "                f32v yr = real.f * inv;\n"
    "                f32v yi = imag.f * inv;\n"
    "                if (HW && MODE >= 2) {\n"
    "                    u32v weight = *((u32v*)(wb + pair));\n"
    "                    bits199 wr, wi;\n"
    "                    wr.u = weight << 16;\n"
    "                    wi.u = weight & 0xffff0000u;\n"
    "                    yr = yr * wr.f;\n"
    "                    yi = yi * wi.f;\n"
    "                }\n"
    "                *((u32v*)(output + pair)) = encode199(yr) | (enco"
    "de199(yi) << 16);\n"
    "            }\n"
    "            for (int pair = 256; pair < 288; pair += 16) {\n"
    "                u32v packed = *((u32v*)(xb + pair));\n"
    "                bits199 real, imag;\n"
    "                real.u = packed << 16;\n"
    "                imag.u = packed & 0xffff0000u;\n"
    "                f32v yr = real.f * inv;\n"
    "                f32v yi = imag.f * inv;\n"
    "                if (HW && MODE >= 2) {\n"
    "                    u32v weight = *((u32v*)(wb + pair));\n"
    "                    bits199 wr, wi;\n"
    "                    wr.u = weight << 16;\n"
    "                    wi.u = weight & 0xffff0000u;\n"
    "                    yr = yr * wr.f;\n"
    "                    yi = yi * wi.f;\n"
    "                }\n"
    "                if (MODE == 3) {\n"
    "                    f32v c = *((f32v*)(fc + pair - 256));\n"
    "                    f32v s = *((f32v*)(fs + pair - 256));\n"
    "                    f32v rr = yr * c - yi * s;\n"
    "                    f32v ri = yr * s + yi * c;\n"
    "                    yr = rr;\n"
    "                    yi = ri;\n"
    "                }\n"
    "                *((u32v*)(output + pair)) = encode199(yr) | (enco"
    "de199(yi) << 16);\n"
    "            }\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(O + row * 576, MODE == 0 ? xb : ou"
    "tput, 0, 1152);\n"
    "        if (MODE == 0)\n"
    "            __builtin_xpu2_mfence(7);\n"
    "        current = 1 - current;\n"
    "    }\n"
    "    if (MODE != 0)\n"
    "        __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW_PIPELINE199, flags=("-ffp-contract=off",))
def _device_pipeline199(X, W, F, Out, M, EPS, HW, MODE, pid, np):
    pass


@triton.jit
def _entry_pipeline199(X, W, F, Out, META: tl.constexpr):
    tle.raw.call(
        _device_pipeline199,
        (
            X,
            W,
            F,
            Out,
            tl.full((), META[0], tl.int32),
            tl.full((), META[1], tl.float32),
            tl.full((), META[2], tl.int32),
            tl.full((), META[3], tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


def fused_norm_rope_inplace(kv, weight, eps, freqs_cis, positions):
    if _raw_supported199(kv, weight, freqs_cis, positions):
        return _raw_launch199(kv, weight, eps, freqs_cis, positions)
    return _recommended199(kv, weight, eps, freqs_cis, positions)


__all__ = ["fused_norm_rope_inplace"]
