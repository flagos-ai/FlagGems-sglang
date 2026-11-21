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
def _gather_freq(
    X,
    W,
    F,
    P,
    O,
    M: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    SX: tl.constexpr,
    SW: tl.constexpr,
    SF0: tl.constexpr,
    SF1: tl.constexpr,
    SP: tl.constexpr,
    EPS: tl.constexpr,
    HW: tl.constexpr,
    HP: tl.constexpr,
    B: tl.constexpr,
    BT: tl.constexpr,
    BR: tl.constexpr,
    WIDE: tl.constexpr,
    NF: tl.constexpr,
):
    for block in range(tl.program_id(0), tl.cdiv(M, BR), tl.num_programs(0)):
        rows = block * BR + tl.arange(0, BR)
        c = tl.arange(0, B)
        x = tl.load(
            X + _wide_mul(rows[:, None], SX, WIDE) + c[None, :],
            (rows[:, None] < M) & (c[None, :] < D),
            0,
        ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
        y = x * inv[:, None]
        if HW:
            w = tl.load(W + _wide_mul(c, SW, WIDE), c < D, 0).to(tl.float32)
            y *= w[None, :]
        if R > 0:
            j = tl.maximum(c - (D - R), 0)
            partner = tl.minimum(D - R + (j ^ 1), D - 1)
            yp = tl.gather(y, tl.broadcast_to(partner[None, :], (BR, B)), 1)
            pos = rows
            if HP:
                pos = tl.load(P + _wide_mul(rows, SP, WIDE), rows < M, 0)
                pos = tl.where(pos < 0, pos + NF, pos)
            tc = tl.arange(0, BT)
            ft = tl.load(
                F
                + _wide_mul(pos[:, None], SF0, WIDE)
                + _wide_mul(tc[None, :] // 2, SF1, WIDE)
                + tc[None, :] % 2,
                (rows[:, None] < M) & (tc[None, :] < R),
                0,
            )
            fi = tl.minimum(j // 2 * 2, R - 2)
            fr = tl.gather(ft, tl.broadcast_to(fi[None, :], (BR, B)), 1)
            fs = tl.gather(ft, tl.broadcast_to((fi + 1)[None, :], (BR, B)), 1)
            rotated = y * fr + tl.where(j[None, :] % 2 == 0, -yp, yp) * fs
            y = tl.where(c[None, :] >= D - R, rotated, y)
        tl.store(
            O + _wide_mul(rows[:, None], D, WIDE) + c[None, :],
            y,
            (rows[:, None] < M) & (c[None, :] < D),
        )


@triton.jit
def _segmented(
    X,
    W,
    F,
    P,
    O,
    M: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    SX: tl.constexpr,
    SW: tl.constexpr,
    SF0: tl.constexpr,
    SF1: tl.constexpr,
    SP: tl.constexpr,
    EPS: tl.constexpr,
    HW: tl.constexpr,
    HP: tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr,
    BR: tl.constexpr,
    WIDE: tl.constexpr,
    NF: tl.constexpr,
):
    for block in range(tl.program_id(0), tl.cdiv(M, BR), tl.num_programs(0)):
        rows = block * BR + tl.arange(0, BR)
        ss = tl.full((BR,), 0, tl.float32)
        if D > R:
            prefix = tl.arange(0, BS)
            xp = tl.load(
                X + _wide_mul(rows[:, None], SX, WIDE) + prefix[None, :],
                (rows[:, None] < M) & (prefix[None, :] < D - R),
                0,
            ).to(tl.float32)
            ss += tl.sum(xp * xp, 1)
        if R > 0:
            tail = tl.arange(0, BT)
            xt = tl.load(
                X + _wide_mul(rows[:, None], SX, WIDE) + D - R + tail[None, :],
                (rows[:, None] < M) & (tail[None, :] < R),
                0,
            ).to(tl.float32)
            ss += tl.sum(xt * xt, 1)
        inv = tl.rsqrt(ss / D + EPS)
        if D > R:
            yp = xp * inv[:, None]
            if HW:
                wp = tl.load(
                    W + _wide_mul(prefix, SW, WIDE), prefix < D - R, 0
                ).to(tl.float32)
                yp *= wp[None, :]
            tl.store(
                O + _wide_mul(rows[:, None], D, WIDE) + prefix[None, :],
                yp,
                (rows[:, None] < M) & (prefix[None, :] < D - R),
            )
        if R > 0:
            yt = xt * inv[:, None]
            if HW:
                wt = tl.load(
                    W + _wide_mul(D - R + tail, SW, WIDE), tail < R, 0
                ).to(tl.float32)
                yt *= wt[None, :]
            pair = tl.arange(0, BT // 2)
            yr = tl.gather(
                yt, tl.broadcast_to((2 * pair)[None, :], (BR, BT // 2)), 1
            )
            yi = tl.gather(
                yt, tl.broadcast_to((2 * pair + 1)[None, :], (BR, BT // 2)), 1
            )
            pos = rows
            if HP:
                pos = tl.load(P + _wide_mul(rows, SP, WIDE), rows < M, 0)
                pos = tl.where(pos < 0, pos + NF, pos)
            ft = tl.load(
                F
                + _wide_mul(pos[:, None], SF0, WIDE)
                + _wide_mul(tail[None, :] // 2, SF1, WIDE)
                + tail[None, :] % 2,
                (rows[:, None] < M) & (tail[None, :] < R),
                0,
            )
            fr = tl.gather(
                ft, tl.broadcast_to((2 * pair)[None, :], (BR, BT // 2)), 1
            )
            fi = tl.gather(
                ft, tl.broadcast_to((2 * pair + 1)[None, :], (BR, BT // 2)), 1
            )
            yr_out = yr * fr - yi * fi
            yi_out = yr * fi + yi * fr
            y = tl.reshape(tl.join(yr_out, yi_out), (BR, BT))
            tl.store(
                O + _wide_mul(rows[:, None], D, WIDE) + D - R + tail[None, :],
                y,
                (rows[:, None] < M) & (tail[None, :] < R),
            )


def _baseline(kv, weight, eps, freqs_cis, positions):
    WIDE = _requires_wide(kv, weight, freqs_cis, positions, False, False)
    m, d = kv.shape
    r = 2 * freqs_cis.shape[-1]
    f = torch.view_as_real(freqs_cis)
    out = torch.empty((m, d), dtype=kv.dtype, device=kv.device)
    if m == 0 or d == 0:
        return out
    br = 4
    args = (
        kv,
        weight,
        f,
        positions,
        out,
        m,
        d,
        r,
        kv.stride(0),
        weight.stride(0) if weight is not None else 0,
        f.stride(0),
        f.stride(1),
        positions.stride(0) if positions is not None else 0,
        eps,
        weight is not None,
        positions is not None,
    )
    if m <= 512:
        _gather_freq.run(
            *args,
            triton.next_power_of_2(d),
            triton.next_power_of_2(max(2, r)),
            br,
            WIDE,
            freqs_cis.shape[0],
            grid=(min(triton.cdiv(m, br), 32768),),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        br = (
            16
            if kv.dtype == torch.bfloat16
            and triton.next_power_of_2(max(1, d - r)) <= 512
            and (triton.next_power_of_2(max(2, r)) <= 64)
            else 4
        )
        _segmented.run(
            *args,
            triton.next_power_of_2(max(1, d - r)),
            triton.next_power_of_2(max(2, r)),
            br,
            WIDE,
            freqs_cis.shape[0],
            grid=(min(triton.cdiv(m, br), 32768),),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _encode_pair(v):
    return v.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535


@triton.jit
def _typed_rms(
    X: tl.pointer_type(tl.uint32),
    W: tl.pointer_type(tl.uint32),
    F: tl.pointer_type(tl.float32),
    P,
    O: tl.pointer_type(tl.uint32),
    META: tl.constexpr,
):
    M: tl.constexpr = META[0]
    SX: tl.constexpr = META[1]
    SF: tl.constexpr = META[2]
    SP: tl.constexpr = META[3]
    EPS: tl.constexpr = META[4]
    HW: tl.constexpr = META[5]
    HP: tl.constexpr = META[6]
    NF: tl.constexpr = META[7]
    BR: tl.constexpr = META[8]
    NB: tl.constexpr = triton.cdiv(M, BR)
    NP: tl.constexpr = min(NB, 40)
    for block in range(
        tl.program_id(0) * NB // NP, (tl.program_id(0) + 1) * NB // NP
    ):
        rows = block * BR + tl.arange(0, BR)
        src = rows
        p = tl.arange(0, 256)
        t = tl.arange(0, 32)
        pb = tl.load(X + src[:, None] * SX + p[None, :], rows[:, None] < M, 0)
        tb = tl.load(
            X + src[:, None] * SX + 256 + t[None, :], rows[:, None] < M, 0
        )
        pr = ((pb & 65535) << 16).to(tl.float32, bitcast=True)
        pi = (pb & 4294901760).to(tl.float32, bitcast=True)
        tr = ((tb & 65535) << 16).to(tl.float32, bitcast=True)
        ti = (tb & 4294901760).to(tl.float32, bitcast=True)
        ss = tl.sum(pr * pr + pi * pi, 1) + tl.sum(tr * tr + ti * ti, 1)
        inv = tl.rsqrt(ss / 576 + EPS)
        pr = pr * inv[:, None]
        pi = pi * inv[:, None]
        tr = tr * inv[:, None]
        ti = ti * inv[:, None]
        if HW:
            wp = tl.load(W + p)
            wt = tl.load(W + 256 + t)
            wpr = ((wp & 65535) << 16).to(tl.float32, bitcast=True)
            wpi = (wp & 4294901760).to(tl.float32, bitcast=True)
            wtr = ((wt & 65535) << 16).to(tl.float32, bitcast=True)
            wti = (wt & 4294901760).to(tl.float32, bitcast=True)
            pr = pr * wpr[None, :]
            pi = pi * wpi[None, :]
            tr = tr * wtr[None, :]
            ti = ti * wti[None, :]
        pos = src
        if HP:
            pos = tl.load(P + src * SP, rows < M, 0)
            pos = tl.where(pos < 0, pos + NF, pos)
        f = tl.load(
            F + pos[:, None] * SF + tl.arange(0, 64)[None, :],
            rows[:, None] < M,
            0,
        )
        fr, fi = tl.split(tl.reshape(f, (BR, 32, 2)))
        rr = tr * fr - ti * fi
        ri = ti * fr + tr * fi
        po = (
            _encode_pair(pr)
            | pi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
        )
        to = (
            _encode_pair(rr)
            | ri.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
        )
        tl.store(O + rows[:, None] * 288 + p[None, :], po, rows[:, None] < M)
        tl.store(
            O + rows[:, None] * 288 + 256 + t[None, :], to, rows[:, None] < M
        )


@triton.jit
def _typed_rms_p32(
    X: tl.pointer_type(tl.uint32),
    W: tl.pointer_type(tl.uint32),
    F: tl.pointer_type(tl.float32),
    P: tl.pointer_type(tl.int32),
    O: tl.pointer_type(tl.uint32),
    META: tl.constexpr,
):
    _typed_rms(X, W, F, P, O, META)


@triton.jit
def _typed_rms_p64(
    X: tl.pointer_type(tl.uint32),
    W: tl.pointer_type(tl.uint32),
    F: tl.pointer_type(tl.float32),
    P: tl.pointer_type(tl.int64),
    O: tl.pointer_type(tl.uint32),
    META: tl.constexpr,
):
    _typed_rms(X, W, F, P, O, META)


def fused_norm_rope_inplace(kv, weight, eps, freqs_cis, positions):
    m, d = kv.shape
    nf, half_r = freqs_cis.shape
    dtype = kv.dtype
    if not (
        0 < m <= 32768
        and d == 576
        and (half_r == 32)
        and (dtype == torch.bfloat16)
        and (freqs_cis.dtype == torch.complex64)
    ):
        return _baseline(kv, weight, eps, freqs_cis, positions)
    sx, sx1 = kv.stride()
    sf, sf1 = freqs_cis.stride()
    sp = positions.stride(0) if positions is not None else 0
    weight_ok = weight is None or (
        weight.dtype == torch.bfloat16
        and weight.stride(0) == 1
        and ((wptr := weight.data_ptr()) % 64 == 0)
    )
    wide = (
        max((m - 1) * sx + 575, 2 * ((nf - 1) * sf + 31) + 1, (m - 1) * sp)
        > 2147483647
    )
    if not (
        sx1 == 1
        and sx % 32 == 0
        and ((xptr := kv.data_ptr()) % 64 == 0)
        and (sf1 == 1)
        and (sf % 8 == 0)
        and ((fptr := freqs_cis.data_ptr()) % 64 == 0)
        and weight_ok
        and (not wide)
    ):
        return _baseline(kv, weight, eps, freqs_cis, positions)
    storage = kv.untyped_storage().data_ptr()
    can_inplace = (
        kv.is_contiguous()
        and (weight is None or weight.untyped_storage().data_ptr() != storage)
        and (
            freqs_cis is None
            or freqs_cis.untyped_storage().data_ptr() != storage
        )
        and (
            positions is None
            or positions.untyped_storage().data_ptr() != storage
        )
    )
    out = (
        kv
        if can_inplace
        else torch.empty_like(kv, memory_format=torch.contiguous_format)
    )
    br = 4 if m <= 512 else 16
    if positions is not None and positions.dtype == torch.int32:
        _typed_rms_p32.run(
            xptr,
            wptr if weight is not None else None,
            fptr,
            positions.data_ptr(),
            xptr if can_inplace else out.data_ptr(),
            (
                m,
                sx // 2,
                sf * 2,
                sp,
                eps,
                weight is not None,
                positions is not None,
                nf,
                br,
            ),
            grid=(min((m + br - 1) // br, 40),),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    elif positions is not None and positions.dtype == torch.int64:
        _typed_rms_p64.run(
            xptr,
            wptr if weight is not None else None,
            fptr,
            positions.data_ptr(),
            xptr if can_inplace else out.data_ptr(),
            (
                m,
                sx // 2,
                sf * 2,
                sp,
                eps,
                weight is not None,
                positions is not None,
                nf,
                br,
            ),
            grid=(min((m + br - 1) // br, 40),),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        _typed_rms.run(
            xptr,
            wptr if weight is not None else None,
            fptr,
            positions,
            xptr if can_inplace else out.data_ptr(),
            (
                m,
                sx // 2,
                sf * 2,
                sp,
                eps,
                weight is not None,
                positions is not None,
                nf,
                br,
            ),
            grid=(min((m + br - 1) // br, 40),),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["fused_norm_rope_inplace"]
