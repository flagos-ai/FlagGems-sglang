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
import triton as tr
import triton.language as tl


@tr.jit
def _zero_span(P, LENGTH: tl.constexpr):
    for offset in tl.static_range(0, LENGTH // 32):
        tl.store(
            P[:, None] + offset * 32 + tl.arange(0, 32)[None, :],
            tl.full((64, 32), 0, tl.uint32),
        )
    if LENGTH % 32 >= 16:
        tl.store(
            P[:, None] + LENGTH // 32 * 32 + tl.arange(0, 16)[None, :],
            tl.full((64, 16), 0, tl.uint32),
        )
    if LENGTH % 16 >= 8:
        tl.store(
            P[:, None] + LENGTH // 16 * 16 + tl.arange(0, 8)[None, :],
            tl.full((64, 8), 0, tl.uint32),
        )
    if LENGTH % 8 >= 4:
        tl.store(
            P[:, None] + LENGTH // 8 * 8 + tl.arange(0, 4)[None, :],
            tl.full((64, 4), 0, tl.uint32),
        )
    if LENGTH % 4 >= 2:
        tl.store(
            P[:, None] + LENGTH // 4 * 4 + tl.arange(0, 2)[None, :],
            tl.full((64, 2), 0, tl.uint32),
        )
    if LENGTH % 2 >= 1:
        tl.store(
            P[:, None] + LENGTH // 2 * 2 + tl.arange(0, 1)[None, :],
            tl.full((64, 1), 0, tl.uint32),
        )


@tr.jit
def _copy_plane_zero(
    C,
    X,
    Y,
    RC: tl.constexpr,
    RX: tl.constexpr,
    CH: tl.constexpr,
    TCN: tl.constexpr,
    TN: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OT: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    WL: tl.constexpr,
    HT: tl.constexpr,
    DL: tl.constexpr,
    BR: tl.constexpr,
    BW: tl.constexpr,
    ZBR: tl.constexpr,
):
    NC: tl.constexpr = tr.cdiv(RC, BR)
    p = tl.program_id(0)
    w = tl.program_id(1) * BW + tl.arange(0, BW)
    if p < NC:
        row = p * BR + tl.arange(0, BR)
        h = row % H
        t = row // H % TCN
        c = row // (H * TCN) % CH
        b = row // (H * TCN * CH)
        dst = (((b * CH + c) * OT + t + DL) * OH + h + HT) * OW + WL
        src = row[:, None] * W + w[None, :]
        tl.store(Y + dst[:, None] + w[None, :], tl.load(C + src))
    elif p < NC + tr.cdiv(RX, BR):
        row = (p - NC) * BR + tl.arange(0, BR)
        h = row % H
        t = row // H % TN
        c = row // (H * TN) % CH
        b = row // (H * TN * CH)
        dst = (((b * CH + c) * OT + t + DL + TCN) * OH + h + HT) * OW + WL
        src = row[:, None] * W + w[None, :]
        tl.store(Y + dst[:, None] + w[None, :], tl.load(X + src))
    elif p < NC + tr.cdiv(RX, BR) + (RC + RX) // BR:
        side_row = (p - NC - tr.cdiv(RX, BR)) * BR + tl.arange(0, BR)
        plane = side_row // H
        out_plane = plane // (TCN + TN) * OT + plane % (TCN + TN) + DL
        row_start = (out_plane * OH + side_row % H + HT) * OW
        if WL > 0:
            tl.store(
                Y + row_start[:, None] + tl.arange(0, WL)[None, :],
                tl.full((BR, WL), 0, Y.dtype.element_ty),
            )
        if OW - W - WL > 0:
            RIGHT: tl.constexpr = OW - W - WL
            tl.store(
                Y
                + (row_start + W + WL)[:, None]
                + tl.arange(0, RIGHT)[None, :],
                tl.full((BR, RIGHT), 0, Y.dtype.element_ty),
            )
    elif p - (RC + RX) // BR < NC + tr.cdiv(RX, BR) + (RC + RX) // H // 64:
        EB: tl.constexpr = tr.cdiv(Y.dtype.element_ty.primitive_bitwidth, 8)
        RW: tl.constexpr = OW * EB // 4
        ZY = Y.to(tl.pointer_type(tl.uint32))
        plane = (p - (RC + RX) // BR - NC - tr.cdiv(RX, BR)) * 64 + tl.arange(
            0, 64
        )
        out_plane = plane // (TCN + TN) * OT + plane % (TCN + TN) + DL
        start = ZY + out_plane * OH * RW
        _zero_span(start, HT * RW)
        _zero_span(start + (H + HT) * RW, HT * RW)
    elif OT > TCN + TN:
        EB: tl.constexpr = tr.cdiv(Y.dtype.element_ty.primitive_bitwidth, 8)
        RW: tl.constexpr = OW * EB // 4
        CAPS: tl.constexpr = (RC + RX) // H // 64
        ZY = Y.to(tl.pointer_type(tl.uint32))
        plane = (
            p - (RC + RX) // BR - NC - tr.cdiv(RX, BR) - CAPS
        ) * 64 + tl.arange(0, 64)
        frame = plane % (OT - TCN - TN)
        out_plane = (
            plane // (OT - TCN - TN) * OT
            + frame
            + tl.where(frame >= DL, TCN + TN, 0)
        )
        _zero_span(ZY + out_plane * OH * RW, OH * RW)


@tr.jit
def _copy_owned_halo(
    C,
    X,
    Y,
    RC: tl.constexpr,
    RX: tl.constexpr,
    CH: tl.constexpr,
    TCN: tl.constexpr,
    TN: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OT: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    WL: tl.constexpr,
    HT: tl.constexpr,
    DL: tl.constexpr,
    BR: tl.constexpr,
    BW: tl.constexpr,
    ZBR: tl.constexpr,
):
    NC: tl.constexpr = tr.cdiv(RC, BR)
    p = tl.program_id(0)
    w = tl.program_id(1) * BW + tl.arange(0, BW)
    if p < NC:
        row = p * BR + tl.arange(0, BR)
        h = row % H
        t = row // H % TCN
        c = row // (H * TCN) % CH
        b = row // (H * TCN * CH)
        dst = (((b * CH + c) * OT + t + DL) * OH + h + HT) * OW + WL
        src = row[:, None] * W + w[None, :]
        tl.store(Y + dst[:, None] + w[None, :], tl.load(C + src))
    elif p < NC + tr.cdiv(RX, BR):
        row = (p - NC) * BR + tl.arange(0, BR)
        h = row % H
        t = row // H % TN
        c = row // (H * TN) % CH
        b = row // (H * TN * CH)
        dst = (((b * CH + c) * OT + t + DL + TCN) * OH + h + HT) * OW + WL
        src = row[:, None] * W + w[None, :]
        tl.store(Y + dst[:, None] + w[None, :], tl.load(X + src))
    elif p < NC + tr.cdiv(RX, BR) + (RC + RX) // BR:
        side_row = (p - NC - tr.cdiv(RX, BR)) * BR + tl.arange(0, BR)
        plane = side_row // H
        out_plane = plane // (TCN + TN) * OT + plane % (TCN + TN) + DL
        row_start = (out_plane * OH + side_row % H + HT) * OW
        if WL > 0:
            tl.store(
                Y + row_start[:, None] + tl.arange(0, WL)[None, :],
                tl.full((BR, WL), 0, Y.dtype.element_ty),
            )
        if OW - W - WL > 0:
            RIGHT: tl.constexpr = OW - W - WL
            tl.store(
                Y
                + (row_start + W + WL)[:, None]
                + tl.arange(0, RIGHT)[None, :],
                tl.full((BR, RIGHT), 0, Y.dtype.element_ty),
            )
    elif (
        p - (RC + RX) // BR
        < NC
        + tr.cdiv(RX, BR)
        + (RC + RX)
        // H
        * 2
        * HT
        * OW
        * tr.cdiv(Y.dtype.element_ty.primitive_bitwidth, 8)
        // 4
        // 64
    ):
        EB: tl.constexpr = tr.cdiv(Y.dtype.element_ty.primitive_bitwidth, 8)
        RW: tl.constexpr = OW * EB // 4
        ZY = Y.to(tl.pointer_type(tl.uint32))
        zr = (p - (RC + RX) // BR - NC - tr.cdiv(RX, BR)) * 64 + tl.arange(
            0, 64
        )
        plane = zr // (2 * HT * RW)
        inside = zr % (2 * HT * RW)
        out_plane = plane // (TCN + TN) * OT + plane % (TCN + TN) + DL
        offset = (
            out_plane * OH * RW
            + inside
            + tl.where(inside >= HT * RW, H * RW, 0)
        )
        tl.store(
            ZY + offset[:, None] + tl.arange(0, 1)[None, :],
            tl.full((64, 1), 0, tl.uint32),
        )
    elif OT > TCN + TN:
        EB: tl.constexpr = tr.cdiv(Y.dtype.element_ty.primitive_bitwidth, 8)
        RW: tl.constexpr = OW * EB // 4
        CAPS: tl.constexpr = (RC + RX) // H * 2 * HT * RW // 64
        ZY = Y.to(tl.pointer_type(tl.uint32))
        zr = (
            p - (RC + RX) // BR - NC - tr.cdiv(RX, BR) - CAPS
        ) * 64 + tl.arange(0, 64)
        plane = zr // (OH * RW)
        inside = zr % (OH * RW)
        frame = plane % (OT - TCN - TN)
        out_plane = (
            plane // (OT - TCN - TN) * OT
            + frame
            + tl.where(frame >= DL, TCN + TN, 0)
        )
        offset = out_plane * OH * RW + inside
        tl.store(
            ZY + offset[:, None] + tl.arange(0, 1)[None, :],
            tl.full((64, 1), 0, tl.uint32),
        )


@tr.jit
def _scalar(X, C, Y, G: tl.constexpr, XS: tl.constexpr, CS: tl.constexpr):
    CH: tl.constexpr = G[0]
    T: tl.constexpr = G[1]
    TC: tl.constexpr = G[2]
    H: tl.constexpr = G[3]
    W: tl.constexpr = G[4]
    OT: tl.constexpr = G[5]
    OH: tl.constexpr = G[6]
    OW: tl.constexpr = G[7]
    WL: tl.constexpr = G[8]
    HT: tl.constexpr = G[9]
    DL: tl.constexpr = G[10]
    N: tl.constexpr = G[11]
    for i in range(tl.program_id(0), N, tl.num_programs(0)):
        w = i % OW - WL
        h = i // OW % OH - HT
        t = i // (OW * OH) % OT - DL
        c = i // (OW * OH * OT) % CH
        b = i // (OW * OH * OT * CH)
        if (
            w >= 0
            and w < W
            and (h >= 0)
            and (h < H)
            and (t >= 0)
            and (t < T + TC)
        ):
            if t < TC:
                tl.store(
                    Y + i,
                    tl.load(
                        C
                        + b * CS[0]
                        + c * CS[1]
                        + t * CS[2]
                        + h * CS[3]
                        + w * CS[4]
                    ),
                )
            else:
                tl.store(
                    Y + i,
                    tl.load(
                        X
                        + b * XS[0]
                        + c * XS[1]
                        + (t - TC) * XS[2]
                        + h * XS[3]
                        + w * XS[4]
                    ),
                )
        else:
            tl.store(Y + i, 0)


def causal_conv3d_cat_pad(x, cache_x, padding):
    b, c, t, h, w = x.shape
    tc = cache_x.shape[2]
    wl, wr, ht, hb, front, back = padding
    ot, oh, ow = (t + front + back, h + ht + hb, w + wl + wr)
    dtype = (
        x.dtype
        if x.dtype == cache_x.dtype
        else torch.promote_types(x.dtype, cache_x.dtype)
    )
    out = torch.empty((b, c, ot, oh, ow), dtype=dtype, device=x.device)
    n = out.numel()
    if not n:
        return out
    eb = out.element_size()
    cap_bytes = b * c * (t + tc) * (ht + hb) * ow * eb
    time_bytes = b * c * (front - tc + back) * oh * ow * eb
    if (
        x.is_contiguous()
        and cache_x.is_contiguous()
        and (x.storage_offset() % 2 == 0)
        and (cache_x.storage_offset() % 2 == 0)
        and (x.dtype == cache_x.dtype)
        and (h > 0)
        and (h <= 64)
        and (64 % h == 0)
        and (w > 0)
        and (w <= 256)
        and (w & w - 1 == 0)
        and (t > 0)
        and (tc > 0)
        and (wl == wr)
        and (ht == hb)
        and (wl >= 0)
        and (ht > 0)
        and (front >= tc)
        and (back >= 0)
        and (wl <= w)
        and (ht <= h)
        and (wl == 0 or wl & wl - 1 == 0)
        and (b * c * tc * h % 64 == 0)
        and (b * c * t * h % 64 == 0)
        and (ow * eb % 4 == 0)
        and (cap_bytes % 256 == 0)
        and (time_bytes % 256 == 0)
    ):
        rc = b * c * tc * h
        rx = b * c * t * h
        if (
            b * c * (t + tc) % 64 == 0
            and b * c * (front - tc + back) % 64 == 0
        ):
            blocks = (rc + rx) // 64 + b * c * ot // 64
            _copy_plane_zero[blocks + (rc + rx) // 64, 1](
                cache_x,
                x,
                out,
                rc,
                rx,
                c,
                tc,
                t,
                h,
                w,
                ot,
                oh,
                ow,
                wl,
                ht,
                front - tc,
                64,
                w,
                64,
                groups_per_cluster=64,
            )
            return out
        _copy_owned_halo[
            (rc + rx) // 64
            + cap_bytes // 256
            + time_bytes // 256
            + (rc + rx) // 64,
            1,
        ](
            cache_x,
            x,
            out,
            rc,
            rx,
            c,
            tc,
            t,
            h,
            w,
            ot,
            oh,
            ow,
            wl,
            ht,
            front - tc,
            64,
            w,
            64,
            groups_per_cluster=64,
        )
        return out
    _scalar[min(n, 64),](
        x,
        cache_x,
        out,
        (c, t, tc, h, w, ot, oh, ow, wl, ht, front - tc, n),
        x.stride(),
        cache_x.stride(),
    )
    return out


__all__ = ["causal_conv3d_cat_pad"]
