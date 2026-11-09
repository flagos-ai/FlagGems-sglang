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
def _pad(
    X,
    C,
    Y,
    CH: tl.constexpr,
    T: tl.constexpr,
    TC: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OT: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    WL: tl.constexpr,
    HT: tl.constexpr,
    DL: tl.constexpr,
    XS: tl.constexpr,
    CS: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
):
    NH: tl.constexpr = tr.cdiv(OH, BH)
    NW: tl.constexpr = tr.cdiv(OW, BW)
    pid = tl.program_id(0)
    k = pid // NW
    plane = k // NH
    h = k % NH * BH + tl.arange(0, BH)
    w = pid % NW * BW + tl.arange(0, BW)
    t = plane % OT
    c = plane // OT % CH
    b = plane // (OT * CH)
    keep = (h < OH)[:, None] & (w < OW)[None, :]
    dst = Y + (plane * OH + h)[:, None] * OW + w[None, :]
    sh = h - HT
    sw = w - WL
    inner = (
        keep
        & ((sh >= 0) & (sh < H))[:, None]
        & ((sw >= 0) & (sw < W))[None, :]
    )
    if t < DL or t >= DL + TC + T:
        tl.store(dst, tl.zeros([BH, BW], dtype=Y.dtype.element_ty), keep)
    elif t < DL + TC:
        v_cache = tl.load(
            C
            + b * CS[0]
            + c * CS[1]
            + (t - DL) * CS[2]
            + sh[:, None] * CS[3]
            + sw[None, :] * CS[4],
            inner,
            other=0,
        )
        tl.store(dst, v_cache, keep)
    else:
        v_x = tl.load(
            X
            + b * XS[0]
            + c * XS[1]
            + (t - DL - TC) * XS[2]
            + sh[:, None] * XS[3]
            + sw[None, :] * XS[4],
            inner,
            other=0,
        )
        tl.store(dst, v_x, keep)


@tr.jit
def _padn(X, C, Y, G: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr):
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
    NH: tl.constexpr = tr.cdiv(OH, BH)
    NW: tl.constexpr = tr.cdiv(OW, BW)
    pid = tl.program_id(0)
    k = pid // NW
    plane = k // NH
    h = k % NH * BH + tl.arange(0, BH)
    w = pid % NW * BW + tl.arange(0, BW)
    t = plane % OT
    bc = plane // OT
    keep = (h < OH)[:, None] & (w < OW)[None, :]
    dst = Y + (plane * OH + h)[:, None] * OW + w[None, :]
    sh = h - HT
    sw = w - WL
    inner = (
        keep
        & ((sh >= 0) & (sh < H))[:, None]
        & ((sw >= 0) & (sw < W))[None, :]
    )
    if t < DL or t >= DL + TC + T:
        tl.store(dst, tl.zeros([BH, BW], dtype=Y.dtype.element_ty), keep)
    elif t < DL + TC:
        src = (bc * TC + t - DL) * H * W + sh[:, None] * W + sw[None, :]
        tl.store(dst, tl.load(C + src, inner, other=0), keep)
    else:
        src = (bc * T + t - DL - TC) * H * W + sh[:, None] * W + sw[None, :]
        tl.store(dst, tl.load(X + src, inner, other=0), keep)


@tr.jit(do_not_specialize=["X", "C", "Y"])
def _padn7(X, C, Y, G: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr):
    T: tl.constexpr = G[0]
    TC: tl.constexpr = G[1]
    H: tl.constexpr = G[2]
    W: tl.constexpr = G[3]
    OT: tl.constexpr = G[4]
    OH: tl.constexpr = G[5]
    OW: tl.constexpr = G[6]
    WL: tl.constexpr = (OW - W) // 2
    HT: tl.constexpr = (OH - H) // 2
    DL: tl.constexpr = OT - T - TC
    NH: tl.constexpr = tr.cdiv(OH, BH)
    NW: tl.constexpr = tr.cdiv(OW, BW)
    WORK: tl.constexpr = G[7]
    if DL > 0:
        for pid in range(tl.program_id(0), WORK * DL, tl.num_programs(0)):
            k = pid // NW
            p = k // NH
            bc = p // DL
            t = p % DL + 0
            h = k % NH * BH + tl.arange(0, BH)
            w = pid % NW * BW + tl.arange(0, BW)
            plane = bc * OT + t
            keep = (h < OH)[:, None] & (w < OW)[None, :]
            dst = Y + (plane * OH + h)[:, None] * OW + w[None, :]
            sh = h - HT
            sw = w - WL
            inner = (
                keep
                & ((sh >= 0) & (sh < H))[:, None]
                & ((sw >= 0) & (sw < W))[None, :]
            )
            tl.store(dst, tl.zeros([BH, BW], dtype=Y.dtype.element_ty), keep)
    if TC > 0:
        for pid in range(tl.program_id(0), WORK * TC, tl.num_programs(0)):
            k = pid // NW
            p = k // NH
            bc = p // TC
            t = p % TC + DL
            h = k % NH * BH + tl.arange(0, BH)
            w = pid % NW * BW + tl.arange(0, BW)
            plane = bc * OT + t
            keep = (h < OH)[:, None] & (w < OW)[None, :]
            dst = Y + (plane * OH + h)[:, None] * OW + w[None, :]
            sh = h - HT
            sw = w - WL
            inner = (
                keep
                & ((sh >= 0) & (sh < H))[:, None]
                & ((sw >= 0) & (sw < W))[None, :]
            )
            src = (bc * TC + t - DL) * H * W + sh[:, None] * W + sw[None, :]
            tl.store(dst, tl.load(C + src, inner, other=0), keep)
    if T > 0:
        for pid in range(tl.program_id(0), WORK * T, tl.num_programs(0)):
            k = pid // NW
            p = k // NH
            bc = p // T
            t = p % T + (DL + TC)
            h = k % NH * BH + tl.arange(0, BH)
            w = pid % NW * BW + tl.arange(0, BW)
            plane = bc * OT + t
            keep = (h < OH)[:, None] & (w < OW)[None, :]
            dst = Y + (plane * OH + h)[:, None] * OW + w[None, :]
            sh = h - HT
            sw = w - WL
            inner = (
                keep
                & ((sh >= 0) & (sh < H))[:, None]
                & ((sw >= 0) & (sw < W))[None, :]
            )
            src = (
                (bc * T + t - DL - TC) * H * W + sh[:, None] * W + sw[None, :]
            )
            tl.store(dst, tl.load(X + src, inner, other=0), keep)


@tr.jit(do_not_specialize=["X", "C", "Y"])
def _flat_gather(X, C, Y, G: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr):
    T: tl.constexpr = G[0]
    TC: tl.constexpr = G[1]
    H: tl.constexpr = G[2]
    W: tl.constexpr = G[3]
    OT: tl.constexpr = G[4]
    OH: tl.constexpr = G[5]
    OW: tl.constexpr = G[6]
    BC: tl.constexpr = G[7]
    WL: tl.constexpr = (OW - W) // 2
    HT: tl.constexpr = (OH - H) // 2
    DL: tl.constexpr = OT - T - TC
    SI: tl.constexpr = tr.next_power_of_2(H * W)
    SO: tl.constexpr = tr.next_power_of_2(OH * OW)
    ix = tl.arange(0, SI)
    iy = tl.arange(0, SO)
    estimate = (iy.to(tl.float32) * (1.0 / OW)).to(tl.int32)
    remainder = iy - estimate * OW
    row = (
        estimate
        + (remainder >= OW).to(tl.int32)
        - (remainder < 0).to(tl.int32)
    )
    h = row.to(tl.int32) - HT
    w = iy - row * OW - WL
    keep = iy < OH * OW
    valid = (h >= 0) & (h < H) & (w >= 0) & (w < W)
    gather_index = h * W + w & SI - 1
    if DL > 0:
        for p in range(tl.program_id(0), BC * DL, tl.num_programs(0)):
            plane = p // DL * OT + p % DL
            tl.store(Y + plane * OH * OW + iy, 0, keep)
    if TC > 0:
        for p in range(tl.program_id(0), BC * TC, tl.num_programs(0)):
            raw_c = tl.load(C + p * H * W + ix, ix < H * W, other=0)
            if raw_c.dtype == tl.bfloat16:
                typed = raw_c.to(tl.float16, bitcast=True)
                gathered = tl.gather(typed, gather_index, 0).to(
                    tl.bfloat16, bitcast=True
                )
                padded_c = tl.where(valid, gathered, 0)
            else:
                padded_c = tl.where(
                    valid, tl.gather(raw_c, gather_index, 0), 0
                )
            plane_c = p // TC * OT + p % TC + DL
            tl.store(Y + plane_c * OH * OW + iy, padded_c, keep)
    if T > 0:
        for p in range(tl.program_id(0), BC * T, tl.num_programs(0)):
            raw_x = tl.load(X + p * H * W + ix, ix < H * W, other=0)
            if raw_x.dtype == tl.bfloat16:
                typed = raw_x.to(tl.float16, bitcast=True)
                gathered = tl.gather(typed, gather_index, 0).to(
                    tl.bfloat16, bitcast=True
                )
                padded_x = tl.where(valid, gathered, 0)
            else:
                padded_x = tl.where(
                    valid, tl.gather(raw_x, gather_index, 0), 0
                )
            plane_x = p // T * OT + p % T + DL + TC
            tl.store(Y + plane_x * OH * OW + iy, padded_x, keep)


@tr.jit
def _zero_output(Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, 0, i < N)


def causal_conv3d_cat_pad(x, cache_x, padding):
    b, c, t, h, w = x.shape
    tc = cache_x.shape[2]
    wl, wr, ht, hb, front, back = padding
    ot, oh, ow = (t + front + back, h + ht + hb, w + wl + wr)
    if x.dtype == cache_x.dtype:
        out = x.new_empty((b, c, ot, oh, ow))
    else:
        out = x.new_empty(
            (b, c, ot, oh, ow),
            dtype=torch.promote_types(x.dtype, cache_x.dtype),
        )
    if b * c * ot * oh * ow == 0:
        return out
    if (
        h > 0
        and w > 0
        and (wl >= 0)
        and (ht >= 0)
        and (wl == wr)
        and (ht == hb)
        and (back == 0)
        and (front >= tc)
        and (ow <= 1024)
        and (oh * ow <= 8192)
    ):
        bw = max(16, 1 << (ow - 1).bit_length())
        bh = 1 << (oh - 1).bit_length()
        if bh * bw <= 8192 and x.is_contiguous() and cache_x.is_contiguous():
            bc = b * c
            _flat_gather.run(
                x,
                cache_x,
                out,
                (t, tc, h, w, ot, oh, ow, bc),
                bh,
                bw,
                grid=(min(40, bc),),
                warmup=False,
            )
            return out
    bw = max(16, min(1024, 1 << (max(1, ow) - 1).bit_length()))
    bh = max(1, min(1 << (max(1, oh) - 1).bit_length(), 8192 // bw))
    nprog = b * c * ot * ((oh + bh - 1) // bh) * ((ow + bw - 1) // bw)
    if not nprog:
        return out
    if h == 0 or w == 0:
        n = b * c * ot * oh * ow
        _zero_output.run(out, n, 1024, grid=(tr.cdiv(n, 1024),), warmup=False)
        return out
    if x.is_contiguous() and cache_x.is_contiguous():
        if wl == wr and ht == hb and (back == 0):
            if (
                oh <= bh
                and ow <= bw
                and (h > 0)
                and (w > 0)
                and (wl >= 0)
                and (ht >= 0)
                and (front >= tc)
                and (oh * ow <= 8192)
            ):
                _flat_gather.run(
                    x,
                    cache_x,
                    out,
                    (t, tc, h, w, ot, oh, ow, nprog // ot),
                    bh,
                    bw,
                    grid=(min(40, nprog // ot),),
                    warmup=False,
                )
            else:
                _padn7.run(
                    x,
                    cache_x,
                    out,
                    (t, tc, h, w, ot, oh, ow, nprog // ot),
                    bh,
                    bw,
                    grid=(min(40, nprog // ot),),
                    warmup=False,
                )
        else:
            _padn.run(
                x,
                cache_x,
                out,
                (c, t, tc, h, w, ot, oh, ow, wl, ht, front - tc),
                bh,
                bw,
                grid=(nprog,),
                warmup=False,
            )
        return out
    _pad.run(
        x,
        cache_x,
        out,
        c,
        t,
        tc,
        h,
        w,
        ot,
        oh,
        ow,
        wl,
        ht,
        front - tc,
        x.stride(),
        cache_x.stride(),
        bh,
        bw,
        grid=(nprog,),
        warmup=False,
    )
    return out


__all__ = ["causal_conv3d_cat_pad"]
