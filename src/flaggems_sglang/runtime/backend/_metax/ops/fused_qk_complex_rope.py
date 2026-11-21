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


@triton.jit
def _rope_qk(
    XQ,
    XK,
    OQ,
    OK,
    F,
    R,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN: tl.constexpr,
    RTZ: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    lane = tl.arange(0, BLOCK_D)
    tok = r // H
    if EVEN:
        off = r[:, None] * D + lane[None, :]
        qv = tl.load(XQ + off).to(tl.float32)
        kv = tl.load(XK + off).to(tl.float32)
        fv = tl.load(F + tok[:, None] * D + lane[None, :])
    else:
        m2 = (r < R)[:, None] & (lane < D)[None, :]
        off = r[:, None] * D + lane[None, :]
        qv = tl.load(XQ + off, mask=m2, other=0.0).to(tl.float32)
        kv = tl.load(XK + off, mask=m2, other=0.0).to(tl.float32)
        fv = tl.load(F + tok[:, None] * D + lane[None, :], mask=m2, other=0.0)
    qr, qi = tl.split(tl.reshape(qv, (BLOCK_R, BLOCK_D // 2, 2)))
    kr, ki = tl.split(tl.reshape(kv, (BLOCK_R, BLOCK_D // 2, 2)))
    fr_, fi_ = tl.split(tl.reshape(fv, (BLOCK_R, BLOCK_D // 2, 2)))
    oqr = qr * fr_ - qi * fi_
    oqi = qr * fi_ + qi * fr_
    okr = kr * fr_ - ki * fi_
    oki = kr * fi_ + ki * fr_
    qo = tl.reshape(tl.join(oqr, oqi), (BLOCK_R, BLOCK_D))
    ko = tl.reshape(tl.join(okr, oki), (BLOCK_R, BLOCK_D))
    if EVEN:
        if RTZ:
            tl.store(
                OQ + off,
                qo.to(OQ.dtype.element_ty, fp_downcast_rounding="rtz"),
            )
            tl.store(
                OK + off,
                ko.to(OK.dtype.element_ty, fp_downcast_rounding="rtz"),
            )
        else:
            tl.store(OQ + off, qo.to(OQ.dtype.element_ty))
            tl.store(OK + off, ko.to(OK.dtype.element_ty))
    else:
        m2 = (r < R)[:, None] & (lane < D)[None, :]
        if RTZ:
            tl.store(
                OQ + off,
                qo.to(OQ.dtype.element_ty, fp_downcast_rounding="rtz"),
                mask=m2,
            )
            tl.store(
                OK + off,
                ko.to(OK.dtype.element_ty, fp_downcast_rounding="rtz"),
                mask=m2,
            )
        else:
            tl.store(OQ + off, qo.to(OQ.dtype.element_ty), mask=m2)
            tl.store(OK + off, ko.to(OK.dtype.element_ty), mask=m2)


@triton.jit
def _rope_qk_tu(
    XQ,
    XK,
    OQ,
    OK,
    F,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RTZ: tl.constexpr,
):
    pid = tl.program_id(0)
    tok = (pid * BLOCK_R) // H
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    lane = tl.arange(0, BLOCK_D)
    off = r[:, None] * D + lane[None, :]
    qv = tl.load(XQ + off).to(tl.float32)
    kv = tl.load(XK + off).to(tl.float32)
    fv = tl.load(F + tok * D + lane)
    qr, qi = tl.split(tl.reshape(qv, (BLOCK_R, BLOCK_D // 2, 2)))
    kr, ki = tl.split(tl.reshape(kv, (BLOCK_R, BLOCK_D // 2, 2)))
    fr_, fi_ = tl.split(tl.reshape(fv, (1, BLOCK_D // 2, 2)))
    oqr = qr * fr_ - qi * fi_
    oqi = qr * fi_ + qi * fr_
    okr = kr * fr_ - ki * fi_
    oki = kr * fi_ + ki * fr_
    qo = tl.reshape(tl.join(oqr, oqi), (BLOCK_R, BLOCK_D))
    ko = tl.reshape(tl.join(okr, oki), (BLOCK_R, BLOCK_D))
    if RTZ:
        tl.store(
            OQ + off, qo.to(OQ.dtype.element_ty, fp_downcast_rounding="rtz")
        )
        tl.store(
            OK + off, ko.to(OK.dtype.element_ty, fp_downcast_rounding="rtz")
        )
    else:
        tl.store(OQ + off, qo.to(OQ.dtype.element_ty))
        tl.store(OK + off, ko.to(OK.dtype.element_ty))


@triton.jit
def _rope_qk_tuf(
    XQ,
    XK,
    OQ,
    OK,
    F,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RTZ: tl.constexpr,
):
    pid = tl.program_id(0)
    tok = (pid * BLOCK_R) // H
    NE: tl.constexpr = BLOCK_R * BLOCK_D
    base = pid * NE
    fl = tl.arange(0, NE)
    lane = tl.arange(0, BLOCK_D)
    qv = tl.reshape(tl.load(XQ + base + fl).to(tl.float32), (BLOCK_R, BLOCK_D))
    kv = tl.reshape(tl.load(XK + base + fl).to(tl.float32), (BLOCK_R, BLOCK_D))
    fv = tl.load(F + tok * D + lane)
    qr, qi = tl.split(tl.reshape(qv, (BLOCK_R, BLOCK_D // 2, 2)))
    kr, ki = tl.split(tl.reshape(kv, (BLOCK_R, BLOCK_D // 2, 2)))
    fr_, fi_ = tl.split(tl.reshape(fv, (1, BLOCK_D // 2, 2)))
    oqr = qr * fr_ - qi * fi_
    oqi = qr * fi_ + qi * fr_
    okr = kr * fr_ - ki * fi_
    oki = kr * fi_ + ki * fr_
    qo = tl.reshape(tl.join(oqr, oqi), (NE,))
    ko = tl.reshape(tl.join(okr, oki), (NE,))
    if RTZ:
        tl.store(
            OQ + base + fl,
            qo.to(OQ.dtype.element_ty, fp_downcast_rounding="rtz"),
        )
        tl.store(
            OK + base + fl,
            ko.to(OK.dtype.element_ty, fp_downcast_rounding="rtz"),
        )
    else:
        tl.store(OQ + base + fl, qo.to(OQ.dtype.element_ty))
        tl.store(OK + base + fl, ko.to(OK.dtype.element_ty))


@triton.jit
def _rope_qk_gen(
    XQ,
    XK,
    OQ,
    OK,
    F,
    R,
    XSZ: tl.constexpr,
    QST: tl.constexpr,
    KST: tl.constexpr,
    FSZ: tl.constexpr,
    FST: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RTZ: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = r < R
    lane = tl.arange(0, BLOCK_D)
    lmask = lane < D
    m2 = rmask[:, None] & lmask[None, :]
    lane64 = lane.to(tl.int64)

    NROW: tl.constexpr = len(XSZ)
    rem = r
    bq = tl.zeros((BLOCK_R,), dtype=tl.int64)
    bk = tl.zeros((BLOCK_R,), dtype=tl.int64)
    for j in tl.static_range(NROW):
        idx = rem % XSZ[j]
        rem = rem // XSZ[j]
        idx = idx.to(tl.int64)
        bq += idx * QST[j]
        bk += idx * KST[j]
    qoff = bq[:, None] + lane64[None, :] * QST[NROW]
    koff = bk[:, None] + lane64[None, :] * KST[NROW]
    qv = tl.load(XQ + qoff, mask=m2, other=0.0).to(tl.float32)
    kv = tl.load(XK + koff, mask=m2, other=0.0).to(tl.float32)

    NF: tl.constexpr = len(FSZ)
    remf = r // H
    bf = tl.zeros((BLOCK_R,), dtype=tl.int64)
    for j in tl.static_range(NF):
        idxf = remf % FSZ[j]
        remf = remf // FSZ[j]
        bf += idxf.to(tl.int64) * FST[j]
    i64 = (lane64 // 2)[None, :]
    foff = bf[:, None] + i64 * FST[NF] + (lane64 % 2)[None, :]
    fv = tl.load(F + foff, mask=m2, other=0.0)

    qr, qi = tl.split(tl.reshape(qv, (BLOCK_R, BLOCK_D // 2, 2)))
    kr, ki = tl.split(tl.reshape(kv, (BLOCK_R, BLOCK_D // 2, 2)))
    fr_, fi_ = tl.split(tl.reshape(fv, (BLOCK_R, BLOCK_D // 2, 2)))
    oqr = qr * fr_ - qi * fi_
    oqi = qr * fi_ + qi * fr_
    okr = kr * fr_ - ki * fi_
    oki = kr * fi_ + ki * fr_
    qo = tl.reshape(tl.join(oqr, oqi), (BLOCK_R, BLOCK_D))
    ko = tl.reshape(tl.join(okr, oki), (BLOCK_R, BLOCK_D))
    ooff = r.to(tl.int64)[:, None] * D + lane64[None, :]
    if RTZ:
        tl.store(
            OQ + ooff,
            qo.to(OQ.dtype.element_ty, fp_downcast_rounding="rtz"),
            mask=m2,
        )
        tl.store(
            OK + ooff,
            ko.to(OK.dtype.element_ty, fp_downcast_rounding="rtz"),
            mask=m2,
        )
    else:
        tl.store(OQ + ooff, qo.to(OQ.dtype.element_ty), mask=m2)
        tl.store(OK + ooff, ko.to(OK.dtype.element_ty), mask=m2)


def fused_qk_complex_rope(q, k, freqs_cis):
    shape = q.shape
    d = shape[-1]
    h = shape[-2]
    oq = torch.empty(shape, dtype=q.dtype, device=q.device)
    ok = torch.empty(shape, dtype=k.dtype, device=k.device)
    rows = q.numel() // d if d else 0
    if rows == 0 or d == 0:
        return oq, ok
    rtz = 1 if q.dtype == torch.bfloat16 else 0
    bd = 1 << (d - 1).bit_length()
    br = 4 if rows >= 4 else 1 << (rows - 1).bit_length()
    flat = 1024 < rows <= 16384 and rows % 16 == 0 and h % 16 == 0
    if 1024 < rows <= 131072 and rows % 8 == 0 and h % 8 == 0:
        br, nw = 8, 4
    else:
        nw = 4 if rows <= 1024 else 2
    if flat:
        br, nw = 16, 2

    if q.is_contiguous() and k.is_contiguous() and freqs_cis.is_contiguous():
        if flat and bd == d:
            grid = (rows // br,)
            _rope_qk_tuf[grid](
                q,
                k,
                oq,
                ok,
                freqs_cis.view(torch.float32),
                h,
                d,
                br,
                bd,
                rtz,
                num_warps=nw,
            )
            return oq, ok
        if bd == d and h % br == 0 and rows % br == 0:
            grid = (rows // br,)
            _rope_qk_tu[grid](
                q,
                k,
                oq,
                ok,
                freqs_cis.view(torch.float32),
                h,
                d,
                br,
                bd,
                rtz,
                num_warps=nw,
            )
            return oq, ok
        even = 1 if (bd == d and rows % br == 0) else 0
        grid = ((rows + br - 1) // br,)
        _rope_qk[grid](
            q,
            k,
            oq,
            ok,
            freqs_cis.view(torch.float32),
            rows,
            h,
            d,
            br,
            bd,
            even,
            rtz,
            num_warps=nw,
        )
        return oq, ok

    frr = torch.view_as_real(freqs_cis)
    grid = ((rows + br - 1) // br,)
    _rope_qk_gen[grid](
        q,
        k,
        oq,
        ok,
        frr,
        rows,
        tuple(reversed(shape[:-1])),
        tuple(reversed(q.stride()[:-1])) + (q.stride(-1),),
        tuple(reversed(k.stride()[:-1])) + (k.stride(-1),),
        tuple(reversed(freqs_cis.shape[:-1])),
        tuple(reversed(frr.stride()[:-2])) + (frr.stride(-2),),
        h,
        d,
        br,
        bd,
        rtz,
        num_warps=nw,
    )
    return oq, ok


__all__ = ["fused_qk_complex_rope"]
