# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import triton
import triton.language as tl


def _loc_arg(loc):
    if loc.dtype == torch.int32:
        return loc
    if loc.dtype != torch.int64:
        return loc.to(torch.int32)
    if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64", "") in (
        "1",
        "true",
    ):
        return loc
    view = torch.empty(0, dtype=torch.int32, device=loc.device)
    view.set_(
        loc.untyped_storage(),
        loc.storage_offset(),
        loc.shape,
        loc.stride(),
    )
    return view


@triton.jit
def _src(i, SH: tl.constexpr, ST: tl.constexpr):
    row = i // (SH[1] * SH[2])
    h = (i // SH[2]) % SH[1]
    d = i % SH[2]
    return (
        row.to(tl.int64) * ST[0]
        + h.to(tl.int64) * ST[1]
        + d.to(tl.int64) * ST[2]
    )


@triton.jit
def _save(
    A,
    B,
    X,
    Y,
    NA: tl.constexpr,
    NB: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    AT: tl.constexpr,
    BT: tl.constexpr,
    W: tl.constexpr,
):
    i = tl.program_id(0) * W + tl.arange(0, W)
    if tl.program_id(0) * W < NA:
        a = tl.load(A + _src(i, AS, AT), i < NA, other=0)
        tl.store(X + i, a, i < NA)
    if tl.program_id(0) * W < NB:
        b = tl.load(B + _src(i, BS, BT), i < NB, other=0)
        tl.store(Y + i, b, i < NB)


@triton.jit
def _pair(
    K,
    V,
    A,
    B,
    L,
    P: tl.constexpr,
    NP: tl.constexpr,
    EK: tl.constexpr,
    EV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    KT: tl.constexpr,
    VT: tl.constexpr,
    AT: tl.constexpr,
    BT: tl.constexpr,
    LS: tl.constexpr,
    KC: tl.constexpr,
    VC: tl.constexpr,
    AC: tl.constexpr,
    BC: tl.constexpr,
    NB: tl.constexpr,
    W: tl.constexpr,
    COUNT: tl.constexpr,
    SERIAL: tl.constexpr,
    PACK: tl.constexpr,
):
    for j in range(COUNT if SERIAL else 1):
        work = j if SERIAL else tl.program_id(0)
        req = (work // NB).to(tl.int64)
        part = work % NB
        col = part * W + tl.arange(0, W)
        loc = tl.load(L + req * LS).to(tl.int64)
        page = loc // P
        token = loc % P
        page = tl.where(token < 0, page - 1, page)
        token = tl.where(token < 0, token + P, token)
        page = tl.where(page < 0, page + NP, page)
        if PACK:
            ai = req * (AT[0] // 4) + col.to(tl.int64)
            bi = req * (BT[0] // 4) + col.to(tl.int64)
            ki = page * (KT[0] // 4) + token * (KT[1] // 4) + col.to(tl.int64)
            vi = page * (VT[0] // 4) + token * (VT[1] // 4) + col.to(tl.int64)
        else:
            ac = (
                col.to(tl.int64)
                if AC
                else (col // DK).to(tl.int64) * AT[1]
                + (col % DK).to(tl.int64) * AT[2]
            )
            bc = (
                col.to(tl.int64)
                if BC
                else (col // DV).to(tl.int64) * BT[1]
                + (col % DV).to(tl.int64) * BT[2]
            )
            kc = (
                col.to(tl.int64)
                if KC
                else (col // DK).to(tl.int64) * KT[2]
                + (col % DK).to(tl.int64) * KT[3]
            )
            vc = (
                col.to(tl.int64)
                if VC
                else (col // DV).to(tl.int64) * VT[2]
                + (col % DV).to(tl.int64) * VT[3]
            )
            ai = req * AT[0] + ac
            bi = req * BT[0] + bc
            ki = page * KT[0] + token * KT[1] + kc
            vi = page * VT[0] + token * VT[1] + vc
        if EK == EV:
            if PACK:
                av = tl.load(
                    A.to(tl.pointer_type(tl.int64)) + ai, col < EK, other=0
                )
                bv = tl.load(
                    B.to(tl.pointer_type(tl.int64)) + bi, col < EV, other=0
                )
                tl.store(K.to(tl.pointer_type(tl.int64)) + ki, av, col < EK)
                tl.store(V.to(tl.pointer_type(tl.int64)) + vi, bv, col < EV)
            else:
                av = tl.load(A + ai, col < EK, other=0)
                bv = tl.load(B + bi, col < EV, other=0)
                tl.store(K + ki, av, col < EK)
                tl.store(V + vi, bv, col < EV)
        else:
            if part * W < EK:
                if PACK:
                    av = tl.load(
                        A.to(tl.pointer_type(tl.int64)) + ai, col < EK, other=0
                    )
                    tl.store(
                        K.to(tl.pointer_type(tl.int64)) + ki, av, col < EK
                    )
                else:
                    av = tl.load(A + ai, col < EK, other=0)
                    tl.store(K + ki, av, col < EK)
            if part * W < EV:
                if PACK:
                    bv = tl.load(
                        B.to(tl.pointer_type(tl.int64)) + bi, col < EV, other=0
                    )
                    tl.store(
                        V.to(tl.pointer_type(tl.int64)) + vi, bv, col < EV
                    )
                else:
                    bv = tl.load(B + bi, col < EV, other=0)
                    tl.store(V + vi, bv, col < EV)


@triton.jit
def _flat(
    K,
    V,
    A,
    B,
    L,
    CAP: tl.constexpr,
    EK: tl.constexpr,
    EV: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    LS: tl.constexpr,
    NB: tl.constexpr,
    W: tl.constexpr,
    COUNT: tl.constexpr,
    SERIAL: tl.constexpr,
    PACK: tl.constexpr,
):
    for j in range(COUNT if SERIAL else 1):
        work = j if SERIAL else tl.program_id(0)
        req = (work // NB).to(tl.int64)
        part = work % NB
        col = part * W + tl.arange(0, W)
        loc = tl.load(L + req * LS).to(tl.int64)
        slot = tl.where(loc < 0, loc + CAP, loc)
        ai = req * AS + col.to(tl.int64)
        bi = req * BS + col.to(tl.int64)
        ki = slot * KS + col.to(tl.int64)
        vi = slot * VS + col.to(tl.int64)
        if EK == EV:
            if PACK:
                av = tl.load(
                    A.to(tl.pointer_type(tl.int64)) + ai, col < EK, other=0
                )
                bv = tl.load(
                    B.to(tl.pointer_type(tl.int64)) + bi, col < EV, other=0
                )
                tl.store(K.to(tl.pointer_type(tl.int64)) + ki, av, col < EK)
                tl.store(V.to(tl.pointer_type(tl.int64)) + vi, bv, col < EV)
            else:
                av = tl.load(A + ai, col < EK, other=0)
                bv = tl.load(B + bi, col < EV, other=0)
                tl.store(K + ki, av, col < EK)
                tl.store(V + vi, bv, col < EV)
        else:
            if part * W < EK:
                if PACK:
                    av = tl.load(
                        A.to(tl.pointer_type(tl.int64)) + ai, col < EK, other=0
                    )
                    tl.store(
                        K.to(tl.pointer_type(tl.int64)) + ki, av, col < EK
                    )
                else:
                    av = tl.load(A + ai, col < EK, other=0)
                    tl.store(K + ki, av, col < EK)
            if part * W < EV:
                if PACK:
                    bv = tl.load(
                        B.to(tl.pointer_type(tl.int64)) + bi, col < EV, other=0
                    )
                    tl.store(
                        V.to(tl.pointer_type(tl.int64)) + vi, bv, col < EV
                    )
                else:
                    bv = tl.load(B + bi, col < EV, other=0)
                    tl.store(V + vi, bv, col < EV)


@triton.jit
def _flat_u8(
    K,
    V,
    A,
    B,
    L,
    CAP,
    N,
    NBLK,
    NPROG,
    KS: tl.constexpr,
    VS: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    LS: tl.constexpr,
    W: tl.constexpr,
):
    pid = tl.program_id(0)
    col = tl.arange(0, W)
    col64 = col.to(tl.int64)
    for blk in range(pid, NBLK, NPROG):
        q0 = tl.maximum(tl.minimum(blk * 8, N - 8), 0)
        r0 = (q0 + 0).to(tl.int64)
        r1 = (q0 + 1).to(tl.int64)
        r2 = (q0 + 2).to(tl.int64)
        r3 = (q0 + 3).to(tl.int64)
        r4 = (q0 + 4).to(tl.int64)
        r5 = (q0 + 5).to(tl.int64)
        r6 = (q0 + 6).to(tl.int64)
        r7 = (q0 + 7).to(tl.int64)
        l0 = tl.load(L + r0 * LS).to(tl.int64)
        l1 = tl.load(L + r1 * LS).to(tl.int64)
        l2 = tl.load(L + r2 * LS).to(tl.int64)
        l3 = tl.load(L + r3 * LS).to(tl.int64)
        l4 = tl.load(L + r4 * LS).to(tl.int64)
        l5 = tl.load(L + r5 * LS).to(tl.int64)
        l6 = tl.load(L + r6 * LS).to(tl.int64)
        l7 = tl.load(L + r7 * LS).to(tl.int64)
        s0 = tl.where(l0 < 0, l0 + CAP, l0)
        s1 = tl.where(l1 < 0, l1 + CAP, l1)
        s2 = tl.where(l2 < 0, l2 + CAP, l2)
        s3 = tl.where(l3 < 0, l3 + CAP, l3)
        s4 = tl.where(l4 < 0, l4 + CAP, l4)
        s5 = tl.where(l5 < 0, l5 + CAP, l5)
        s6 = tl.where(l6 < 0, l6 + CAP, l6)
        s7 = tl.where(l7 < 0, l7 + CAP, l7)
        a0 = tl.load(A + r0 * AS + col64)
        a1 = tl.load(A + r1 * AS + col64)
        a2 = tl.load(A + r2 * AS + col64)
        a3 = tl.load(A + r3 * AS + col64)
        a4 = tl.load(A + r4 * AS + col64)
        a5 = tl.load(A + r5 * AS + col64)
        a6 = tl.load(A + r6 * AS + col64)
        a7 = tl.load(A + r7 * AS + col64)
        b0 = tl.load(B + r0 * BS + col64)
        b1 = tl.load(B + r1 * BS + col64)
        b2 = tl.load(B + r2 * BS + col64)
        b3 = tl.load(B + r3 * BS + col64)
        b4 = tl.load(B + r4 * BS + col64)
        b5 = tl.load(B + r5 * BS + col64)
        b6 = tl.load(B + r6 * BS + col64)
        b7 = tl.load(B + r7 * BS + col64)
        tl.store(K + s0 * KS + col64, a0)
        tl.store(V + s0 * VS + col64, b0)
        tl.store(K + s1 * KS + col64, a1)
        tl.store(V + s1 * VS + col64, b1)
        tl.store(K + s2 * KS + col64, a2)
        tl.store(V + s2 * VS + col64, b2)
        tl.store(K + s3 * KS + col64, a3)
        tl.store(V + s3 * VS + col64, b3)
        tl.store(K + s4 * KS + col64, a4)
        tl.store(V + s4 * VS + col64, b4)
        tl.store(K + s5 * KS + col64, a5)
        tl.store(V + s5 * VS + col64, b5)
        tl.store(K + s6 * KS + col64, a6)
        tl.store(V + s6 * VS + col64, b6)
        tl.store(K + s7 * KS + col64, a7)
        tl.store(V + s7 * VS + col64, b7)


def _inner(t):
    return t.stride(-1) == 1 and (
        t.shape[-2] == 1 or t.stride(-2) == t.shape[-1]
    )


def store_cache_4d(k_view, v_view, cache_k, cache_v, loc, page_size):
    loc = _loc_arg(loc)
    n = loc.numel()
    ek = k_view.shape[2] * k_view.shape[3]
    ev = v_view.shape[2] * v_view.shape[3]
    if n == 0 or max(ek, ev) == 0:
        return k_view, v_view
    kr = k_view.untyped_storage().data_ptr()
    vr = v_view.untyped_storage().data_ptr()
    if cache_k.untyped_storage().data_ptr() in (
        kr,
        vr,
    ) or cache_v.untyped_storage().data_ptr() in (kr, vr):
        a = torch.empty(
            cache_k.shape, device=cache_k.device, dtype=cache_k.dtype
        )
        b = torch.empty(
            cache_v.shape, device=cache_v.device, dtype=cache_v.dtype
        )
        _save[(triton.cdiv(max(cache_k.numel(), cache_v.numel()), 1024),)](
            cache_k,
            cache_v,
            a,
            b,
            cache_k.numel(),
            cache_v.numel(),
            tuple(cache_k.shape),
            tuple(cache_v.shape),
            tuple(cache_k.stride()),
            tuple(cache_v.stride()),
            1024,
            num_warps=2,
            num_stages=1,
        )
        cache_k = a
        cache_v = b
    kc = _inner(k_view)
    vc = _inner(v_view)
    ac = _inner(cache_k)
    bc = _inner(cache_v)
    serial = not (
        kc
        and vc
        and all(
            t.data_ptr() % 128 == 0
            and t.stride(0) * t.element_size() % 128 == 0
            and t.stride(1) * t.element_size() % 128 == 0
            for t in (k_view, v_view)
        )
    )
    pack = False
    x = ek // 4 if pack else ek
    y = ev // 4 if pack else ev
    w = min(2048, max(16, triton.next_power_of_2(max(x, y))))
    nb = triton.cdiv(max(x, y), w)
    count = n * nb
    if (
        kc
        and vc
        and ac
        and bc
        and k_view.stride(0) == page_size * k_view.stride(1)
        and v_view.stride(0) == page_size * v_view.stride(1)
    ):
        if not serial and x == y and x == w and n > 48:
            nblk = (n + 7) // 8
            _flat_u8[(min(nblk, 48),)](
                k_view,
                v_view,
                cache_k,
                cache_v,
                loc,
                k_view.shape[0] * page_size,
                n,
                nblk,
                min(nblk, 48),
                k_view.stride(1),
                v_view.stride(1),
                cache_k.stride(0),
                cache_v.stride(0),
                loc.stride(0),
                W=w,
                num_warps=2,
                num_stages=1,
            )
            return k_view, v_view
        div = 4 if pack else 1
        if not serial and n * triton.cdiv(max(x, y), min(256, w)) <= 48:
            wf = min(256, w)
            nw = 1
        else:
            wf = w
            nw = 2
        nbf = triton.cdiv(max(x, y), wf)
        countf = n * nbf
        _flat[(1 if serial else countf,)](
            k_view,
            v_view,
            cache_k,
            cache_v,
            loc,
            k_view.shape[0] * page_size,
            x,
            y,
            k_view.stride(1) // div,
            v_view.stride(1) // div,
            cache_k.stride(0) // div,
            cache_v.stride(0) // div,
            loc.stride(0),
            nbf,
            wf,
            countf,
            serial,
            pack,
            num_warps=nw,
            num_stages=1,
        )
        return k_view, v_view
    _pair[(1 if serial else count,)](
        k_view,
        v_view,
        cache_k,
        cache_v,
        loc,
        page_size,
        k_view.shape[0],
        x,
        y,
        k_view.shape[3],
        v_view.shape[3],
        tuple(k_view.stride()),
        tuple(v_view.stride()),
        tuple(cache_k.stride()),
        tuple(cache_v.stride()),
        loc.stride(0),
        kc,
        vc,
        ac,
        bc,
        nb,
        w,
        count,
        serial,
        pack,
        num_warps=2,
        num_stages=1,
    )
    return k_view, v_view


__all__ = ["store_cache_4d"]
