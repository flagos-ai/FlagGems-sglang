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

import torch
import triton
import triton.language as tl


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
        if EK == EV and not PACK:
            pre_ac = (
                col.to(tl.int64)
                if AC
                else (col // DK).to(tl.int64) * AT[1]
                + (col % DK).to(tl.int64) * AT[2]
            )
            pre_bc = (
                col.to(tl.int64)
                if BC
                else (col // DV).to(tl.int64) * BT[1]
                + (col % DV).to(tl.int64) * BT[2]
            )
            pre_a = tl.load(A + req * AT[0] + pre_ac, col < EK, other=0)
            pre_b = tl.load(B + req * BT[0] + pre_bc, col < EV, other=0)
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
                av = pre_a
                bv = pre_b
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
    N: tl.constexpr = COUNT // NB
    TAIL: tl.constexpr = NB * W - 1
    I32: tl.constexpr = (
        (W == 256)
        and 0 < CAP
        and CAP < 2147483648
        and (0 < COUNT)
        and (COUNT < 2147483648)
        and (0 <= TAIL)
        and (TAIL < 2147483648)
        and (0 <= KS)
        and (KS < 2147483648)
        and (0 <= VS)
        and (VS < 2147483648)
        and (0 <= AS)
        and (AS < 2147483648)
        and (0 <= BS)
        and (BS < 2147483648)
        and (0 <= LS)
        and (LS < 2147483648)
        and ((CAP - 1) * KS + TAIL < 2147483648)
        and ((CAP - 1) * VS + TAIL < 2147483648)
        and ((N - 1) * AS + TAIL < 2147483648)
        and ((N - 1) * BS + TAIL < 2147483648)
        and ((N - 1) * LS < 2147483648)
    )
    for j in range(COUNT if SERIAL else 1):
        work = j if SERIAL else tl.program_id(0)
        if I32:
            req = (work // NB).to(tl.int32)
            part = work % NB
            col = part * W + tl.arange(0, W)
            if EK == EV and not PACK:
                pre_ai = req * AS + col.to(tl.int32)
                pre_bi = req * BS + col.to(tl.int32)
                pre_a = tl.load(A + pre_ai, col < EK, other=0)
                pre_b = tl.load(B + pre_bi, col < EV, other=0)
            loc = tl.load(L + req * LS).to(tl.int32)
            slot = loc + tl.where(loc < 0, CAP, 0)
            ai = req * AS + col.to(tl.int32)
            bi = req * BS + col.to(tl.int32)
            ki = slot * KS + col.to(tl.int32)
            vi = slot * VS + col.to(tl.int32)
        else:
            req = (work // NB).to(tl.int64)
            part = work % NB
            col = part * W + tl.arange(0, W)
            if EK == EV and not PACK:
                pre_ai = req * AS + col.to(tl.int64)
                pre_bi = req * BS + col.to(tl.int64)
                pre_a = tl.load(A + pre_ai, col < EK, other=0)
                pre_b = tl.load(B + pre_bi, col < EV, other=0)
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
                av = pre_a
                bv = pre_b
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


def _inner(t):
    return t.stride(-1) == 1 and (
        t.shape[-2] == 1 or t.stride(-2) == t.shape[-1]
    )


def store_cache_4d(k_view, v_view, cache_k, cache_v, loc, page_size):
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
            num_warps=4,
            num_stages=1,
        )
        cache_k = a
        cache_v = b
    kc = _inner(k_view)
    vc = _inner(v_view)
    ac = _inner(cache_k)
    bc = _inner(cache_v)
    serial = False
    pack = False
    x = ek // 4 if pack else ek
    y = ev // 4 if pack else ev
    w = min(1024 if n >= 48 else 256, triton.next_power_of_2(max(x, y)))
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
        div = 4 if pack else 1
        _flat[(1 if serial else count,)](
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
            nb,
            w,
            count,
            serial,
            pack,
            num_warps=4,
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
        num_warps=4,
        num_stages=1,
    )
    return k_view, v_view


__all__ = ["store_cache_4d"]
