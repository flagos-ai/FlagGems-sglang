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
        loc = tl.load(L + req * LS).to(tl.int64)
        if NP * P < 9223372036854775808:
            slot = loc + tl.where(loc < 0, tl.full((), NP * P, tl.int64), 0)
            page = slot // P
            token = slot - page * P
        else:
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
            if AC:
                ac = col.to(tl.int64)
            elif AT[1] == DK * AT[2]:
                ac = col.to(tl.int64) * AT[2]
            else:
                ac = (col // DK).to(tl.int64) * AT[1] + (col % DK).to(
                    tl.int64
                ) * AT[2]
            if BC:
                bc = col.to(tl.int64)
            elif BT[1] == DV * BT[2]:
                bc = col.to(tl.int64) * BT[2]
            else:
                bc = (col // DV).to(tl.int64) * BT[1] + (col % DV).to(
                    tl.int64
                ) * BT[2]
            if KC:
                kc = col.to(tl.int64)
            elif KT[2] == DK * KT[3]:
                kc = col.to(tl.int64) * KT[3]
            else:
                kc = (col // DK).to(tl.int64) * KT[2] + (col % DK).to(
                    tl.int64
                ) * KT[3]
            if VC:
                vc = col.to(tl.int64)
            elif VT[2] == DV * VT[3]:
                vc = col.to(tl.int64) * VT[3]
            else:
                vc = (col // DV).to(tl.int64) * VT[2] + (col % DV).to(
                    tl.int64
                ) * VT[3]
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
def _owner_stream(
    X,
    Y,
    L,
    OWNER,
    N: tl.constexpr,
    P: tl.constexpr,
    CAP: tl.constexpr,
    E: tl.constexpr,
    PS: tl.constexpr,
    XS: tl.constexpr,
    LS: tl.constexpr,
    BE: tl.constexpr,
    NB: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    page = OWNER // NB
    chunk = OWNER - page * NB
    first = chunk * BE
    end = tl.minimum(first + BE, P * E)
    request = tl.arange(0, R)
    if LS == 2:
        span = tl.arange(0, 2 * R)
        raw = tl.load(L + span.to(tl.int64), span < 2 * N - 1, other=0).to(
            tl.int32
        )
        slot = tl.gather(raw.to(tl.float32), request * 2, axis=0).to(tl.int32)
    else:
        slot = tl.load(L + request.to(tl.int64) * LS, request < N, other=0).to(
            tl.int32
        )
    slot = tl.where(slot < 0, slot + CAP, slot)
    begin_token = first // E
    end_token = (end + E - 1) // E
    lane = tl.arange(0, B)
    for token in range(begin_token, end_token):
        wanted = page * P + token
        selected = tl.max(
            tl.where((request < N) & (slot == wanted), request, -1), 0
        )
        if selected >= 0:
            low = tl.maximum(first - token * E, 0)
            high = tl.minimum(end - token * E, E)
            col = low + lane
            value = tl.load(
                X + selected.to(tl.int64) * XS + col.to(tl.int64),
                col < high,
                other=0,
            )
            offset = (
                page.to(tl.int64) * PS
                + tl.cast(token, tl.int64) * E
                + col.to(tl.int64)
            )
            tl.store(Y + offset, value, col < high)


@triton.jit
def _page_owners(
    K,
    V,
    A,
    B,
    L,
    N: tl.constexpr,
    P: tl.constexpr,
    CAP: tl.constexpr,
    EK: tl.constexpr,
    EV: tl.constexpr,
    KPS: tl.constexpr,
    VPS: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    LS: tl.constexpr,
    BEK: tl.constexpr,
    BEV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    GK: tl.constexpr,
    R: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    owner = tl.program_id(0)
    if owner < GK:
        _owner_stream(
            A, K, L, owner, N, P, CAP, EK, KPS, AS, LS, BEK, NK, R, BK
        )
    else:
        _owner_stream(
            B, V, L, owner - GK, N, P, CAP, EV, VPS, BS, LS, BEV, NV, R, BV
        )


def store_cache_4d(k_view, v_view, cache_k, cache_v, loc, page_size):
    kshape = k_view.shape
    vshape = v_view.shape
    n = loc.numel()
    ek = kshape[2] * kshape[3]
    ev = vshape[2] * vshape[3]
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
        _save[(((max(cache_k.numel(), cache_v.numel()) + 1023) // 1024),)](
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
    ks = k_view.stride()
    vs = v_view.stride()
    as_ = cache_k.stride()
    bs_ = cache_v.stride()
    ls = loc.stride(0)
    kc = ks[-1] == 1 and (kshape[-2] == 1 or ks[-2] == kshape[-1])
    vc = vs[-1] == 1 and (vshape[-2] == 1 or vs[-2] == vshape[-1])
    ac = as_[-1] == 1 and (
        cache_k.shape[-2] == 1 or as_[-2] == cache_k.shape[-1]
    )
    bc = bs_[-1] == 1 and (
        cache_v.shape[-2] == 1 or bs_[-2] == cache_v.shape[-1]
    )
    if kc and vc:
        ke = k_view.element_size()
        ve = v_view.element_size()
        kp = k_view.data_ptr()
        vp = v_view.data_ptr()
        page_aligned = (
            kp % 128 == 0
            and vp % 128 == 0
            and ks[0] * ke % 128 == 0
            and vs[0] * ve % 128 == 0
        )
        serial = not (
            page_aligned and ks[1] * ke % 128 == 0 and vs[1] * ve % 128 == 0
        )
    else:
        serial = True
    if (
        serial
        and 32 <= n <= 512
        and kr != vr
        and kc
        and vc
        and ac
        and bc
        and loc.dtype in (torch.int32, torch.int64)
        and ls > 0
        and loc.untyped_storage().data_ptr() not in (kr, vr)
        and kshape[0] == vshape[0]
        and kshape[1] == vshape[1] == page_size
        and page_size > 0
        and kshape[0] * page_size <= 2 * n
        and ke in (2, 4, 8)
        and ve in (2, 4, 8)
        and 16 <= ek * ke < 128
        and 16 <= ev * ve < 128
        and ks[1] == ek
        and vs[1] == ev
        and as_[0] >= ek
        and bs_[0] >= ev
        and page_aligned
        and ks[0] >= ((page_size * ek * ke + 127) // 128) * (128 // ke)
        and vs[0] >= ((page_size * ev * ve + 127) // 128) * (128 // ve)
    ):
        bek = 128 // ke
        bev = 128 // ve
        nk = (page_size * ek + bek - 1) // bek
        nv = (page_size * ev + bev - 1) // bev
        gk = kshape[0] * nk
        gv = vshape[0] * nv
        if (
            2 <= gk + gv <= 256
            and (gk + gv) * n * loc.element_size() <= 1048576
        ):
            _page_owners[(gk + gv,)](
                k_view,
                v_view,
                cache_k,
                cache_v,
                loc,
                n,
                page_size,
                kshape[0] * page_size,
                ek,
                ev,
                ks[0],
                vs[0],
                as_[0],
                bs_[0],
                ls,
                bek,
                bev,
                nk,
                nv,
                gk,
                1 << (n - 1).bit_length(),
                1 << (ek - 1).bit_length(),
                1 << (ev - 1).bit_length(),
                num_warps=4,
                num_stages=1,
            )
            return k_view, v_view
    pack = False
    x = ek // 4 if pack else ek
    y = ev // 4 if pack else ev
    w = min(2048, max(16, (1 << (max(x, y) - 1).bit_length())))
    nb = (max(x, y) + w - 1) // w
    count = n * nb
    _pair[(1 if serial else count,)](
        k_view,
        v_view,
        cache_k,
        cache_v,
        loc,
        page_size,
        kshape[0],
        x,
        y,
        kshape[3],
        vshape[3],
        ks,
        vs,
        as_,
        bs_,
        ls,
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
