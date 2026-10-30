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

import triton
import triton.language as tl


@triton.jit
def _tile(
    X,
    W,
    group,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    WS: tl.constexpr,
    E: tl.constexpr,
    SCALE: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
):
    if M < 512:
        row = group * R + tl.arange(0, R)
        col = tl.arange(0, B)
        offset = row[:, None] // H * S + row[:, None] % H * D + col[None, :]
        mask = (row[:, None] < M * H) & (col[None, :] < D)
        x = tl.load(X + offset, mask, other=0).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E)
        if SCALE:
            w = tl.load(
                tl.broadcast_to(W + col[None, :] * WS, (R, B)), mask, other=0
            ).to(tl.float32)
            y = y * w
        tl.store(X + offset, y, mask)
    else:
        row = group * R + tl.arange(0, R)
        col = tl.arange(0, B)
        offset = row[:, None] // H * S + row[:, None] % H * D + col[None, :]
        mask = (row[:, None] < M * H) & (col[None, :] < D)
        x = tl.load(X + offset, mask, other=0).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E)
        if SCALE:
            w = tl.load(W + col * WS, col < D, other=0).to(tl.float32)
            y = y * w[None, :]
        tl.store(X + offset, y, mask)


@triton.jit
def _dispatch(
    Q,
    K,
    V,
    WQ,
    WK,
    group,
    M: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    WQS: tl.constexpr,
    WKS: tl.constexpr,
    E: tl.constexpr,
    KV: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
):
    GQ: tl.constexpr = (M * HQ + R - 1) // R
    GK: tl.constexpr = (M * HK + R - 1) // R
    if group < GQ:
        _tile(Q, WQ, group, M, HQ, D, SQ, WQS, E, True, B, R)
    elif KV:
        if group < GQ + GK:
            _tile(K, WK, group - GQ, M, HK, D, SK, WKS, E, True, B, R)
        else:
            _tile(V, WQ, group - GQ - GK, M, HK, D, SV, WQS, E, False, B, R)


@triton.jit
def _kernel(Q, K, V, WQ, WK, META: tl.constexpr):
    M: tl.constexpr = META[0]
    HQ: tl.constexpr = META[1]
    HK: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    SQ: tl.constexpr = META[4]
    SK: tl.constexpr = META[5]
    SV: tl.constexpr = META[6]
    WQS: tl.constexpr = META[7]
    WKS: tl.constexpr = META[8]
    E: tl.constexpr = META[9]
    KV: tl.constexpr = META[10]
    B: tl.constexpr = META[11]
    R: tl.constexpr = META[12]
    PERSIST: tl.constexpr = META[13]
    if PERSIST:
        TOTAL: tl.constexpr = (M * HQ + R - 1) // R + (
            2 * ((M * HK + R - 1) // R) if KV else 0
        )
        for group in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
            _dispatch(
                Q,
                K,
                V,
                WQ,
                WK,
                group,
                M,
                HQ,
                HK,
                D,
                SQ,
                SK,
                SV,
                WQS,
                WKS,
                E,
                KV,
                B,
                R,
            )
    else:
        _dispatch(
            Q,
            K,
            V,
            WQ,
            WK,
            tl.program_id(0),
            M,
            HQ,
            HK,
            D,
            SQ,
            SK,
            SV,
            WQS,
            WKS,
            E,
            KV,
            B,
            R,
        )


def _separate(
    q, k, v, q_weight, k_weight, num_q_heads, num_kv_heads, head_dim, eps
):
    m = q.shape[0]
    kv = k is not None and v is not None
    rows = 4
    groups = ((m * num_q_heads + rows - 1) // rows) + (
        2 * ((m * num_kv_heads + rows - 1) // rows) if kv else 0
    )
    if groups:
        _kernel.run(
            q,
            k if kv else q,
            v if kv else q,
            q_weight,
            k_weight if kv else q_weight,
            (
                m,
                num_q_heads,
                num_kv_heads,
                head_dim,
                q.stride(0),
                k.stride(0) if kv else 0,
                v.stride(0) if kv else 0,
                q_weight.stride(0),
                k_weight.stride(0) if kv else 0,
                eps,
                kv,
                (1 << (head_dim - 1).bit_length()),
                rows,
                False,
            ),
            num_warps=4,
            enable_fp_fusion=False,
            grid=(groups,),
            warmup=False,
        )
    return (q, k, v)


@triton.jit
def _unified(Q, K, V, WQ, WK, META: tl.constexpr):
    if META[0] < 512:
        M: tl.constexpr = META[0]
        HQ: tl.constexpr = META[1]
        HK: tl.constexpr = META[2]
        D: tl.constexpr = META[3]
        SQ: tl.constexpr = META[4]
        SK: tl.constexpr = META[5]
        SV: tl.constexpr = META[6]
        WQS: tl.constexpr = META[7]
        WKS: tl.constexpr = META[8]
        E: tl.constexpr = META[9]
        KV: tl.constexpr = META[10]
        R: tl.constexpr = META[11]
        B: tl.constexpr = META[12]
        which = tl.program_id(1)
        row = tl.program_id(0) * R + tl.arange(0, R)
        col = tl.arange(0, B)
        X = Q
        W = WQ
        off = row[:, None] // HQ * SQ + row[:, None] % HQ * D + col[None, :]
        rows = M * HQ
        ws = WQS
        if KV:
            if which == 1:
                X = K
                W = WK
                off = (
                    row[:, None] // HK * SK
                    + row[:, None] % HK * D
                    + col[None, :]
                )
                rows = M * HK
                ws = WKS
            if which == 2:
                X = V
                off = (
                    row[:, None] // HK * SV
                    + row[:, None] % HK * D
                    + col[None, :]
                )
                rows = M * HK
        mask = (row[:, None] < rows) & (col[None, :] < D)
        x = tl.load(X + off, mask, other=0).to(tl.float32)
        w = tl.load(
            tl.broadcast_to(W + col[None, :] * ws, (R, B)), mask, other=0
        ).to(tl.float32)
        w = tl.where(which == 2, 1.0, w)
        y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E) * w
        tl.store(X + off, y, mask)
    else:
        M: tl.constexpr = META[0]
        HQ: tl.constexpr = META[1]
        HK: tl.constexpr = META[2]
        D: tl.constexpr = META[3]
        SQ: tl.constexpr = META[4]
        SK: tl.constexpr = META[5]
        SV: tl.constexpr = META[6]
        WQS: tl.constexpr = META[7]
        WKS: tl.constexpr = META[8]
        E: tl.constexpr = META[9]
        KV: tl.constexpr = META[10]
        R: tl.constexpr = META[11]
        B: tl.constexpr = META[12]
        which = tl.program_id(1)
        row = tl.program_id(0) * R + tl.arange(0, R)
        col = tl.arange(0, B)
        X = Q
        W = WQ
        off = row[:, None] // HQ * SQ + row[:, None] % HQ * D + col[None, :]
        rows = M * HQ
        ws = WQS
        if KV:
            if which == 1:
                X = K
                W = WK
                off = (
                    row[:, None] // HK * SK
                    + row[:, None] % HK * D
                    + col[None, :]
                )
                rows = M * HK
                ws = WKS
            if which == 2:
                X = V
                off = (
                    row[:, None] // HK * SV
                    + row[:, None] % HK * D
                    + col[None, :]
                )
                rows = M * HK
        mask = (row[:, None] < rows) & (col[None, :] < D)
        x = tl.load(X + off, mask, other=0).to(tl.float32)
        w = tl.load(W + col * ws, col < D, other=0).to(tl.float32)
        w = tl.where(which == 2, 1.0, w)
        y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E) * w[None, :]
        tl.store(X + off, y, mask)


@triton.jit
def _lane(Q, K, V, WQ, WK, META: tl.constexpr):
    M: tl.constexpr = META[0]
    HQ: tl.constexpr = META[1]
    HK: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    SQ: tl.constexpr = META[4]
    SK: tl.constexpr = META[5]
    SV: tl.constexpr = META[6]
    WQS: tl.constexpr = META[7]
    WKS: tl.constexpr = META[8]
    E: tl.constexpr = META[9]
    KV: tl.constexpr = META[10]
    R: tl.constexpr = META[11]
    B: tl.constexpr = META[12]
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, B)
    off = row[:, None] // HQ * SQ + row[:, None] % HQ * D + col[None, :]
    ptr = Q + off
    wptr = WQ + col[None, :] * WQS + tl.zeros((R, 1), tl.int32)
    total: tl.constexpr = M * HQ + (2 * M * HK if KV else 0)
    if KV:
        kr = row - M * HQ
        vr = row - M * (HQ + HK)
        ko = kr[:, None] // HK * SK + kr[:, None] % HK * D + col[None, :]
        vo = vr[:, None] // HK * SV + vr[:, None] % HK * D + col[None, :]
        ptr = tl.where(
            row[:, None] < M * HQ,
            ptr,
            tl.where(row[:, None] < M * (HQ + HK), K + ko, V + vo),
        )
        wptr = tl.where(row[:, None] < M * HQ, wptr, WK + col[None, :] * WKS)
    mask = (row[:, None] < total) & (col[None, :] < D)
    x = tl.load(ptr, mask, other=0).to(tl.float32)
    w = tl.load(wptr, mask, other=0).to(tl.float32)
    if KV:
        w = tl.where(row[:, None] >= M * (HQ + HK), 1.0, w)
    y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E) * w
    tl.store(ptr, y, mask)


@triton.jit
def _uniform(Q, K, V, WQ, WK, META: tl.constexpr):
    if META[3] == 256:
        M: tl.constexpr = META[0]
        HQ: tl.constexpr = META[1]
        HK: tl.constexpr = META[2]
        D: tl.constexpr = META[3]
        SQ: tl.constexpr = META[4]
        SK: tl.constexpr = META[5]
        SV: tl.constexpr = META[6]
        WQS: tl.constexpr = META[7]
        WKS: tl.constexpr = META[8]
        E: tl.constexpr = META[9]
        KV: tl.constexpr = META[10]
        R: tl.constexpr = META[11]
        B: tl.constexpr = META[12]
        GQ: tl.constexpr = (M * HQ + R - 1) // R
        GK: tl.constexpr = (M * HK + R - 1) // R
        p = tl.program_id(0)
        which = tl.where(p < GQ, 0, tl.where(p < GQ + GK, 1, 2))
        group = p - tl.where(which == 0, 0, tl.where(which == 1, GQ, GQ + GK))
        row = group * R + tl.arange(0, R)
        col = tl.max_contiguous(tl.arange(0, B // 2), 2)
        ptr = Q + row // HQ * SQ + row % HQ * D
        wp = WQ + col * WQS
        count = M * HQ
        if KV:
            ptr = tl.where(
                which == 0,
                ptr,
                tl.where(
                    which == 1,
                    K + row // HK * SK + row % HK * D,
                    V + row // HK * SV + row % HK * D,
                ),
            )
            wp = tl.where(which == 0, wp, WK + col * WKS)
            count = tl.where(which == 0, M * HQ, M * HK)
        mask = (row[:, None] < count) & (col[None, :] < D)
        wstep = WQS
        if KV:
            wstep = tl.where(which == 0, WQS, WKS)
        x0 = tl.load(
            ptr[:, None] + col[None, :] + 0 * (B // 2), mask, other=0
        ).to(tl.float32)
        x1 = tl.load(
            ptr[:, None] + col[None, :] + 1 * (B // 2), mask, other=0
        ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1, 1) / D + E)
        w0 = tl.load(
            tl.broadcast_to(wp[None, :] + 0 * (B // 2) * wstep, (R, B // 2)),
            mask,
            other=0,
        ).to(tl.float32)
        if KV:
            w0 = tl.where(which == 2, 1.0, w0)
        y0 = x0 * inv[:, None] * w0
        tl.store(ptr[:, None] + col[None, :] + 0 * (B // 2), y0, mask)
        w1 = tl.load(
            tl.broadcast_to(wp[None, :] + 1 * (B // 2) * wstep, (R, B // 2)),
            mask,
            other=0,
        ).to(tl.float32)
        if KV:
            w1 = tl.where(which == 2, 1.0, w1)
        y1 = x1 * inv[:, None] * w1
        tl.store(ptr[:, None] + col[None, :] + 1 * (B // 2), y1, mask)
    else:
        M: tl.constexpr = META[0]
        HQ: tl.constexpr = META[1]
        HK: tl.constexpr = META[2]
        D: tl.constexpr = META[3]
        SQ: tl.constexpr = META[4]
        SK: tl.constexpr = META[5]
        SV: tl.constexpr = META[6]
        WQS: tl.constexpr = META[7]
        WKS: tl.constexpr = META[8]
        E: tl.constexpr = META[9]
        KV: tl.constexpr = META[10]
        R: tl.constexpr = META[11]
        B: tl.constexpr = META[12]
        GQ: tl.constexpr = (M * HQ + R - 1) // R
        GK: tl.constexpr = (M * HK + R - 1) // R
        p = tl.program_id(0)
        which = tl.where(p < GQ, 0, tl.where(p < GQ + GK, 1, 2))
        group = p - tl.where(which == 0, 0, tl.where(which == 1, GQ, GQ + GK))
        row = group * R + tl.arange(0, R)
        col = tl.arange(0, B)
        ptr = Q + row // HQ * SQ + row % HQ * D
        wp = WQ + col * WQS
        count = M * HQ
        if KV:
            ptr = tl.where(
                which == 0,
                ptr,
                tl.where(
                    which == 1,
                    K + row // HK * SK + row % HK * D,
                    V + row // HK * SV + row % HK * D,
                ),
            )
            wp = tl.where(which == 0, wp, WK + col * WKS)
            count = tl.where(which == 0, M * HQ, M * HK)
        mask = (row[:, None] < count) & (col[None, :] < D)
        x = tl.load(ptr[:, None] + col[None, :], mask, other=0).to(tl.float32)
        w = tl.load(wp, col < D, other=0).to(tl.float32)
        if KV:
            w = tl.where(which == 2, 1.0, w)
        y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E) * w[None, :]
        tl.store(ptr[:, None] + col[None, :], y, mask)


@triton.jit
def _flat(
    Q,
    K,
    V,
    WQ,
    WK,
    M: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    WQS: tl.constexpr,
    WKS: tl.constexpr,
    E: tl.constexpr,
    KV: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
):
    GQ: tl.constexpr = (M * HQ + R - 1) // R
    GK: tl.constexpr = (M * HK + R - 1) // R
    p = tl.program_id(0)
    group = p
    base = Q
    weight = WQ
    ws = WQS
    count = M * HQ
    scale = True
    if KV:
        which = tl.where(p < GQ, 0, tl.where(p < GQ + GK, 1, 2))
        group = p - tl.where(which == 0, 0, tl.where(which == 1, GQ, GQ + GK))
        base = tl.where(which == 0, Q, tl.where(which == 1, K, V))
        weight = tl.where(which == 0, WQ, WK)
        ws = tl.where(which == 0, WQS, WKS)
        count = tl.where(which == 0, M * HQ, M * HK)
        scale = which != 2
    row = group * R + tl.arange(0, R)
    col = tl.arange(0, B)
    if M >= 256:
        col = tl.max_contiguous(col, 2 if B >= 2 else B)
    offset = row[:, None] * D + col[None, :]
    mask = (row[:, None] < count) & (col[None, :] < D)
    complete: tl.constexpr = (M * HQ) % R == 0 and (
        not KV or (M * HK) % R == 0
    )
    memory_mask = (
        tl.broadcast_to(col[None, :] < D, (R, B)) if complete else mask
    )
    x = tl.load(base + offset, memory_mask, other=0).to(tl.float32)
    total = tl.sum(x * x, 1)
    y = x * tl.rsqrt(total[:, None] / D + E)
    w = tl.load(
        tl.broadcast_to(weight + col[None, :] * ws, (R, B)),
        memory_mask & scale,
        other=1,
    ).to(tl.float32)
    y = y * w
    tl.store(base + offset, y, memory_mask)


@triton.jit
def _small_flat(Q, K, V, WQ, WK, META: tl.constexpr):
    M: tl.constexpr = META[0]
    HQ: tl.constexpr = META[1]
    HK: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    SQ: tl.constexpr = META[4]
    SK: tl.constexpr = META[5]
    SV: tl.constexpr = META[6]
    WQS: tl.constexpr = META[7]
    WKS: tl.constexpr = META[8]
    E: tl.constexpr = META[9]
    KV: tl.constexpr = META[10]
    R: tl.constexpr = META[11]
    B: tl.constexpr = META[12]
    GQ: tl.constexpr = (M * HQ + R - 1) // R
    GK: tl.constexpr = (M * HK + R - 1) // R
    p = tl.program_id(0)
    which = tl.where(p < GQ, 0, tl.where(p < GQ + GK, 1, 2))
    group = p - tl.where(which == 0, 0, tl.where(which == 1, GQ, GQ + GK))
    row = group * R + tl.arange(0, R)
    col = tl.arange(0, B)
    X = Q
    W = WQ
    off = row[:, None] // HQ * SQ + row[:, None] % HQ * D + col[None, :]
    rows = M * HQ
    ws = WQS
    if KV:
        if which == 1:
            X = K
            W = WK
            off = (
                row[:, None] // HK * SK + row[:, None] % HK * D + col[None, :]
            )
            rows = M * HK
            ws = WKS
        if which == 2:
            X = V
            off = (
                row[:, None] // HK * SV + row[:, None] % HK * D + col[None, :]
            )
            rows = M * HK
    complete: tl.constexpr = (M * HQ) % R == 0 and (
        not KV or (M * HK) % R == 0
    )
    mask = (
        tl.broadcast_to(col[None, :] < D, (R, B))
        if complete
        else ((row[:, None] < rows) & (col[None, :] < D))
    )
    x = tl.load(X + off, mask, other=0).to(tl.float32)
    w = tl.load(
        tl.broadcast_to(W + col[None, :] * ws, (R, B)), mask, other=0
    ).to(tl.float32)
    w = tl.where(which == 2, 1.0, w)
    y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E) * w
    tl.store(X + off, y, mask)


@triton.jit
def _entry(Q, K, V, WQ, WK, META: tl.constexpr):
    M: tl.constexpr = META[0]
    HQ: tl.constexpr = META[1]
    HK: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    SQ: tl.constexpr = META[4]
    SK: tl.constexpr = META[5]
    SV: tl.constexpr = META[6]
    WQS: tl.constexpr = META[7]
    WKS: tl.constexpr = META[8]
    E: tl.constexpr = META[9]
    KV: tl.constexpr = META[10]
    R: tl.constexpr = META[11]
    B: tl.constexpr = META[12]
    if (
        not KV
        or Q.dtype.element_ty == K.dtype.element_ty
        and Q.dtype.element_ty == V.dtype.element_ty
        and WQ.dtype.element_ty == WK.dtype.element_ty
    ):
        if M >= 512:
            if D >= 256:
                _uniform(Q, K, V, WQ, WK, META)
            else:
                if SQ == HQ * D and (
                    not KV or (SK == HK * D and SV == HK * D)
                ):
                    _flat(Q, K, V, WQ, WK, M, HQ, HK, D, WQS, WKS, E, KV, B, R)
                else:
                    _lane(Q, K, V, WQ, WK, META)
        else:
            _small_flat(Q, K, V, WQ, WK, META)
    else:
        if M >= 512:
            _dispatch(
                Q,
                K,
                V,
                WQ,
                WK,
                tl.program_id(0),
                M,
                HQ,
                HK,
                D,
                SQ,
                SK,
                SV,
                WQS,
                WKS,
                E,
                KV,
                B,
                R,
            )
        else:
            _dispatch(
                Q,
                K,
                V,
                WQ,
                WK,
                tl.program_id(0),
                M,
                HQ,
                HK,
                D,
                SQ,
                SK,
                SV,
                WQS,
                WKS,
                E,
                KV,
                B,
                R,
            )


def gemma_qkv_rmsnorm(
    q, k, v, q_weight, k_weight, num_q_heads, num_kv_heads, head_dim, eps
):
    m = q.shape[0]
    kv = k is not None and v is not None
    rows = (
        (16 if head_dim >= 256 else 32)
        if m >= 512
        else (
            8 if m * max(num_q_heads, num_kv_heads if kv else 0) < 128 else 4
        )
    )
    warps = 2 if m >= 512 else 4
    if m >= 512 and head_dim < 256:
        rows = 16
        warps = 2
        homogeneous = not kv or (
            q.dtype == k.dtype == v.dtype and q_weight.dtype == k_weight.dtype
        )
        contiguous = q.stride(0) == num_q_heads * head_dim and (
            not kv
            or (
                k.stride(0) == num_kv_heads * head_dim
                and v.stride(0) == num_kv_heads * head_dim
            )
        )
        if homogeneous and contiguous:
            rows = 128
            warps = 16
    if m:
        meta = (
            m,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q.stride(0),
            k.stride(0) if kv else 0,
            v.stride(0) if kv else 0,
            q_weight.stride(0),
            k_weight.stride(0) if kv else 0,
            eps,
            kv,
            rows,
            (1 << (head_dim - 1).bit_length()),
        )
        if m >= 512:
            _entry.run(
                q,
                k if kv else q,
                v if kv else q,
                q_weight,
                k_weight if kv else q_weight,
                meta,
                grid=(
                    ((m * num_q_heads + rows - 1) // rows)
                    + (
                        2 * ((m * num_kv_heads + rows - 1) // rows)
                        if kv
                        else 0
                    ),
                ),
                warmup=False,
                num_warps=warps,
                enable_fp_fusion=False,
            )
        else:
            _entry.run(
                q,
                k if kv else q,
                v if kv else q,
                q_weight,
                k_weight if kv else q_weight,
                meta,
                grid=(
                    ((m * num_q_heads + rows - 1) // rows)
                    + (
                        2 * ((m * num_kv_heads + rows - 1) // rows)
                        if kv
                        else 0
                    ),
                ),
                warmup=False,
                num_warps=(4 if m < 512 else 4),
                enable_fp_fusion=False,
            )
    return q, k, v


__all__ = ["gemma_qkv_rmsnorm"]
