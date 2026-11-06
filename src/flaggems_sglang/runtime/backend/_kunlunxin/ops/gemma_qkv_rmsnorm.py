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
def _small_combined(Q, K, V, WQ, WK, META: tl.constexpr):
    M: tl.constexpr = META[0]
    HQ: tl.constexpr = META[1]
    HK: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    SQ: tl.constexpr = META[4]
    SK: tl.constexpr = META[5]
    SV: tl.constexpr = META[6]
    E: tl.constexpr = META[7]
    R: tl.constexpr = META[8]
    r = tl.program_id(0) * R + tl.arange(0, R)
    c = tl.arange(0, D)
    which = tl.where(r < M * HQ, 0, tl.where(r < M * (HQ + HK), 1, 2))
    krow = r - M * HQ
    vrow = r - M * (HQ + HK)
    qp = Q + (r // HQ * SQ + r % HQ * D)[:, None] + c[None, :]
    kp = K + (krow // HK * SK + krow % HK * D)[:, None] + c[None, :]
    vp = V + (vrow // HK * SV + vrow % HK * D)[:, None] + c[None, :]
    ptr = tl.where(
        which[:, None] == 0, qp, tl.where(which[:, None] == 1, kp, vp)
    )
    wp = tl.where(which[:, None] == 1, WK + c[None, :], WQ + c[None, :])
    x = tl.load(ptr, offset_state_policy="continuous").to(tl.float32)
    w = tl.load(wp, offset_state_policy="continuous").to(tl.float32)
    w = tl.where(which[:, None] == 2, 1.0, w)
    y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E) * w
    tl.store(ptr, y, offset_state_policy="continuous")


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
    p = tl.program_id(0)
    X = Q
    W = WQ
    stride = SQ
    ws = WQS
    h = HQ
    row = p
    scale = True
    if KV:
        if p >= M * HQ:
            h = HK
            row = p - M * HQ
            X = K
            W = WK
            stride = SK
            ws = WKS
            if p >= M * (HQ + HK):
                row = p - M * (HQ + HK)
                X = V
                W = WQ
                stride = SV
                ws = WQS
                scale = False
    c = tl.arange(0, B)
    offset = row // h * stride + row % h * D + c
    x = tl.load(X + offset, c < D, other=0).to(tl.float32)
    w = tl.load(W + c * ws, c < D, other=0).to(tl.float32)
    w = tl.where(scale, w, 1.0)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / D + E) * w
    tl.store(X + offset, y, c < D)


@triton.jit
def _grouped(Q, K, V, WQ, WK, META: tl.constexpr):
    GQ: tl.constexpr = META[0]
    GK: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    E: tl.constexpr = META[3]
    KV: tl.constexpr = META[4]
    R: tl.constexpr = META[5]
    PQ: tl.constexpr = META[6]
    PK: tl.constexpr = META[7]
    PV: tl.constexpr = META[8]
    pid = tl.program_id(0)
    which = tl.where(pid < PQ, 0, tl.where(pid < PQ + PK, 1, 2))
    local = tl.where(
        which == 0, pid, tl.where(which == 1, pid - PQ, pid - PQ - PK)
    )
    step = tl.where(which == 0, PQ, tl.where(which == 1, PK, PV))
    X = Q
    W = WQ
    groups = GQ
    if KV:
        if which == 1:
            X = K
            W = WK
        if which == 2:
            X = V
        groups = tl.where(which == 0, GQ, GK)
    c = tl.arange(0, D)
    w = tl.load(W + c).to(tl.float32)
    w = tl.where(which == 2, 1.0, w)
    for group in range(local, groups, step):
        r = group * R + tl.arange(0, R)
        offset = r[:, None] * D + c[None, :]
        x = tl.load(X + offset, offset_state_policy="continuous").to(
            tl.float32
        )
        y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E) * w[None, :]
        tl.store(X + offset, y, offset_state_policy="continuous")


@triton.jit
def _strided(Q, K, V, WQ, WK, META: tl.constexpr):
    GQ: tl.constexpr = META[0]
    GK: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    E: tl.constexpr = META[3]
    KV: tl.constexpr = META[4]
    R: tl.constexpr = META[5]
    HQ: tl.constexpr = META[6]
    HK: tl.constexpr = META[7]
    SQ: tl.constexpr = META[8]
    SK: tl.constexpr = META[9]
    SV: tl.constexpr = META[10]
    PQ: tl.constexpr = META[11]
    PK: tl.constexpr = META[12]
    PV: tl.constexpr = META[13]
    pid = tl.program_id(0)
    which = tl.where(pid < PQ, 0, tl.where(pid < PQ + PK, 1, 2))
    local = tl.where(
        which == 0, pid, tl.where(which == 1, pid - PQ, pid - PQ - PK)
    )
    step = tl.where(which == 0, PQ, tl.where(which == 1, PK, PV))
    X = Q
    W = WQ
    groups = GQ
    if KV:
        if which == 1:
            X = K
            W = WK
        if which == 2:
            X = V
        groups = tl.where(which == 0, GQ, GK)
    c = tl.arange(0, D)
    w = tl.load(W + c).to(tl.float32)
    w = tl.where(which == 2, 1.0, w)
    for group in range(local, groups, step):
        r = group * R + tl.arange(0, R)
        qo = r[:, None] // HQ * SQ + r[:, None] % HQ * D + c[None, :]
        if KV:
            ko = r[:, None] // HK * SK + r[:, None] % HK * D + c[None, :]
            vo = r[:, None] // HK * SV + r[:, None] % HK * D + c[None, :]
            offset = tl.where(which == 0, qo, tl.where(which == 1, ko, vo))
        else:
            offset = qo
        x = tl.load(X + offset, offset_state_policy="continuous").to(
            tl.float32
        )
        y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E) * w[None, :]
        tl.store(X + offset, y, offset_state_policy="continuous")


@triton.jit
def _packed(Q, WQ, WK, META: tl.constexpr):
    M: tl.constexpr = META[0]
    HQ: tl.constexpr = META[1]
    HK: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    E: tl.constexpr = META[4]
    R: tl.constexpr = META[5]
    H: tl.constexpr = HQ + 2 * HK
    c = tl.arange(0, D)
    wq = tl.load(WQ + c).to(tl.float32)
    wk = tl.load(WK + c).to(tl.float32)
    for group in range(tl.program_id(0), M * H // R, tl.num_programs(0)):
        r = group * R + tl.arange(0, R)
        h = r % H
        off = r[:, None] * D + c[None, :]
        x = tl.load(Q + off, offset_state_policy="continuous").to(tl.float32)
        w = tl.where(
            h[:, None] < HQ,
            wq[None, :],
            tl.where(h[:, None] < HQ + HK, wk[None, :], 1.0),
        )
        y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E) * w
        tl.store(Q + off, y, offset_state_policy="continuous")


@triton.jit
def _mixed(Q, K, V, WQ, WK, META: tl.constexpr):
    HQ: tl.constexpr = META[1]
    HK: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    SQ: tl.constexpr = META[4]
    SK: tl.constexpr = META[5]
    SV: tl.constexpr = META[6]
    WQS: tl.constexpr = META[7]
    WKS: tl.constexpr = META[8]
    E: tl.constexpr = META[9]
    B: tl.constexpr = META[10]
    for head in range(HQ + 2 * HK):
        token = tl.program_id(0)
        p = head
        c = tl.arange(0, B)
        qr = p
        kr = p - HQ
        vr = p - (HQ + HK)
        qo = token * SQ + qr % HQ * D + c
        ko = token * SK + kr % HK * D + c
        vo = token * SV + vr % HK * D + c
        sc = tl.minimum(c, D - 1)
        sqr = tl.minimum(qr, HQ - 1)
        skr = tl.minimum(tl.maximum(kr, 0), HK - 1)
        svr = tl.maximum(vr, 0)
        qx = tl.load(Q + token * SQ + sqr % HQ * D + sc).to(tl.float32)
        kx = tl.load(K + token * SK + skr % HK * D + sc).to(tl.float32)
        vx = tl.load(V + token * SV + svr % HK * D + sc).to(tl.float32)
        qw = tl.load(WQ + sc * WQS).to(tl.float32)
        kw = tl.load(WK + sc * WKS).to(tl.float32)
        x = tl.where(p < HQ, qx, tl.where(p < (HQ + HK), kx, vx))
        x = tl.where(c < D, x, 0.0)
        w = tl.where(p < HQ, qw, tl.where(p < (HQ + HK), kw, 1.0))
        y = x * tl.rsqrt(tl.sum(x * x, 0) / D + E) * w
        if p < HQ:
            tl.store(Q + qo, y, c < D)
        elif p < (HQ + HK):
            tl.store(K + ko, y, c < D)
        else:
            tl.store(V + vo, y, c < D)


@triton.jit
def _small_exact(Q, K, V, WQ, WK, META: tl.constexpr):
    D: tl.constexpr = META[2]
    E: tl.constexpr = META[3]
    KV: tl.constexpr = META[4]
    R: tl.constexpr = META[5]
    HQ: tl.constexpr = META[6]
    HK: tl.constexpr = META[7]
    SQ: tl.constexpr = META[8]
    SK: tl.constexpr = META[9]
    SV: tl.constexpr = META[10]
    PQ: tl.constexpr = META[11]
    PK: tl.constexpr = META[12]
    pid = tl.program_id(0)
    which = tl.where(pid < PQ, 0, tl.where(pid < PQ + PK, 1, 2))
    local = tl.where(
        which == 0, pid, tl.where(which == 1, pid - PQ, pid - PQ - PK)
    )
    X = Q
    W = WQ
    if KV:
        if which == 1:
            X = K
            W = WK
        if which == 2:
            X = V
    c = tl.arange(0, D)
    w = tl.load(W + c).to(tl.float32)
    w = tl.where(which == 2, 1.0, w)
    group = local
    r = group * R + tl.arange(0, R)
    qo = r[:, None] // HQ * SQ + r[:, None] % HQ * D + c[None, :]
    if KV:
        ko = r[:, None] // HK * SK + r[:, None] % HK * D + c[None, :]
        vo = r[:, None] // HK * SV + r[:, None] % HK * D + c[None, :]
        offset = tl.where(which == 0, qo, tl.where(which == 1, ko, vo))
    else:
        offset = qo
    x = tl.load(X + offset, offset_state_policy="continuous").to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E) * w[None, :]
    tl.store(X + offset, y, offset_state_policy="continuous")


def gemma_qkv_rmsnorm(
    q, k, v, q_weight, k_weight, num_q_heads, num_kv_heads, head_dim, eps
):
    m = q.shape[0]
    kv = k is not None and v is not None
    groups = m * (num_q_heads + (2 * num_kv_heads if kv else 0))
    if (
        groups
        and kv
        and (
            q.dtype != k.dtype
            or q.dtype != v.dtype
            or q_weight.dtype != k_weight.dtype
        )
    ):
        meta = (
            m,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q.stride()[0],
            k.stride()[0],
            v.stride()[0],
            q_weight.stride()[0],
            k_weight.stride()[0],
            eps,
            1 << (head_dim - 1).bit_length(),
        )
        _mixed.run(
            q,
            k,
            v,
            q_weight,
            k_weight,
            meta,
            grid=(m,),
            warmup=False,
            isCloseMemoryAsync=False,
        )
        return (q, k, v)
    if (
        m
        and m <= 8
        and 64 <= head_dim <= 256
        and kv
        and num_q_heads > 0
        and num_kv_heads > 0
        and head_dim == 1 << (head_dim - 1).bit_length()
        and q_weight.stride()[0] == k_weight.stride()[0] == 1
    ):
        rows = min(64, groups & -groups)
        meta = (
            m,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q.stride()[0],
            k.stride()[0],
            v.stride()[0],
            eps,
            rows,
        )
        _small_combined.run(
            q,
            k,
            v,
            q_weight,
            k_weight,
            meta,
            groups_per_cluster=rows,
            isCloseUnrollControl=True,
            isCloseMemoryAsync=False,
            grid=(groups // rows,),
            warmup=False,
        )
        return (q, k, v)
    if (
        m
        and num_q_heads > 0
        and (not kv or num_kv_heads > 0)
        and (m * num_q_heads <= 32)
        and (not kv or m * num_kv_heads <= 32)
        and (head_dim == 1 << (head_dim - 1).bit_length())
        and (q_weight.stride()[0] == 1)
        and (not kv or k_weight.stride()[0] == 1)
    ):
        n = m * num_q_heads | (m * num_kv_heads if kv else 0)
        rows = min(64, n & -n)
        if m * (num_q_heads + (2 * num_kv_heads if kv else 0)) // rows <= 24:
            gq = m * num_q_heads // rows
            gk = m * num_kv_heads // rows if kv else 0
            pq, pk, pv = (gq, gk, gk)
            meta = (
                gq,
                gk,
                head_dim,
                eps,
                kv,
                rows,
                num_q_heads,
                num_kv_heads,
                q.stride()[0],
                k.stride()[0] if kv else 0,
                v.stride()[0] if kv else 0,
                pq,
                pk,
                pv,
            )
            _small_exact.run(
                q,
                k if kv else None,
                v if kv else None,
                q_weight,
                k_weight if kv else None,
                meta,
                groups_per_cluster=rows,
                isCloseUnrollControl=True,
                isCloseMemoryAsync=False,
                grid=(pq + pk + pv,),
                warmup=False,
            )
            return (q, k, v)
    fast = (
        m * num_q_heads % 64 == 0
        and head_dim == 1 << (head_dim - 1).bit_length()
        and q.is_contiguous()
        and (q_weight.stride()[0] == 1)
    )
    if kv:
        fast = (
            fast
            and m * num_kv_heads % 64 == 0
            and k.is_contiguous()
            and v.is_contiguous()
            and (k_weight.stride()[0] == 1)
        )
    packed = False
    if (
        not fast
        and kv
        and m
        and (m * (num_q_heads + 2 * num_kv_heads) % 64 == 0)
        and (head_dim == 1 << (head_dim - 1).bit_length())
        and (q.dtype == k.dtype == v.dtype)
        and (q_weight.stride()[0] == k_weight.stride()[0] == 1)
    ):
        width = (num_q_heads + 2 * num_kv_heads) * head_dim
        packed = (
            q.stride()[0] == k.stride()[0] == v.stride()[0] == width
            and q.stride()[1] == k.stride()[1] == v.stride()[1] == 1
        )
        packed = (
            packed
            and k.data_ptr()
            == q.data_ptr() + num_q_heads * head_dim * q.element_size()
            and (
                v.data_ptr()
                == k.data_ptr() + num_kv_heads * head_dim * k.element_size()
            )
        )
    if packed:
        _packed.run(
            q,
            q_weight,
            k_weight,
            (m, num_q_heads, num_kv_heads, head_dim, eps, 64),
            groups_per_cluster=64,
            isCloseUnrollControl=True,
            isCloseMemoryAsync=False,
            grid=(min(m * (num_q_heads + 2 * num_kv_heads) // 64, 12),),
            warmup=False,
        )
    elif groups and fast:
        gq = m * num_q_heads // 64
        gk = m * num_kv_heads // 64 if kv else 0
        pq = max(1, min(10, 12 * gq // (gq + 2 * gk))) if kv else min(gq, 12)
        pk = max(1, (12 - pq) // 2) if kv else 0
        pv = 12 - pq - pk if kv else 0
        _grouped.run(
            q,
            k if kv else None,
            v if kv else None,
            q_weight,
            k_weight if kv else None,
            (gq, gk, head_dim, eps, kv, 64, pq, pk, pv),
            groups_per_cluster=64,
            isCloseUnrollControl=True,
            isCloseMemoryAsync=False,
            grid=(pq + pk + pv,),
            warmup=False,
        )
    elif (
        groups
        and head_dim == 1 << (head_dim - 1).bit_length()
        and (num_q_heads % 2 == 0)
        and (not kv or num_kv_heads % 2 == 0)
        and (q_weight.stride()[0] == 1)
        and (not kv or k_weight.stride()[0] == 1)
    ):
        sr = (
            64
            if m * num_q_heads % 64 == 0
            and (not kv or m * num_kv_heads % 64 == 0)
            else 2
        )
        gq = m * num_q_heads // sr
        gk = m * num_kv_heads // sr if kv else 0
        pq = max(1, min(10, 12 * gq // (gq + 2 * gk))) if kv else min(gq, 12)
        pk = max(1, (12 - pq) // 2) if kv else 0
        pv = 12 - pq - pk if kv else 0
        _strided.run(
            q,
            k if kv else None,
            v if kv else None,
            q_weight,
            k_weight if kv else None,
            (
                gq,
                gk,
                head_dim,
                eps,
                kv,
                sr,
                num_q_heads,
                num_kv_heads,
                q.stride()[0],
                k.stride()[0] if kv else 0,
                v.stride()[0] if kv else 0,
                pq,
                pk,
                pv,
            ),
            groups_per_cluster=sr,
            isCloseUnrollControl=True,
            isCloseMemoryAsync=False,
            grid=(pq + pk + pv,),
            warmup=False,
        )
    elif groups:
        _kernel.run(
            q,
            k if kv else None,
            v if kv else None,
            q_weight,
            k_weight if kv else None,
            (
                m,
                num_q_heads,
                num_kv_heads,
                head_dim,
                q.stride()[0],
                k.stride()[0] if kv else 0,
                v.stride()[0] if kv else 0,
                q_weight.stride()[0],
                k_weight.stride()[0] if kv else 0,
                eps,
                kv,
                1 << (head_dim - 1).bit_length(),
            ),
            isCloseMemoryAsync=False,
            grid=(groups,),
            warmup=False,
        )
    return (q, k, v)


__all__ = ["gemma_qkv_rmsnorm"]
