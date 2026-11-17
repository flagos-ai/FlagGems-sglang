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

"""recompute_w_u for RDNA3 (gfx1100) (Triton).

``w = A @ (k * beta * exp(g))`` and ``u = A @ (v * beta)``, chunk by chunk.
From 512 chunk-heads on, the generic op on HIP first stages the scaled k, v
and A in packed buffers and multiplies them in a second kernel; on gfx1100
one kernel that computes every tile from the inputs is about twice as fast.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_kernel(
    Kp,
    Vp,
    Bp,
    Gp,
    Ap,
    Wp,
    Up,
    H: tl.constexpr,
    HG: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    INDEX64: tl.constexpr,
):
    # One program per (chunk-head, row block, column block of w or of u).
    q = tl.program_id(0)
    if INDEX64:
        q = q.to(tl.int64)
    tile = tl.program_id(1)
    h = q % H
    t = q // H * BT
    ntiles: tl.constexpr = tl.cdiv(DK, BN) + tl.cdiv(DV, BN)
    ni = tile % ntiles
    mr = tile // ntiles * BM + tl.arange(0, BM)
    rr = tl.arange(0, BR)
    a = tl.load(
        Ap + ((t + mr[:, None]) * H + h) * BT + rr[None, :],
        (mr[:, None] < BT) & (rr[None, :] < BT),
        0,
    )
    beta = tl.load(Bp + (t + rr) * H + h, rr < BT, 0).to(tl.float32)
    is_k = ni < tl.cdiv(DK, BN)
    nn = tl.where(is_k, ni, ni - tl.cdiv(DK, BN)) * BN + tl.arange(0, BN)
    if is_k:
        gate = tl.exp(
            tl.load(Gp + (t + rr) * H + h, rr < BT, 0).to(tl.float32)
        )
        k = tl.load(
            Kp + ((t + rr[:, None]) * HG + h // (H // HG)) * DK + nn[None, :],
            (rr[:, None] < BT) & (nn[None, :] < DK),
            0,
        )
        x = ((k.to(tl.float32) * beta[:, None]) * gate[:, None]).to(k.dtype)
        tl.store(
            Wp + ((t + mr[:, None]) * H + h) * DK + nn[None, :],
            tl.dot(a, x),
            (mr[:, None] < BT) & (nn[None, :] < DK),
        )
    else:
        v = tl.load(
            Vp + ((t + rr[:, None]) * H + h) * DV + nn[None, :],
            (rr[:, None] < BT) & (nn[None, :] < DV),
            0,
        )
        x = (v.to(tl.float32) * beta[:, None]).to(v.dtype)
        tl.store(
            Up + ((t + mr[:, None]) * H + h) * DV + nn[None, :],
            tl.dot(a, x),
            (mr[:, None] < BT) & (nn[None, :] < DV),
        )


@triton.jit
def _strided_kernel(
    Kp,
    Vp,
    Bp,
    Gp,
    Ap,
    Wp,
    Up,
    T: tl.constexpr,
    H: tl.constexpr,
    HG: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    KS3: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    VS3: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BS2: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    AS3: tl.constexpr,
    US0: tl.constexpr,
    US1: tl.constexpr,
    US2: tl.constexpr,
    US3: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    tile = tl.program_id(1)
    q = tl.program_id(0).to(tl.int64)
    h = q % H
    c = q // H % (T // BT)
    b = q // (H * (T // BT))
    mr = tile // tl.cdiv(tl.maximum(DK, DV), BN) * BM + tl.arange(0, BM)
    mr = mr.to(tl.int64)
    nn = tile % tl.cdiv(tl.maximum(DK, DV), BN) * BN + tl.arange(0, BN)
    nn = nn.to(tl.int64)
    rr = tl.arange(0, BR).to(tl.int64)
    a = tl.load(
        Ap
        + b * AS0
        + (c * BT + mr[:, None]) * AS1
        + h * AS2
        + rr[None, :] * AS3,
        (mr[:, None] < BT) & (rr[None, :] < BT),
        0,
    )
    beta = tl.load(
        Bp + b * BS0 + (c * BT + rr) * BS1 + h * BS2, rr < BT, 0
    ).to(tl.float32)
    gate = tl.exp(
        tl.load(Gp + b * GS0 + (c * BT + rr) * GS1 + h * GS2, rr < BT, 0).to(
            tl.float32
        )
    )
    v = tl.load(
        Vp
        + b * VS0
        + (c * BT + rr[:, None]) * VS1
        + h * VS2
        + nn[None, :] * VS3,
        (rr[:, None] < BT) & (nn[None, :] < DV),
        0,
    )
    k = tl.load(
        Kp
        + b * KS0
        + (c * BT + rr[:, None]) * KS1
        + (h // (H // HG)) * KS2
        + nn[None, :] * KS3,
        (rr[:, None] < BT) & (nn[None, :] < DK),
        0,
    )
    vb = (v.to(tl.float32) * beta[:, None]).to(v.dtype)
    kb = ((k.to(tl.float32) * beta[:, None]) * gate[:, None]).to(k.dtype)
    tl.store(
        Wp + ((b * T + c * BT + mr[:, None]) * H + h) * DK + nn[None, :],
        tl.dot(a, kb),
        (mr[:, None] < BT) & (nn[None, :] < DK),
    )
    tl.store(
        Up
        + b * US0
        + (c * BT + mr[:, None]) * US1
        + h * US2
        + nn[None, :] * US3,
        tl.dot(a, vb),
        (mr[:, None] < BT) & (nn[None, :] < DV),
    )


def _compact_config(chunk_heads):
    """(BM, BN, num_warps)."""
    if chunk_heads < 256:
        return 64, 32, 4
    return 64, 64, 8


def recompute_w_u(k, v, beta, g_cumsum, A, cu_seqlens):
    assert cu_seqlens is None
    B, T, Hg, K = k.shape
    H, V = v.shape[-2:]
    BT = A.shape[-1]
    w = torch.empty((B, T, H, K), dtype=k.dtype, device=k.device)
    u = torch.empty_like(v)
    q = B * (T // BT) * H
    if q == 0:
        return w, u
    br = max(16, triton.next_power_of_2(BT))
    if all(t.is_contiguous() for t in (k, v, beta, g_cumsum, A)):
        bm, bn, warps = _compact_config(q)
        tiles = triton.cdiv(K, bn) + triton.cdiv(V, bn)
        _compact_kernel[(q, triton.cdiv(BT, bm) * tiles)](
            k,
            v,
            beta,
            g_cumsum,
            A,
            w,
            u,
            H,
            Hg,
            K,
            V,
            BT,
            bm,
            bn,
            br,
            B * T * H * max(K, V, BT) >= 2**31,
            num_warps=warps,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return w, u
    _strided_kernel[(q, triton.cdiv(BT, 64) * triton.cdiv(max(K, V), 64))](
        k,
        v,
        beta,
        g_cumsum,
        A,
        w,
        u,
        T,
        H,
        Hg,
        K,
        V,
        BT,
        *k.stride(),
        *v.stride(),
        *beta.stride(),
        *g_cumsum.stride(),
        *A.stride(),
        *u.stride(),
        64,
        64,
        br,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return w, u


__all__ = ["recompute_w_u"]
