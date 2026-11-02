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
def _ba_copy(
    row2,
    nrows,
    ba_ptr,
    b_ptr,
    a_ptr,
    NQK: tl.constexpr,
    NV: tl.constexpr,
    R: tl.constexpr,
):
    """ba 拷贝: (RB, NQK*R) 的 2D perm tile."""
    c = tl.arange(0, NQK * R)[None, :]
    m = row2 < nrows
    base = row2 * (NQK * 2 * R) + (c // R) * (2 * R) + (c % R)
    xb = tl.load(ba_ptr + base, mask=m, other=0)
    xa = tl.load(ba_ptr + base + R, mask=m, other=0)
    o2 = row2 * NV + c
    tl.store(b_ptr + o2, xb, mask=m)
    tl.store(a_ptr + o2, xa, mask=m)


@triton.jit
def _qkvzba_kernel_row(
    qkvz_ptr,
    ba_ptr,
    qkv_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    NQK: tl.constexpr,
    W: tl.constexpr,
    HQK: tl.constexpr,
    RV: tl.constexpr,
    QKVD: tl.constexpr,
    NVHV: tl.constexpr,
    NV: tl.constexpr,
    R: tl.constexpr,
):
    """小 batch 路径: 每 program 精确处理 1 行, 2D tile (NQK, HQK/RV)."""
    row = tl.program_id(0)
    h = tl.arange(0, NQK)[:, None]
    jq = tl.arange(0, HQK)[None, :]
    jv = tl.arange(0, RV)[None, :]
    p = row * NQK + h

    xq = tl.load(qkvz_ptr + p * W + jq)
    tl.store(qkv_ptr + row * QKVD + h * HQK + jq, xq)
    xk = tl.load(qkvz_ptr + p * W + HQK + jq)
    tl.store(qkv_ptr + row * QKVD + NQK * HQK + h * HQK + jq, xk)
    xv = tl.load(qkvz_ptr + p * W + 2 * HQK + jv)
    tl.store(qkv_ptr + row * QKVD + 2 * NQK * HQK + h * RV + jv, xv)
    xz = tl.load(qkvz_ptr + p * W + 2 * HQK + RV + jv)
    tl.store(z_ptr + row * NVHV + h * RV + jv, xz)

    c = tl.arange(0, NQK * R)[None, :]
    base = row * (NQK * 2 * R) + (c // R) * (2 * R) + (c % R)
    xb = tl.load(ba_ptr + base)
    xa = tl.load(ba_ptr + base + R)
    o2 = row * NV + c
    tl.store(b_ptr + o2, xb)
    tl.store(a_ptr + o2, xa)


@triton.jit
def _qkvzba_kernel(
    qkvz_ptr,
    ba_ptr,
    qkv_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    nrows,
    NQK: tl.constexpr,
    W: tl.constexpr,
    HQK: tl.constexpr,
    RV: tl.constexpr,
    QKVD: tl.constexpr,
    NVHV: tl.constexpr,
    NV: tl.constexpr,
    R: tl.constexpr,
    RB: tl.constexpr,
):
    """EVEN 路径: nqk/hqk/r*hv/r 全为 2 的幂, j 维无谓词 (仅 row mask)."""
    pid = tl.program_id(0)
    i = tl.arange(0, RB)[:, None, None]
    h = tl.arange(0, NQK)[None, :, None]
    jq = tl.arange(0, HQK)[None, None, :]
    jv = tl.arange(0, RV)[None, None, :]
    row = pid * RB + i
    p = row * NQK + h
    m = row < nrows

    xq = tl.load(qkvz_ptr + p * W + jq, mask=m, other=0)
    tl.store(qkv_ptr + row * QKVD + h * HQK + jq, xq, mask=m)
    xk = tl.load(qkvz_ptr + p * W + HQK + jq, mask=m, other=0)
    tl.store(qkv_ptr + row * QKVD + NQK * HQK + h * HQK + jq, xk, mask=m)
    xv = tl.load(qkvz_ptr + p * W + 2 * HQK + jv, mask=m, other=0)
    tl.store(qkv_ptr + row * QKVD + 2 * NQK * HQK + h * RV + jv, xv, mask=m)
    xz = tl.load(qkvz_ptr + p * W + 2 * HQK + RV + jv, mask=m, other=0)
    tl.store(z_ptr + row * NVHV + h * RV + jv, xz, mask=m)

    _ba_copy(
        pid * RB + tl.arange(0, RB)[:, None],
        nrows,
        ba_ptr,
        b_ptr,
        a_ptr,
        NQK,
        NV,
        R,
    )


@triton.jit
def _qkvzba_kernel_masked(
    qkvz_ptr,
    ba_ptr,
    qkv_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    nrows,
    NQK: tl.constexpr,
    PNQK: tl.constexpr,
    W: tl.constexpr,
    HQK: tl.constexpr,
    PHK: tl.constexpr,
    RV: tl.constexpr,
    PRV: tl.constexpr,
    QKVD: tl.constexpr,
    NVHV: tl.constexpr,
    NV: tl.constexpr,
    R: tl.constexpr,
    RB: tl.constexpr,
):
    """通用兜底: 段宽 pad 到 2 的幂 + mask 收尾 (任意 head 配置)."""
    pid = tl.program_id(0)
    i = tl.arange(0, RB)[:, None, None]
    h = tl.arange(0, PNQK)[None, :, None]
    jq = tl.arange(0, PHK)[None, None, :]
    jv = tl.arange(0, PRV)[None, None, :]
    row = pid * RB + i
    p = row * PNQK + h
    mrow = row < nrows
    mh = mrow & (h < NQK)
    mq = mh & (jq < HQK)
    mv = mh & (jv < RV)

    xq = tl.load(qkvz_ptr + p * W + jq, mask=mq, other=0)
    tl.store(qkv_ptr + row * QKVD + h * HQK + jq, xq, mask=mq)
    xk = tl.load(qkvz_ptr + p * W + HQK + jq, mask=mq, other=0)
    tl.store(qkv_ptr + row * QKVD + NQK * HQK + h * HQK + jq, xk, mask=mq)
    xv = tl.load(qkvz_ptr + p * W + 2 * HQK + jv, mask=mv, other=0)
    tl.store(qkv_ptr + row * QKVD + 2 * NQK * HQK + h * RV + jv, xv, mask=mv)
    xz = tl.load(qkvz_ptr + p * W + 2 * HQK + RV + jv, mask=mv, other=0)
    tl.store(z_ptr + row * NVHV + h * RV + jv, xz, mask=mv)

    _ba_copy(
        pid * RB + tl.arange(0, RB)[:, None],
        nrows,
        ba_ptr,
        b_ptr,
        a_ptr,
        NQK,
        NV,
        R,
    )


# kernel 变体 id: 0 = 3D even (带 row mask), 1 = masked 兜底, 2 = 2D row-per-
# program.
_V_EVEN = 0
_V_MASKED = 1
_V_ROW = 2

# RB (每 program 行数) / num_warps 分档, 按 batch 升序; 上限含等号.
# 承接 v1/v2/v3 实测标定 (do_bench): 大 batch RB2 w8 已达带宽平台.
_QKVZBA_TIERS = (
    (512, 2, 4),
    (1 << 62, 2, 8),
)

# 小 batch row kernel 的 num_warps 分档 (b <= 上限含等号):
# 实测 do_bench b=1 w2 37.5us, b=8 w4 37.4us, b>=64 w2/w8 持平 (~43us).
_ROW_TIERS = (
    (1, 2),
    (8, 4),
    (256, 4),
)


def _geometry(num_heads_qk, num_heads_v, head_qk, head_v):
    """几何预计算: even 判定 + pad 后段宽 (host 侧每调用都要用)."""
    hqk = head_qk
    hv = head_v
    r = num_heads_v // num_heads_qk
    rv = r * hv
    w = 2 * hqk + 2 * rv
    qkvd = num_heads_qk * (2 * hqk + rv)
    pnqk = triton.next_power_of_2(max(num_heads_qk, 1))
    phk = triton.next_power_of_2(max(hqk, 1))
    prv = triton.next_power_of_2(max(rv, 1))
    even = num_heads_qk == pnqk and hqk == phk and rv == prv
    return r, rv, w, qkvd, pnqk, phk, prv, even


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
):
    b = mixed_qkvz.shape[0]

    hqk = head_qk
    hv = head_v
    r, rv, w, qkvd, pnqk, phk, prv, even = _geometry(
        num_heads_qk, num_heads_v, hqk, hv
    )
    nv = num_heads_v

    if not mixed_qkvz.is_contiguous():
        mixed_qkvz = mixed_qkvz.contiguous()
    if not mixed_ba.is_contiguous():
        mixed_ba = mixed_ba.contiguous()

    dt_in, dt_ba = mixed_qkvz.dtype, mixed_ba.dtype

    mixed_qkv = torch.empty((b, qkvd), dtype=dt_in, device=mixed_qkvz.device)
    z = torch.empty((b, nv, hv), dtype=dt_in, device=mixed_qkvz.device)
    b_out = torch.empty((b, nv), dtype=dt_ba, device=mixed_ba.device)
    a_out = torch.empty((b, nv), dtype=dt_ba, device=mixed_ba.device)

    if b == 0 or (w == 0 and mixed_ba.shape[-1] == 0):
        return mixed_qkv, z, b_out, a_out

    # kernel 变体选择: even 且 b 足够小时走 row kernel (无 mask, tile 更小);
    # 其余小/中 batch 走 even 3D; 非幂几何只能走 masked 兜底.
    if b > 256 or not even:
        variant = _V_MASKED if not even else _V_EVEN
        rb, nw = (2, 4)
        for lim, rb_t, nw_t in _QKVZBA_TIERS:
            if b <= lim:
                rb, nw = rb_t, nw_t
                break
        grid = (triton.cdiv(b, rb),)
    else:
        variant = _V_ROW
        rb = 1
        nw = 4
        for lim, nw_t in _ROW_TIERS:
            if b <= lim:
                nw = nw_t
                break
        grid = (b,)

    if variant == _V_ROW:
        _qkvzba_kernel_row[grid](
            mixed_qkvz,
            mixed_ba,
            mixed_qkv,
            z,
            b_out,
            a_out,
            NQK=num_heads_qk,
            W=w,
            HQK=hqk,
            RV=rv,
            QKVD=qkvd,
            NVHV=nv * hv,
            NV=nv,
            R=r,
            num_warps=nw,
            num_stages=1,
            debug=False,
        )
        return mixed_qkv, z, b_out, a_out
    kwargs = dict(
        NQK=num_heads_qk,
        W=w,
        HQK=hqk,
        RV=rv,
        QKVD=qkvd,
        NVHV=nv * hv,
        NV=nv,
        R=r,
        RB=rb,
        num_warps=nw,
        num_stages=1,
        debug=False,
    )
    if variant == _V_EVEN:
        _qkvzba_kernel[grid](
            mixed_qkvz, mixed_ba, mixed_qkv, z, b_out, a_out, b, **kwargs
        )
    else:
        kwargs.update(PNQK=pnqk, PHK=phk, PRV=prv)
        _qkvzba_kernel_masked[grid](
            mixed_qkvz, mixed_ba, mixed_qkv, z, b_out, a_out, b, **kwargs
        )
    return mixed_qkv, z, b_out, a_out


__all__ = ["fused_qkvzba_split_reshape_cat"]
