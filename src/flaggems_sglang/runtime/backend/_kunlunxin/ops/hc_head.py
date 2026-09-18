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


# ---------------------------------------------------------------------------
# Kernel 1a: partial reduce  (T, ND) -> partials [T, ND, 5]
# ---------------------------------------------------------------------------
@triton.jit
def _hc_head_reduce_partial(
    x_ptr,  # [T, hc_mult, hidden]   (input dtype; contiguous over hidden)
    hc_fn_ptr,  # [hc_mult, D]           float32, contiguous over D
    partial_ptr,  # [T, ND, 5]             float32 scratch
    T,
    HIDDEN,
    D,  # = HC_MULT * HIDDEN
    HC_MULT: tl.constexpr,
    stride_xt,  # x stride on the T axis (in elements) = HC_MULT * HIDDEN
    ND: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_cols < D

    # One token's D-row; x is contiguous over (hc_mult, hidden) so the flat
    # row begins at pid_t * stride_xt.
    x_tile = tl.load(
        x_ptr + pid_t * stride_xt + d_cols, mask=d_mask, other=0.0
    ).to(tl.float32)

    ss = tl.sum(x_tile * x_tile)
    fn0 = tl.load(hc_fn_ptr + 0 * D + d_cols, mask=d_mask, other=0.0)
    a0 = tl.sum(x_tile * fn0)
    fn1 = tl.load(hc_fn_ptr + 1 * D + d_cols, mask=d_mask, other=0.0)
    a1 = tl.sum(x_tile * fn1)
    if HC_MULT == 4:
        fn2 = tl.load(hc_fn_ptr + 2 * D + d_cols, mask=d_mask, other=0.0)
        a2 = tl.sum(x_tile * fn2)
        fn3 = tl.load(hc_fn_ptr + 3 * D + d_cols, mask=d_mask, other=0.0)
        a3 = tl.sum(x_tile * fn3)

    pbase = (pid_t * ND + pid_d) * 5
    tl.store(partial_ptr + pbase + 0, ss)
    tl.store(partial_ptr + pbase + 1, a0)
    tl.store(partial_ptr + pbase + 2, a1)
    if HC_MULT == 4:
        tl.store(partial_ptr + pbase + 3, a2)
        tl.store(partial_ptr + pbase + 4, a3)


# ---------------------------------------------------------------------------
# Kernel 1b: finalise  partials [T, ND, 5] -> pre [T, hc_mult]
# ---------------------------------------------------------------------------
@triton.jit
def _hc_head_reduce_finalize(
    partial_ptr,  # [T, ND, 5]             float32
    hc_scale_ptr,  # [1]                   float32
    hc_base_ptr,  # [hc_mult]             float32
    pre_ptr,  # [T, hc_mult]          float32
    T,
    HC_MULT: tl.constexpr,
    D,  # = HC_MULT * HIDDEN (used for the RMS mean)
    NORM_EPS,  # float32 scalar
    HC_EPS,  # float32 scalar
    stride_pret,  # pre stride on the T axis (in elements) = HC_MULT
    ND: tl.constexpr,
    BLOCK_ND: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_ND)
    mask = offs < ND
    pbase = (pid * ND + offs) * 5

    ss = tl.load(partial_ptr + pbase + 0, mask=mask, other=0.0)
    a0 = tl.load(partial_ptr + pbase + 1, mask=mask, other=0.0)
    a1 = tl.load(partial_ptr + pbase + 2, mask=mask, other=0.0)
    ss_t = tl.sum(ss)
    a0_t = tl.sum(a0)
    a1_t = tl.sum(a1)
    if HC_MULT == 4:
        a2 = tl.load(partial_ptr + pbase + 3, mask=mask, other=0.0)
        a3 = tl.load(partial_ptr + pbase + 4, mask=mask, other=0.0)
        a2_t = tl.sum(a2)
        a3_t = tl.sum(a3)

    r = tl.rsqrt(ss_t / D + NORM_EPS)
    scale = tl.load(hc_scale_ptr)
    b0 = tl.load(hc_base_ptr + 0)
    b1 = tl.load(hc_base_ptr + 1)
    pre_row = pid * stride_pret
    tl.store(pre_ptr + pre_row + 0, tl.sigmoid(a0_t * r * scale + b0) + HC_EPS)
    tl.store(pre_ptr + pre_row + 1, tl.sigmoid(a1_t * r * scale + b1) + HC_EPS)
    if HC_MULT == 4:
        b2 = tl.load(hc_base_ptr + 2)
        b3 = tl.load(hc_base_ptr + 3)
        tl.store(
            pre_ptr + pre_row + 2, tl.sigmoid(a2_t * r * scale + b2) + HC_EPS
        )
        tl.store(
            pre_ptr + pre_row + 3, tl.sigmoid(a3_t * r * scale + b3) + HC_EPS
        )


# ---------------------------------------------------------------------------
# Kernel 2: fold  pre[T, hc_mult] + x[T, hc_mult, hidden] -> y[T, hidden]
# ---------------------------------------------------------------------------
# NOTE on the h-tile: this XPU Triton backend miscompiles the 2-D masked
# x/y store for BLOCK_H in {256, 512, 4096}. We therefore restrict BLOCK_H to
# {1024, 2048}: for hidden <= 1024 we use 1024 (a single padded h-block, whose
# all-true mask is fine at 1024); for larger hidden we use 2048 (multi-block).
# BLOCK_H == hidden with hidden a power of two would also be all-true; we
# avoid that by never picking BLOCK_H == hidden (the bench hidden 7168 is not a
# power of two and always multi-block, and the small cases use 1024 < 128 is
# impossible, but for hidden=512 we pick 1024 > hidden -- a padded single
# block, which is all-true and known-good for 1024).
@triton.jit
def _hc_head_fold(
    x_ptr,  # [T, hc_mult, hidden]
    pre_ptr,  # [T, hc_mult]   float32
    y_ptr,  # [T, hidden]
    T,
    HIDDEN: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Use *constexpr* strides for all address arithmetic. On this XPU Triton
    # backend, computing the 2D x/y index with a *runtime* stride scalar
    # miscompiles for several BLOCK_H values; using the constexpr
    # ``HC_MULT * HIDDEN`` / ``HIDDEN`` (both compile-time) lowers cleanly.
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < HIDDEN

    D = HC_MULT * HIDDEN  # constexpr
    # Unroll the fold over hc_mult (constexpr): each column j contributes
    # pre[:, j] * x[:, j, h_offs]. The per-token gate for column j is loaded
    # as a 1D [BLOCK_T] vector (this build does not support indexing a 2D
    # gate tile with a constexpr column).
    acc = tl.zeros([BLOCK_T, BLOCK_H], dtype=tl.float32)
    for j in tl.static_range(0, HC_MULT):
        pre_j = tl.load(
            pre_ptr + t_offs * HC_MULT + j, mask=t_mask, other=0.0
        )  # [BLOCK_T] float32
        x_idx = t_offs[:, None] * D + j * HIDDEN + h_offs[None, :]
        x_tile = tl.load(
            x_ptr + x_idx, mask=t_mask[:, None] & h_mask[None, :], other=0.0
        ).to(tl.float32)
        acc += pre_j[:, None] * x_tile

    out = acc.to(y_ptr.dtype.element_ty)
    y_idx = t_offs[:, None] * HIDDEN + h_offs[None, :]
    tl.store(y_ptr + y_idx, out, mask=t_mask[:, None] & h_mask[None, :])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _reduce_block_d(D):
    """Pick the D-tile width for the partial reduction.

    On this XPU Triton backend ``BLOCK_D`` of 8192 compiles correctly and
    gives the best trade-off between parallelism (more tiles -> more
    programs) and the constant ``hc_fn`` re-read each program does (fewer
    tiles -> fewer re-reads). ``BLOCK_D`` of 16384 compiles but mis-reads the
    tile; ``BLOCK_D`` of 4096 is correct but slower. We cap at 8192 and force
    at least 2 tiles (``ND >= 2``) so the masked 1-D load always has at least
    one masked-off lane.
    """
    BD = min(8192, _next_pow2(D))
    while BD >= D:
        BD //= 2
    # guarantee >= 2 tiles (avoid the all-true-tile quirk)
    if BD >= D:
        BD = max(1, D // 2)
    return BD


def _fold_config(T, hidden):
    """Pick (BLOCK_T, BLOCK_H, num_warps) for the fold kernel.

    BLOCK_H is restricted to the codegen-safe set {1024, 2048} (this XPU
    backend miscompiles the 2-D masked x/y store for BLOCK_H in {256, 512,
    4096}). BLOCK_T is a small row tile.
    """
    if hidden >= 2048:
        BLOCK_H = 2048
        num_warps = 8
    else:
        BLOCK_H = 1024
        num_warps = 4

    if T >= 16:
        BLOCK_T = 16
    elif T >= 4:
        BLOCK_T = 4
    elif T >= 2:
        BLOCK_T = 2
    else:
        BLOCK_T = 1
    return BLOCK_T, BLOCK_H, num_warps


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    shape, dtype = x.size(), x.dtype
    T, hc_mult, hidden = shape
    D = hc_mult * hidden

    # hc_mult is a power of two (2 or 4 in every provided case).
    hc_mult_c = _next_pow2(hc_mult)
    assert hc_mult_c in (2, 4), f"unsupported hc_mult {hc_mult}"

    # ---- reduce: 2-stage (partial across D-tiles, then finalise) ----
    BLOCK_D = _reduce_block_d(D)
    ND = (D + BLOCK_D - 1) // BLOCK_D
    BLOCK_ND = _next_pow2(ND)

    partial = torch.empty((T, ND, 5), dtype=torch.float32, device=x.device)
    pre = torch.empty((T, hc_mult), dtype=torch.float32, device=x.device)
    y = torch.empty((T, hidden), dtype=dtype, device=x.device)

    stride_xt = x.stride(0)
    stride_pret = hc_mult  # pre is contiguous [T, hc_mult]

    _hc_head_reduce_partial[(T, ND)](
        x,
        hc_fn,
        partial,
        T,
        hidden,
        D,
        hc_mult_c,
        stride_xt,
        ND,
        BLOCK_D,
        num_warps=8,
        num_stages=2,
    )
    _hc_head_reduce_finalize[(T,)](
        partial,
        hc_scale,
        hc_base,
        pre,
        T,
        hc_mult_c,
        D,
        float(norm_eps),
        float(hc_eps),
        stride_pret,
        ND,
        BLOCK_ND,
        num_warps=4,
        num_stages=2,
    )

    # ---- fold: weighted reduction over hc_mult ----
    BLOCK_T, BLOCK_H, num_warps = _fold_config(T, hidden)
    grid_fold = (triton.cdiv(T, BLOCK_T), triton.cdiv(hidden, BLOCK_H))
    _hc_head_fold[grid_fold](
        x,
        pre,
        y,
        T,
        hidden,
        hc_mult_c,
        BLOCK_T=BLOCK_T,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=2,
    )
    return y


__all__ = ["hc_head"]
