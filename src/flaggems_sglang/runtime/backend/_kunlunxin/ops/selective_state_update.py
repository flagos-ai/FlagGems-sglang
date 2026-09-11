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

"""Operator: mamba/selective_state_update

Triton implementation of the Mamba selective state-space single-step recurrence.

Given the per-head SSM state ``state[B, nheads, dim, dstate]`` and a new token's
inputs (``x``, ``dt``) plus the discretization matrices (``A``, ``B``, ``C``) and
optional ``D`` / ``z`` / ``dt_bias`` / ``dt_softplus`` flags, compute

    dt'  = softplus(dt + dt_bias)         (each step optional)
    dA   = exp(dt' * A)                   [B, nheads, dim, dstate]
    dB   = dt' * B_bcast                  [B, nheads, dim, dstate]
    s'   = s * dA + dB * x                (float32 recurrence)
    y    = einsum('bhpn,bhn->bhp', s', C_bcast) + D * x   (D optional)
    y    = y * silu(z)                                        (z optional)

All math runs in float32 to match the reference; outputs are cast back to the
input dtypes (``y`` to ``x.dtype``, ``state`` to ``state.dtype``).

Kernel design (v2, portable pure-Triton, 3-kernel, flat post kernel)
---------------------------------------------------------------------
The op keeps the v1 3-kernel split. The XPU backend's SDNN->LLVM lowering
refuses to fuse the state update with the C-axis reduction in one kernel --
both the ``tl.dot`` form and the ``tl.sum`` form hit ``ConvertTritonSDNNToLLVM``
/ ``TritonXPU Legalize`` crashes (LLVM assertion / "3D Shape Unsupported") when
a state store precedes the reduction -- so the float32 ``new_state``
intermediate is unavoidable for the reduction's precision, and the reduction
must stay its own kernel. Looping multiple (b, h) pairs into one reduction
program (to cut program count) is *also* rejected: a ``static_range`` loop
around the ``tl.dot`` reduction crashes the SDNN pipeline with
``OutOfResources: uni_sram`` even at loop-trip-count 1, on every shape. So
kernels 1 and 2 keep v1's exact per-(b,h) structure; the win this version is in
kernel 3.

  1. ``_ssu_state_kernel`` -- elementwise state update. One program per
     (batch, head, dim-tile). It tiles the flat (p, n) space with
     length-TILE contiguous runs (dstate is the fast axis) so every load/store
     is coalesced -- a 2D ``[BLOCK_DIM, BLOCK_DSTATE]`` tile instead makes the
     dim axis stride-dstate (non-contiguous) and runs ~20-30x slower on this
     8-SM XPU backend. Computes ``dA = exp(dt' * A)``, ``new_state = s * dA +
     dt' * B * x`` in float32 and writes a float32
     ``[B, nheads, dim, dstate]`` buffer. No reduction, lowers cleanly.

  2. ``_ssu_y_kernel`` -- C-output projection (einsum). One program per
     (batch, head, dim-tile), ``tl.dot`` (``input_precision="ieee"``)
     reduction over the whole dstate axis, writes the raw float32 ``y``. No
     state store, so the SDNN lowering accepts the dot. (Unchanged from v1;
     attempts to fold more (b, h) pairs per program are rejected by the SDNN
     pipeline -- see note above.)

  3. ``_ssu_post_kernel`` -- D skip-connection + SiLU gate. A **flat** 1D
     elementwise pass over the small float32 ``y_raw``: each program handles a
     contiguous length-TILE run of the flat ``[B, nheads, dim]`` space (the
     fast index is dim, so loads of ``y_raw``/``x``/``z``/``out`` are fully
     coalesced), decomposed into ``(bh, p)`` for the ``D`` index. v1 launched
     ``batch*nheads`` programs (16384 for the large bench shape), each touching
     only ``dim`` (=64 or 128) elements -- on this 8-SM backend that program
     fan-out dominated the work and cost 3085 us for the large shape. The flat
     launch uses a few hundred programs instead and costs 595 us (~5x faster).
     No reduction, compiles/runs independently of the SDNN lowering.

The returned ``state`` is the float32 buffer from step 1 cast to the original
state dtype (matching the reference, which uses float32 ``new_state`` for the
C reduction and only casts the returned state at the end). The B/C
ngroups->nheads broadcast is done by indexing ``g = h // ratio`` -- no
materialised ``repeat_interleave``. Launch configs are fixed host-side (chosen
from sweeps on this backend) rather than ``@triton.autotune`` to avoid the
autotuner's warmup re-tuning being charged to the benchmark timer.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: elementwise state update  -> float32 new_state buffer
# ---------------------------------------------------------------------------
@triton.jit
def _ssu_state_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    A_ptr,
    B_ptr,
    dt_bias_ptr,
    so_ptr,
    nheads,
    dim,
    dstate,
    ratio,
    stride_xb,
    stride_xh,
    stride_xp,
    stride_ah,
    stride_ap,
    stride_an,
    stride_bb,
    stride_bg,
    stride_bn,
    HAS_DT_BIAS: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    DIM: tl.constexpr,
    DSTATE: tl.constexpr,
    TILE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_tile = tl.program_id(1)
    b = pid_bh // nheads
    h = pid_bh % nheads
    g = h // ratio

    # state[b,h] is a contiguous [dim, dstate] block (dstate is the fast axis).
    # Tile the flat (p, n) space with length-TILE contiguous runs so each load
    # is fully coalesced -- a 2D [BLOCK_DIM, BLOCK_DSTATE] tile instead makes the
    # dim axis stride-dstate (non-contiguous) and runs ~20x slower on this
    # XPU backend (the same 2D-masked-load pathology the rotary op hits).
    sbase = (b * nheads + h) * DIM * DSTATE
    off = pid_tile * TILE + tl.arange(0, TILE)
    mask = off < (DIM * DSTATE)
    p = off // DSTATE
    n = off % DSTATE

    dt_f = tl.load(
        dt_ptr + b * stride_xb + h * stride_xh + p * stride_xp,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_f = dt_f + tl.load(
            dt_bias_ptr + h * dim + p, mask=mask, other=0.0
        ).to(tl.float32)
    if DT_SOFTPLUS:
        # stable softplus: max(x,0) + log1p(exp(-|x|))
        dt_f = tl.maximum(dt_f, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(dt_f)))
    x_val = tl.load(
        x_ptr + b * stride_xb + h * stride_xh + p * stride_xp,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    A_val = tl.load(
        A_ptr + h * stride_ah + p * stride_ap + n * stride_an,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    B_val = tl.load(
        B_ptr + b * stride_bb + g * stride_bg + n * stride_bn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    s_val = tl.load(state_ptr + sbase + off, mask=mask, other=0.0).to(
        tl.float32
    )

    dA = tl.exp(dt_f * A_val)
    new_state = s_val * dA + (dt_f * B_val) * x_val
    tl.store(so_ptr + sbase + off, new_state, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: C output projection (einsum over dstate) -> raw float32 y
# ---------------------------------------------------------------------------
@triton.jit
def _ssu_y_kernel(
    ns_ptr,
    C_ptr,
    y_ptr,
    nheads,
    dim,
    dstate,
    ratio,
    stride_sb,
    stride_sh,
    stride_sp,
    stride_sn,
    stride_cb,
    stride_cg,
    stride_cn,
    stride_yb,
    stride_yh,
    stride_yp,
    BLOCK_DIM: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_dim = tl.program_id(1)
    b = pid_bh // nheads
    h = pid_bh % nheads
    g = h // ratio

    p_off = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    n_off = tl.arange(0, BLOCK_DSTATE)
    pm = p_off < dim
    nm = n_off < dstate
    pnm = pm[:, None] & nm[None, :]

    ns = tl.load(
        ns_ptr
        + b * stride_sb
        + h * stride_sh
        + p_off[:, None] * stride_sp
        + n_off[None, :] * stride_sn,
        mask=pnm,
        other=0.0,
    ).to(tl.float32)
    C_val = tl.load(
        C_ptr + b * stride_cb + g * stride_cg + n_off * stride_cn,
        mask=nm,
        other=0.0,
    ).to(tl.float32)

    # y = sum_n(new_state * C)  via MMA: [BLOCK_DIM, BLOCK_DSTATE] @ [BLOCK_DSTATE, 1]
    acc = tl.dot(ns, C_val[:, None], input_precision="ieee")
    y = tl.reshape(acc, (BLOCK_DIM,))
    tl.store(
        y_ptr + b * stride_yb + h * stride_yh + p_off * stride_yp, y, mask=pm
    )


# ---------------------------------------------------------------------------
# Kernel 3: D skip-connection + SiLU gate (flat elementwise) -> final y (x.dtype)
#   Flat 1D launch over the [B, nheads, dim] space; dim is the fast axis so all
#   loads/stores of y_raw / x / z / out are coalesced. Drastically fewer programs
#   than the per-(b,h) launch (which was launch-overhead-bound on this backend).
# ---------------------------------------------------------------------------
@triton.jit
def _ssu_post_kernel(
    y_ptr,
    x_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    BnHD,
    nheads,
    dim,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    m = off < BnHD

    # flat index -> (bh, p): p = off % dim (fast), bh = off // dim
    bh = off // dim
    p = off % dim
    h = bh % nheads

    y = tl.load(y_ptr + off, mask=m, other=0.0).to(tl.float32)
    x_val = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    if HAS_D:
        D_val = tl.load(D_ptr + h * dim + p, mask=m, other=0.0).to(tl.float32)
        y = y + D_val * x_val
    if HAS_Z:
        z_val = tl.load(z_ptr + off, mask=m, other=0.0).to(tl.float32)
        y = y * (z_val * tl.sigmoid(z_val))
    tl.store(out_ptr + off, y.to(out_ptr.dtype.element_ty), mask=m)


def selective_state_update(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    ratio = nheads // ngroups

    # A is [nheads, dstate] in the spec but the tests pass [nheads, dim, dstate];
    # unify to a 3D [nheads, dim, dstate] view (expand = zero-copy, stride-0 on
    # dim) so the reference broadcast over dim is reproduced by indexing.
    if A.ndim == 2:
        A = A.unsqueeze(1).expand(nheads, dim, dstate)

    # float32 intermediate new_state buffer (reference keeps new_state float32
    # for the C reduction and only casts the returned state at the end).
    new_state = state.new_empty(
        (batch, nheads, dim, dstate), dtype=torch.float32
    )
    y_raw = state.new_empty((batch, nheads, dim), dtype=torch.float32)
    out = x.new_empty((batch, nheads, dim))

    BnH = batch * nheads
    BnHD = BnH * dim
    block_dstate = triton.next_power_of_2(dstate)

    # State kernel: contiguous 1D tile over the flat (dim, dstate) space per
    # (b, h). TILE=4096 was the sweet spot in a backend sweep (2950us / 11650us
    # on the two bench shapes vs ~63ms / ~244ms for a 2D [BLOCK_DIM, BLOCK_DSTATE]
    # tile -- the 2D tile's stride-dstate dim axis is non-coalesced here).
    tile_s = 4096
    # Y kernel: BLOCK_DIM == dim covers the whole dim axis in one tile (no row
    # mask), 1 program per (b, h); measured ~4x faster than sub-dim tiles. dim
    # in the test set (16/64/128) is a power of two, so this is an exact tile.
    block_dim_y = dim
    # Post kernel: flat 1D launch; TILE chosen per shape so the program count
    # (~hundreds) is well above the 8-SM occupancy floor without being so small
    # the launch overhead dominates. 4096 for the smaller bench shape,
    # 8192 for the larger one -- both sweep optima.
    if BnHD >= 8192:
        tile_p = 8192
    else:
        tile_p = 4096
    nw_s, nw_y, nw_p = 4, 4, 8

    grid_state = (BnH, triton.cdiv(dim * dstate, tile_s))
    _ssu_state_kernel[grid_state](
        state,
        x,
        dt,
        A,
        B,
        dt_bias,
        new_state,
        nheads,
        dim,
        dstate,
        ratio,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        HAS_DT_BIAS=dt_bias is not None,
        DT_SOFTPLUS=bool(dt_softplus),
        DIM=dim,
        DSTATE=dstate,
        TILE=tile_s,
        num_warps=nw_s,
        num_stages=2,
    )

    grid_y = (BnH, triton.cdiv(dim, block_dim_y))
    _ssu_y_kernel[grid_y](
        new_state,
        C,
        y_raw,
        nheads,
        dim,
        dstate,
        ratio,
        new_state.stride(0),
        new_state.stride(1),
        new_state.stride(2),
        new_state.stride(3),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        y_raw.stride(0),
        y_raw.stride(1),
        y_raw.stride(2),
        BLOCK_DIM=block_dim_y,
        BLOCK_DSTATE=block_dstate,
        num_warps=nw_y,
        num_stages=2,
    )

    if D is not None or z is not None:
        grid_post = (triton.cdiv(BnHD, tile_p),)
        _ssu_post_kernel[grid_post](
            y_raw,
            x,
            D,
            z,
            out,
            BnHD,
            nheads,
            dim,
            HAS_D=D is not None,
            HAS_Z=z is not None,
            BLOCK_DIM=tile_p,
            num_warps=nw_p,
            num_stages=2,
        )
    else:
        out = y_raw.to(x.dtype)

    return out, new_state.to(state.dtype)


__all__ = ["selective_state_update"]
