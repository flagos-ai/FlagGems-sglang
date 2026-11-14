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

"""Triton implementation of mamba/selective_state_update (single-step SSM recurrence).

Given the current per-head SSM state ``state[B, nheads, dim, dstate]`` and one
new token's inputs (``x``, ``dt``, ``A``, ``B``, ``C``, optional ``D``/``z``/
``dt_bias``/``dt_softplus``), update the state and emit the output ``y[B, nheads, dim]``.

Per (b, h, p) row the work is:
    dt'  = softplus(dt + dt_bias)?            : [B, nheads, dim]
    dA   = exp(dt' * A)                       : [B, nheads, dim, dstate]
    dB   = dt' * B_bcast                      : [B, nheads, dim, dstate]
    new_state = state * dA + dB * x           : [B, nheads, dim, dstate]
    y    = sum_n(new_state * C_bcast)         : [B, nheads, dim]
    y   += D * x            (if D)
    y   *= silu(z)          (if z)

All math runs in float32 for accuracy; ``new_state`` is stored directly to the
output buffer in ``state.dtype`` (the fp32 register value is rounded on store,
matching reference's ``new_state.to(state.dtype)``), and ``y`` is produced from
the fp32 register-resident ``new_state`` before the store — so no separate
fp32 scratch buffer and no separate dtype-conversion kernel are needed.

Launch strategy — two paths:

Fast path (``_ssu_khead_kernel``, used when every tensor is contiguous and
``dim``/``dstate`` are powers of two):
  - Grid ``(nheads // K, cdiv(batch, BLOCK_B))``: one program owns ``K``
    consecutive heads and a contiguous slice of the batch axis which it loops
    over with a *static* trip count (``BLOCK_B`` divides ``batch``), so the
    loop body is schedule-friendly and needs no masks.
  - Each program's state tile is ``[K * dim, dstate]``: because ``state`` (and
    ``A``, ``dt``, ``x``, ``z``, ``y``) are laid out with the head axis
    outermost-after-batch, ``K`` consecutive heads of one batch form one
    contiguous row block, so every access stays a plain affine 2D tile with no
    masking. Making the row tile span heads decouples the tile height from
    ``dim``: on GCU the per-iteration tile latency is roughly constant, so a
    ``[256, dstate]`` tile moves ~2x the bytes of a ``[128, dstate]`` tile per
    iteration at the same latency (measured: the dim=64 bench case went from
    1.61 ms at [64,128] tiles to 0.78 ms at [256,128] tiles). ``K`` is chosen
    as the largest power of two with ``K * dim <= 256`` (bounded also by the
    DSM budget for the ~5 live fp32 tiles) that divides both ``nheads`` and
    ``ratio = nheads // ngroups`` — the divisibility by ``ratio`` guarantees
    all ``K`` heads share one B/C group, so ``g`` stays a per-program scalar.
  - Head-constant tiles (``A``, ``D``, ``dt_bias``) are loaded once per
    program and reused across the program's whole batch slice.

Fallback path (``_ssu_kernel``, unchanged): arbitrary strides/shapes via the
masked, dynamically-bounded batch loop with ``@triton.autotune``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fast path: K-head spanning tiles, static (mask-free) batch loop.
# Eligibility is decided on the host (contiguity + power-of-two dims + the
# K/BLOCK_B divisibility below); the kernel itself is fully unmasked.
# ---------------------------------------------------------------------------


@triton.jit
def _ssu_khead_kernel(
    state_ptr,  # [B, NHEADS, DIM, DSTATE] (contiguous)
    x_ptr,  # [B, NHEADS, DIM]            (contiguous)
    dt_ptr,  # [B, NHEADS, DIM]            (contiguous)
    A_ptr,  # [NHEADS, DIM, DSTATE]        (contiguous)
    B_ptr,  # [B, NGROUPS, DSTATE]         (contiguous)
    C_ptr,  # [B, NGROUPS, DSTATE]         (contiguous)
    D_ptr,  # [NHEADS, DIM]   or dummy
    z_ptr,  # [B, NHEADS, DIM] or dummy
    dt_bias_ptr,  # [NHEADS, DIM]  or dummy
    y_ptr,  # [B, NHEADS, DIM]
    new_state_ptr,  # [B, NHEADS, DIM, DSTATE]  (state.dtype)
    NHEADS: tl.constexpr,
    DIM: tl.constexpr,
    DSTATE: tl.constexpr,
    NGROUPS: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    h0 = pid_h * K
    g = h0 // RATIO
    BLOCK_M: tl.constexpr = K * DIM

    m_off = tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, DSTATE)

    # head-constant tiles, loaded once and reused across the batch slice
    A = tl.load(
        A_ptr + (h0 * DIM + m_off[:, None]) * DSTATE + n_off[None, :]
    ).to(tl.float32)
    if HAS_D:
        Dv = tl.load(D_ptr + h0 * DIM + m_off).to(tl.float32)
    if HAS_DT_BIAS:
        db = tl.load(dt_bias_ptr + h0 * DIM + m_off).to(tl.float32)

    for i in range(BLOCK_B):
        b = pid_b * BLOCK_B + i
        rowbase = (b * NHEADS + h0) * DIM
        s_offs = (rowbase + m_off[:, None]) * DSTATE + n_off[None, :]
        s2d = tl.load(state_ptr + s_offs).to(tl.float32)

        dt = tl.load(dt_ptr + rowbase + m_off).to(tl.float32)
        if HAS_DT_BIAS:
            dt = dt + db
        if DT_SOFTPLUS:
            # softplus(x) = log(1 + exp(x)); branch on sign for numerical
            # stability (matches F.softplus up to fp rounding).
            dt = tl.where(
                dt > 0.0,
                dt + tl.log(1.0 + tl.exp(-dt)),
                tl.log(1.0 + tl.exp(dt)),
            )

        x = tl.load(x_ptr + rowbase + m_off).to(tl.float32)
        Bc = tl.load(B_ptr + b * (NGROUPS * DSTATE) + g * DSTATE + n_off).to(
            tl.float32
        )

        # new_state = state * exp(dt'*A) + (dt'*x) * B_bcast
        dA = tl.exp(dt[:, None] * A)
        ns2d = s2d * dA + (dt * x)[:, None] * Bc[None, :]
        tl.store(new_state_ptr + s_offs, ns2d)

        # y = sum_n(new_state * C_bcast), reusing the register-resident
        # new_state (no extra global read of the just-stored values).
        Cc = tl.load(C_ptr + b * (NGROUPS * DSTATE) + g * DSTATE + n_off).to(
            tl.float32
        )
        y = tl.sum(ns2d * Cc[None, :], axis=1)
        if HAS_D:
            y = y + Dv * x
        if HAS_Z:
            z = tl.load(z_ptr + rowbase + m_off).to(tl.float32)
            y = y * (z * tl.sigmoid(z))
        tl.store(y_ptr + rowbase + m_off, y)


# ---------------------------------------------------------------------------
# Autotune configs. BLOCK_P sweeps the ``dim`` axis tile size; BLOCK_B sweeps the
# ``batch`` axis tile size (how many consecutive batch elements one program
# processes). BLOCK_DSTATE is padded to a power of two >= dstate (passed as a
# constexpr, fixed per shape). A compact set keeps compile/autotune budget
# small while covering the bench shapes (dim 64/128, dstate 128, B 64/256) and
# the small correctness shapes.
# ---------------------------------------------------------------------------


def _main_configs():
    configs = []
    # BLOCK_P covers the dim axis (64/128 cover both bench shapes' dim in one
    # tile; 32 covers the dim=128 case if two tiles help occupancy). BLOCK_B
    # covers the batch axis: 1 = whole-batch-in-one-program (best for small B
    # where amortising head-constant loads dominates) up to 32/64 (best for the
    # batch256 case where more programs are needed to saturate bandwidth). A
    # compact, targeted set keeps the autotune compile budget small.
    for bp in (32, 64, 128):
        for bb in (1, 8, 16, 32, 64):
            for nw, ns in ((2, 2), (4, 2), (4, 3), (8, 3), (8, 4)):
                configs.append(
                    triton.Config(
                        {"BLOCK_P": bp, "BLOCK_B": bb},
                        num_warps=nw,
                        num_stages=ns,
                    )
                )
    return configs


@triton.autotune(
    configs=_main_configs(),
    key=[
        "B",
        "NHEADS",
        "DIM",
        "DSTATE",
        "HAS_Z",
        "DT_SOFTPLUS",
        "BLOCK_DSTATE",
    ],
)
@triton.jit
def _ssu_kernel(
    state_ptr,  # [B, NHEADS, DIM, DSTATE]
    x_ptr,  # [B, NHEADS, DIM]
    dt_ptr,  # [B, NHEADS, DIM]
    A_ptr,  # [NHEADS, DIM, DSTATE]
    B_ptr,  # [B, NGROUPS, DSTATE]   (read directly, broadcast via h//ratio)
    C_ptr,  # [B, NGROUPS, DSTATE]   (read directly, broadcast via h//ratio)
    D_ptr,  # [NHEADS, DIM]   or dummy
    z_ptr,  # [B, NHEADS, DIM] or dummy
    dt_bias_ptr,  # [NHEADS, DIM]  or dummy
    y_ptr,  # [B, NHEADS, DIM]
    new_state_ptr,  # [B, NHEADS, DIM, DSTATE]  (state.dtype)
    state_stride_b,
    state_stride_h,
    state_stride_p,
    state_stride_n,
    x_stride_b,
    x_stride_h,
    x_stride_p,
    dt_stride_b,
    dt_stride_h,
    dt_stride_p,
    A_stride_h,
    A_stride_p,
    A_stride_n,
    B_stride_b,
    B_stride_g,
    B_stride_n,
    C_stride_b,
    C_stride_g,
    C_stride_n,
    D_stride_h,
    D_stride_p,
    z_stride_b,
    z_stride_h,
    z_stride_p,
    dt_bias_stride_h,
    dt_bias_stride_p,
    y_stride_b,
    y_stride_h,
    y_stride_p,
    new_state_stride_b,
    new_state_stride_h,
    new_state_stride_p,
    new_state_stride_n,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    B: tl.constexpr,
    NHEADS: tl.constexpr,
    DIM: tl.constexpr,
    DSTATE: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    # program -> (h, p_tile, b_tile); each program loops over its [b_start, b_end)
    # slice of the batch axis.
    h = tl.program_id(0)
    pt = tl.program_id(1)
    bt = tl.program_id(2)
    g = h // RATIO

    p_off = pt * BLOCK_P + tl.arange(0, BLOCK_P)
    p_valid = p_off < DIM

    n_off = tl.arange(0, BLOCK_DSTATE)
    n_valid = n_off < DSTATE

    b_start = bt * BLOCK_B
    b_end = b_start + BLOCK_B
    if b_end > B:
        b_end = B

    # --- head-constant tiles, loaded once and reused across this program's
    # batch slice -----------------------------------------------
    # A[h, p_tile, :] : [BLOCK_P, BLOCK_DSTATE]  (shared by every b in the slice)
    A = tl.load(
        A_ptr
        + h * A_stride_h
        + p_off[:, None] * A_stride_p
        + n_off[None, :] * A_stride_n,
        mask=p_valid[:, None] & n_valid[None, :],
        other=0.0,
    ).to(tl.float32)

    # D[h, p_tile] / dt_bias[h, p_tile] : [BLOCK_P]  (shared by every b in the slice)
    if HAS_D:
        Dv = tl.load(
            D_ptr + h * D_stride_h + p_off * D_stride_p,
            mask=p_valid,
            other=0.0,
        ).to(tl.float32)
    if HAS_DT_BIAS:
        db = tl.load(
            dt_bias_ptr + h * dt_bias_stride_h + p_off * dt_bias_stride_p,
            mask=p_valid,
            other=0.0,
        ).to(tl.float32)

    # --- per-batch work ----------------------------------------------------
    for b in range(b_start, b_end):
        # dt' : [BLOCK_P] (float32) — streamed once per step
        dt = tl.load(
            dt_ptr + b * dt_stride_b + h * dt_stride_h + p_off * dt_stride_p,
            mask=p_valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if HAS_DT_BIAS:
            dt = dt + db
        if DT_SOFTPLUS:
            # softplus(x) = log(1 + exp(x)); branch on sign for numerical stability
            # (matches F.softplus up to fp rounding).
            dt = tl.where(
                dt > 0.0,
                dt + tl.log(1.0 + tl.exp(-dt)),
                tl.log(1.0 + tl.exp(dt)),
            )

        # dA = exp(dt' * A) : [BLOCK_P, BLOCK_DSTATE]
        dA = tl.exp(dt[:, None] * A)

        # B_bcast[b, g, :] : [BLOCK_DSTATE] — one-shot stream
        Bc = tl.load(
            B_ptr + b * B_stride_b + g * B_stride_g + n_off * B_stride_n,
            mask=n_valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        dB = dt[:, None] * Bc[None, :]

        # x[b, h, p] : [BLOCK_P] — one-shot stream
        x = tl.load(
            x_ptr + b * x_stride_b + h * x_stride_h + p_off * x_stride_p,
            mask=p_valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        # state[b, h, p, :] : [BLOCK_P, BLOCK_DSTATE] — dominant traffic, one-shot
        s_offs = (
            b * state_stride_b
            + h * state_stride_h
            + p_off[:, None] * state_stride_p
            + n_off[None, :] * state_stride_n
        )
        s_mask = p_valid[:, None] & n_valid[None, :]
        state = tl.load(
            state_ptr + s_offs,
            mask=s_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        # new_state = state * dA + dB * x : [BLOCK_P, BLOCK_DSTATE]
        new_state = state * dA + dB * x[:, None]

        # store new_state directly in state.dtype (pointer dtype = state.dtype);
        # the fp32 register value is rounded on store, matching reference's
        # ``new_state.to(state.dtype)``. Avoids a fp32 scratch buffer + conversion.
        # evict_first: just-written state is not re-read in this op.
        ns_offs = (
            b * new_state_stride_b
            + h * new_state_stride_h
            + p_off[:, None] * new_state_stride_p
            + n_off[None, :] * new_state_stride_n
        )
        tl.store(
            new_state_ptr + ns_offs,
            new_state,
            mask=s_mask,
            eviction_policy="evict_first",
        )

        # y = sum_n(new_state * C_bcast) : [BLOCK_P]  (reuse register-resident
        # new_state — no extra global read of the just-stored values).
        # C_bcast is a one-shot stream.
        Cc = tl.load(
            C_ptr + b * C_stride_b + g * C_stride_g + n_off * C_stride_n,
            mask=n_valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        y = tl.sum(new_state * Cc[None, :], axis=1)

        # optional D skip connection (Dv is the reused head-constant tile)
        if HAS_D:
            y = y + Dv * x

        # optional z gate : y * silu(z) — z is a one-shot stream
        if HAS_Z:
            z = tl.load(
                z_ptr + b * z_stride_b + h * z_stride_h + p_off * z_stride_p,
                mask=p_valid,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            y = y * (z * tl.sigmoid(z))

        # store y (will be cast to x dtype by the output tensor dtype).
        y_offs = b * y_stride_b + h * y_stride_h + p_off * y_stride_p
        tl.store(
            y_ptr + y_offs, y, mask=p_valid, eviction_policy="evict_first"
        )


def _next_pow2(x):
    x = int(x)
    p = 1
    while p < x:
        p <<= 1
    return p


def _pow2_le(x):
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


# DSM budget for the fast path: ~5 live [BLOCK_M, DSTATE] fp32 tiles
# (A + state + dA + new_state + the y product); Libra hardware limit 917504 B.
_GCU_MAX_DSM_MEMORY = 917504
_KHEAD_MAX_BLOCK_M = 256
# Measured gcu400 hardware limit for grid.y is 255 (triton OutOfResources:
# "grid.y, Required: 256, Hardware limit: 255"); 65535 would let the fast
# path crash instead of doubling BLOCK_B / falling back for batch >= 2048.
_KHEAD_GRID_Y_LIMIT = 255


def _fast_path_params(batch, nheads, dim, dstate, ngroups, tensors, a_ndim):
    """Pick (K, BLOCK_B, num_warps) for ``_ssu_khead_kernel`` or None."""
    for t in tensors:
        if t is not None and not t.is_contiguous():
            return None
    if a_ndim != 3:
        return None
    if dim & (dim - 1) or dstate & (dstate - 1):
        return None
    if ngroups < 1 or nheads % ngroups != 0:
        return None
    ratio = nheads // ngroups
    max_m = _pow2_le(
        min(_KHEAD_MAX_BLOCK_M, _GCU_MAX_DSM_MEMORY // (20 * dstate))
    )
    if max_m < dim:
        return None
    k = max_m // dim
    while k > 1 and (nheads % k != 0 or ratio % k != 0):
        k //= 2
    block_b = 8
    while block_b > 1 and batch % block_b != 0:
        block_b //= 2
    while (
        batch // block_b > _KHEAD_GRID_Y_LIMIT and batch % (block_b * 2) == 0
    ):
        block_b *= 2
    if batch // block_b > _KHEAD_GRID_Y_LIMIT:
        return None
    num_warps = 4 if k * dim >= 128 else 2
    return k, block_b, num_warps


def selective_state_update(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    """Triton single-step Mamba SSM state update. See module docstring."""
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    ratio = nheads // ngroups

    # Output tensors. new_state is allocated in state.dtype and written
    # directly by the kernel (fp32 register value rounded on store) — no fp32
    # scratch, no separate dtype-conversion pass. y is in x.dtype.
    y = torch.empty((batch, nheads, dim), device=state.device, dtype=x.dtype)
    new_state = torch.empty(
        (batch, nheads, dim, dstate), device=state.device, dtype=state.dtype
    )

    HAS_D = D is not None
    HAS_Z = z is not None
    HAS_DT_BIAS = dt_bias is not None

    fast = _fast_path_params(
        batch,
        nheads,
        dim,
        dstate,
        ngroups,
        (state, x, dt, A, B, C, D, z, dt_bias),
        A.ndim,
    )
    if fast is not None:
        k, block_b, num_warps = fast
        dummy = x  # never dereferenced when the constexpr flag is False
        _ssu_khead_kernel[(nheads // k, batch // block_b)](
            state,
            x,
            dt,
            A,
            B,
            C,
            D if HAS_D else dummy,
            z if HAS_Z else dummy,
            dt_bias if HAS_DT_BIAS else dummy,
            y,
            new_state,
            NHEADS=nheads,
            DIM=dim,
            DSTATE=dstate,
            NGROUPS=ngroups,
            RATIO=ratio,
            HAS_D=HAS_D,
            HAS_Z=HAS_Z,
            HAS_DT_BIAS=HAS_DT_BIAS,
            DT_SOFTPLUS=bool(dt_softplus),
            K=k,
            BLOCK_B=block_b,
            num_warps=num_warps,
            num_stages=2,
        )
        return y, new_state

    BLOCK_DSTATE = _next_pow2(dstate)

    # Strides
    ss = state.stride()
    xs = x.stride()
    dts = dt.stride()
    As = A.stride()
    Bs = B.stride()
    Cs = C.stride()
    ys = y.stride()
    nss = new_state.stride()

    # Grid: one program per (head, dim-tile, batch-tile). The batch axis is
    # split into ``cdiv(B, BLOCK_B)`` tiles so the program count can saturate
    # the device's parallelism (the v3 layout folded the whole batch into one
    # program per (h, p-tile), which under-utilised the batch256 case). Each
    # program still loops over its own contiguous batch slice, so the
    # head-constant A/D/dt_bias are loaded once per program and reused across
    # that slice.
    grid = lambda meta: (
        nheads,
        triton.cdiv(dim, meta["BLOCK_P"]),
        triton.cdiv(batch, meta["BLOCK_B"]),
    )

    # Dummy pointers for absent tensors (Triton requires a tensor arg; we pass
    # a valid tensor and gate use with the constexpr flag so it's never read).
    dummy = (
        x  # any tensor on the right device; never loaded when flag is False
    )

    _ssu_kernel[grid](
        state,
        x,
        dt,
        A,
        B,
        C,
        D if HAS_D else dummy,
        z if HAS_Z else dummy,
        dt_bias if HAS_DT_BIAS else dummy,
        y,
        new_state,
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        xs[0],
        xs[1],
        xs[2],
        dts[0],
        dts[1],
        dts[2],
        As[0],
        As[1],
        As[2],
        Bs[0],
        Bs[1],
        Bs[2],
        Cs[0],
        Cs[1],
        Cs[2],
        D.stride(0) if HAS_D else 0,
        D.stride(1) if HAS_D else 1,
        z.stride(0) if HAS_Z else 0,
        z.stride(1) if HAS_Z else 0,
        z.stride(2) if HAS_Z else 1,
        dt_bias.stride(0) if HAS_DT_BIAS else 0,
        dt_bias.stride(1) if HAS_DT_BIAS else 1,
        ys[0],
        ys[1],
        ys[2],
        nss[0],
        nss[1],
        nss[2],
        nss[3],
        HAS_D=HAS_D,
        HAS_Z=HAS_Z,
        HAS_DT_BIAS=HAS_DT_BIAS,
        DT_SOFTPLUS=dt_softplus,
        B=batch,
        NHEADS=nheads,
        DIM=dim,
        DSTATE=dstate,
        RATIO=ratio,
        BLOCK_DSTATE=BLOCK_DSTATE,
    )

    return y, new_state


__all__ = ["selective_state_update"]
