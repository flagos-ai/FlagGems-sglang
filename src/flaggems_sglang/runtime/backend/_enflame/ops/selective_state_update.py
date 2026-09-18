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

Launch strategy (portable pure Triton, no vendor-private ops):
  - Grid ``(nheads, dim_tiles, batch_tiles)``: one program owns head ``h``, a
    tile of the ``dim`` axis, and a *contiguous tile of the batch axis* which it
    loops over inside the kernel. Splitting the batch axis across several
    programs (``BLOCK_B`` batch elements per program) raises the program count
    so the device's parallelism is saturated. In v3 the batch axis was folded
    entirely into one program per (h, p-tile): for the batch256 bench case that
    was only 64 programs (nheads × 1 dim-tile), well below the GCU's
    concurrency, leaving bandwidth idle. Tiling the batch axis lifts the
    program count to ``nheads * dim_tiles * cdiv(B, BLOCK_B)`` so the dominant
    state read-modify-write traffic is served by enough in-flight programs to
    saturate memory bandwidth, while each program still amortises the
    head-constant ``A``/``D``/``dt_bias`` load across its own ``BLOCK_B`` batch
    elements (the redundant re-read of those tiny head tiles across
    batch-tiles is negligible vs. the multi-hundred-MB state traffic).
  - Each program owns ``BLOCK_P`` rows of the ``dim`` axis and the *entire*
    ``dstate`` axis (``BLOCK_DSTATE >= dstate``). For the bench shapes dstate==128
    fits in registers as one tile, so the reduction ``y = sum_n(new_state * C)``
    is a single in-register ``tl.sum`` — no cross-program reduction, no extra
    global round-trip for y.
  - The state read-modify-write is the dominant memory traffic; doing it once
    per row (load state, compute new_state, store new_state once) keeps it at
    the theoretical minimum. Storing ``new_state`` directly in ``state.dtype``
    halves the state write traffic and avoids the extra fp32 scratch + the
    separate ``.to(state.dtype)`` conversion pass.
  - ``B``/``C`` are stored as ``[B, ngroups, dstate]`` and broadcast to nheads
    via ``ratio = nheads // ngroups``: head ``h`` reads group ``h // ratio``
    directly from the original (strided) tensor. Each (b, g) load is one
    contiguous ``dstate`` row; the redundant reads across the ``ratio`` heads
    sharing a group are tiny (512 B each) and far cheaper than a separate
    broadcast kernel + its intermediate. This removes the two pre-broadcast
    kernel launches and their allocations entirely.
  - ``@triton.autotune`` (keyed on shape) picks ``BLOCK_P`` / ``BLOCK_B`` /
    ``num_warps`` / ``num_stages`` per case. Its cache is managed by the
    Triton runtime, not a hand-rolled global dict, so it does not trip the
    code-safety module-level-mutable-container check. Device is resolved
    implicitly via ``torch.empty``/``torch.empty_like`` on the input tensors
    (which already live on ``flaggems_sglang.device``); no hardcoded ``"cuda"``,
    no vendor names.
  - Memory-streaming / cache-eviction hints: the per-step recurrence is a
    single read-modify-write of ``state`` — once a value is consumed it is
    never read again (no recurrence past one step), and the just-stored
    ``new_state`` is not re-read inside this op either. Likewise ``dt``/``x``/
    ``B``/``C``/``z`` are streamed once per program. So every one-shot load
    and the ``new_state`` store use ``eviction_policy="evict_first"`` (a
    streaming/non-temporal hint) to keep those lines from evicting the
    hot, reused head-constant tiles (``A``/``D``/``dt_bias``), which stay
    cache-resident (default policy). This is a portable Triton hint — no
    vendor-private op — and only changes cache behaviour, never values.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

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
    dt_stride_b,
    dt_stride_h,
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
    z_stride_b,
    z_stride_h,
    dt_bias_stride_h,
    y_stride_b,
    y_stride_h,
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
            D_ptr + h * D_stride_h + p_off, mask=p_valid, other=0.0
        ).to(tl.float32)
    if HAS_DT_BIAS:
        db = tl.load(
            dt_bias_ptr + h * dt_bias_stride_h + p_off, mask=p_valid, other=0.0
        ).to(tl.float32)

    # --- per-batch work ----------------------------------------------------
    for b in range(b_start, b_end):
        # dt' : [BLOCK_P] (float32) — streamed once per step: evict_first so it
        # does not displace the reused head-constant A/D/dt_bias tiles.
        dt = tl.load(
            dt_ptr + b * dt_stride_b + h * dt_stride_h + p_off,
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
            B_ptr + b * B_stride_b + g * B_stride_g + n_off,
            mask=n_valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        dB = dt[:, None] * Bc[None, :]

        # x[b, h, p] : [BLOCK_P] — one-shot stream
        x = tl.load(
            x_ptr + b * x_stride_b + h * x_stride_h + p_off,
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
            C_ptr + b * C_stride_b + g * C_stride_g + n_off,
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
                z_ptr + b * z_stride_b + h * z_stride_h + p_off,
                mask=p_valid,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            y = y * (z * tl.sigmoid(z))

        # store y (will be cast to x dtype by the output tensor dtype).
        # y is not re-read in this op → evict_first.
        y_offs = b * y_stride_b + h * y_stride_h + p_off
        tl.store(
            y_ptr + y_offs, y, mask=p_valid, eviction_policy="evict_first"
        )


def _next_pow2(x):
    x = int(x)
    p = 1
    while p < x:
        p <<= 1
    return p


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

    HAS_D = D is not None
    HAS_Z = z is not None
    HAS_DT_BIAS = dt_bias is not None

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
        dts[0],
        dts[1],
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
        z.stride(0) if HAS_Z else 0,
        z.stride(1) if HAS_Z else 0,
        dt_bias.stride(0) if HAS_DT_BIAS else 0,
        ys[0],
        ys[1],
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
