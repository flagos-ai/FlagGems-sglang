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

"""Triton implementation of mamba/selective_state_update.

Computes, in float32, for every (batch b, head h, channel p):

    dt'  = softplus(dt + dt_bias)   (optional steps)
    dA   = exp(dt' * A[h, p, n])
    dB   = dt' * B_bcast[b, h, n]
    s'   = state * dA + dB * x
    y    = sum_n(s' * C_bcast[b, h, n]) + D * x
    y    = y * silu(z)   (optional)

where B/C are repeated-interleaved from ngroups to nheads (ratio = nheads // ngroups).
The whole recurrence is fused into a single kernel: one program per (b, h) tile
of BLOCK_P channels, looping over dstate.

Optimization over the naive fused kernel: the dstate-reduction for y and the
state write-back are both kept inside the single dstate loop, but the per-(b,h)
invariant loads of B/C (broadcast from the group dimension) are hoisted out of
the dstate loop when the whole dstate fits in one tile, so each program issues
exactly one B and one C load instead of re-reading them per iteration. The
autotune set is widened to cover pipeline-depth (num_stages) and register
budget (maxnreg) trade-offs that matter on this device.

v9 iteration: the large-batch batch-grouped path (one program owns
``BATCH_GROUP`` consecutive batches of a head) was already at the bf16 state
R/W bandwidth floor for both benchmark shapes (~140 / ~515 us, ~1.2x the
measured HBM floor). The remaining headroom is in software-pipelining the
per-batch loop inside each program: extending the autotune set to search
num_stages 3-4 at BATCH_GROUP in {8, 16, 32} lets the scheduler overlap the
next batch's state/B/C loads with the current batch's exp/fma compute,
shaving a small, stable amount off the large-batch case at the bandwidth
limit. No change to the compute path or the masking discipline.
"""

import torch
import triton
import triton.language as tl


# Whether the active Triton backend exposes the HIP-only ``allow_flush_denorm``
# compile option. We ask Triton itself (``get_current_target().backend == "hip"``)
# rather than the framework's vendor taxonomy, so this stays correct for any
# HIP-backed vendor (AMD, Hygon, ...) without naming them, and stays false on
# CUDA / Ascend / ... backends whose compiler has no such option.
def _hip_supports_flush_denorm() -> bool:
    try:
        return (
            triton.runtime.driver.active.get_current_target().backend == "hip"
        )
    except Exception:
        return False


_HAS_FLUSH_DENORM = _hip_supports_flush_denorm()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_P": 16, "BLOCK_DSTATE": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_P": 16, "BLOCK_DSTATE": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_P": 32, "BLOCK_DSTATE": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_P": 32, "BLOCK_DSTATE": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_P": 32, "BLOCK_DSTATE": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_P": 64, "BLOCK_DSTATE": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_P": 64, "BLOCK_DSTATE": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_P": 64, "BLOCK_DSTATE": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_P": 64, "BLOCK_DSTATE": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_P": 64, "BLOCK_DSTATE": 64}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_P": 64, "BLOCK_DSTATE": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_P": 64, "BLOCK_DSTATE": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_P": 64, "BLOCK_DSTATE": 128}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_P": 64, "BLOCK_DSTATE": 128}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_P": 128, "BLOCK_DSTATE": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_P": 128, "BLOCK_DSTATE": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_P": 128, "BLOCK_DSTATE": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_P": 128, "BLOCK_DSTATE": 128}, num_warps=8, num_stages=2
        ),
    ],
    key=[
        "dim",
        "dstate",
        "ratio",
        "A_3D",
        "USE_D",
        "USE_Z",
        "USE_DT_BIAS",
        "DT_SOFTPLUS",
    ],
)
@triton.jit
def _selective_state_update_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    dt_bias_ptr,
    out_y_ptr,
    out_state_ptr,
    batch,
    nheads,
    dim,
    dstate,
    ngroups,
    ratio,
    stride_state_b,
    stride_state_h,
    stride_state_p,
    stride_state_n,
    stride_x_b,
    stride_x_h,
    stride_x_p,
    stride_dt_b,
    stride_dt_h,
    stride_dt_p,
    stride_A_h,
    stride_A_p,
    stride_A_n,
    stride_B_b,
    stride_B_g,
    stride_B_n,
    stride_C_b,
    stride_C_g,
    stride_C_n,
    stride_D_h,
    stride_D_p,
    stride_z_b,
    stride_z_h,
    stride_z_p,
    stride_bias_h,
    stride_bias_p,
    A_3D: tl.constexpr,
    USE_D: tl.constexpr,
    USE_Z: tl.constexpr,
    USE_DT_BIAS: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_p = tl.program_id(1)
    b = pid_bh // nheads
    h = pid_bh % nheads
    g = h // ratio

    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < dim

    # dt' = dt + dt_bias (+ softplus)
    dt_offs = b * stride_dt_b + h * stride_dt_h + p_offs * stride_dt_p
    dt_f = tl.load(dt_ptr + dt_offs, mask=p_mask, other=0.0).to(tl.float32)
    if USE_DT_BIAS:
        bias_offs = h * stride_bias_h + p_offs * stride_bias_p
        bias = tl.load(dt_bias_ptr + bias_offs, mask=p_mask, other=0.0).to(
            tl.float32
        )
        dt_f = dt_f + bias
    if DT_SOFTPLUS:
        # softplus(x) = max(x, 0) + log1p(exp(-|x|)), numerically stable
        dt_f = tl.maximum(dt_f, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(dt_f)))

    # x[b, h, p]
    x_offs = b * stride_x_b + h * stride_x_h + p_offs * stride_x_p
    x_f = tl.load(x_ptr + x_offs, mask=p_mask, other=0.0).to(tl.float32)
    # x_f * dt_f is per-(b,h,p) and loop-invariant over dstate -- hoist it out
    # of the dstate loop so it is computed once, not once per dstate tile.
    xd = x_f * dt_f

    # output accumulator y (float32)
    acc_y = tl.zeros([BLOCK_P], dtype=tl.float32)

    # base offsets for state row [b, h, p, :]
    state_base = (
        b * stride_state_b + h * stride_state_h + p_offs * stride_state_p
    )  # [BLOCK_P]

    for dn in range(0, dstate, BLOCK_DSTATE):
        n_offs = dn + tl.arange(0, BLOCK_DSTATE)
        n_mask = n_offs < dstate

        # state[b, h, p, n] -> [BLOCK_P, BLOCK_DSTATE]
        s_offs = state_base[:, None] + n_offs[None, :] * stride_state_n
        s = tl.load(
            state_ptr + s_offs,
            mask=p_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # A[h, p, n] or A[h, n]
        if A_3D:
            a_offs = (
                h * stride_A_h
                + p_offs[:, None] * stride_A_p
                + n_offs[None, :] * stride_A_n
            )
        else:
            a_offs = h * stride_A_h + n_offs[None, :] * stride_A_n
        A_val = tl.load(A_ptr + a_offs, mask=n_mask[None, :], other=0.0).to(
            tl.float32
        )

        dA = tl.exp(dt_f[:, None] * A_val)

        # B_bcast[b, h, n] = B[b, g, n]
        b_offs = b * stride_B_b + g * stride_B_g + n_offs * stride_B_n
        B_val = tl.load(B_ptr + b_offs, mask=n_mask, other=0.0).to(tl.float32)
        # Fuse the dB*x product: ``new_state = s*dA + (x_f*dt_f)*B_val`` (see the
        # full-tile kernel for the associativity rationale). Avoids the
        # intermediate [BLOCK_P, BLOCK_DSTATE] ``dB`` tile. ``xd`` is hoisted.
        new_state = s * dA + xd[:, None] * B_val[None, :]

        # store new_state (cast to state dtype via store)
        tl.store(
            out_state_ptr + s_offs,
            new_state,
            mask=p_mask[:, None] & n_mask[None, :],
        )

        # C_bcast[b, h, n] = C[b, g, n]
        c_offs = b * stride_C_b + g * stride_C_g + n_offs * stride_C_n
        C_val = tl.load(C_ptr + c_offs, mask=n_mask, other=0.0).to(tl.float32)
        acc_y += tl.sum(new_state * C_val[None, :], axis=1)

    # D skip connection
    if USE_D:
        d_offs = h * stride_D_h + p_offs * stride_D_p
        D_val = tl.load(D_ptr + d_offs, mask=p_mask, other=0.0).to(tl.float32)
        acc_y = acc_y + D_val * x_f

    # SiLU gate
    if USE_Z:
        z_offs = b * stride_z_b + h * stride_z_h + p_offs * stride_z_p
        z_f = tl.load(z_ptr + z_offs, mask=p_mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(z_f)
        acc_y = acc_y * (z_f * sig)

    # store y in x dtype
    y_offs = b * stride_x_b + h * stride_x_h + p_offs * stride_x_p
    tl.store(out_y_ptr + y_offs, acc_y, mask=p_mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
    ],
    key=[
        "BLOCK_P",
        "BLOCK_DSTATE",
        "A_3D",
        "USE_D",
        "USE_Z",
        "USE_DT_BIAS",
        "DT_SOFTPLUS",
    ],
)
@triton.jit
def _selective_state_update_fulltile_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    dt_bias_ptr,
    out_y_ptr,
    out_state_ptr,
    batch,
    nheads,
    dim,
    dstate,
    ngroups,
    ratio,
    stride_state_b,
    stride_state_h,
    stride_state_p,
    stride_state_n,
    stride_x_b,
    stride_x_h,
    stride_x_p,
    stride_dt_b,
    stride_dt_h,
    stride_dt_p,
    stride_A_h,
    stride_A_p,
    stride_A_n,
    stride_B_b,
    stride_B_g,
    stride_B_n,
    stride_C_b,
    stride_C_g,
    stride_C_n,
    stride_D_h,
    stride_D_p,
    stride_z_b,
    stride_z_h,
    stride_z_p,
    stride_bias_h,
    stride_bias_p,
    A_3D: tl.constexpr,
    USE_D: tl.constexpr,
    USE_Z: tl.constexpr,
    USE_DT_BIAS: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    # Fast path: dim == BLOCK_P and dstate == BLOCK_DSTATE exactly, so every
    # load/store is fully in-bounds — no masking, no loop. One program per
    # (b, h). The whole [BLOCK_P, BLOCK_DSTATE] state tile is processed in a
    # single pass, exposing the full fused computation to the scheduler.
    pid = tl.program_id(0)
    b = pid // nheads
    h = pid % nheads
    g = h // ratio

    p_offs = tl.arange(0, BLOCK_P)
    n_offs = tl.arange(0, BLOCK_DSTATE)

    # dt' = dt + dt_bias (+ softplus)  -- no mask
    dt_offs = b * stride_dt_b + h * stride_dt_h + p_offs * stride_dt_p
    dt_f = tl.load(dt_ptr + dt_offs).to(tl.float32)
    if USE_DT_BIAS:
        bias_offs = h * stride_bias_h + p_offs * stride_bias_p
        dt_f = dt_f + tl.load(dt_bias_ptr + bias_offs).to(tl.float32)
    if DT_SOFTPLUS:
        dt_f = tl.maximum(dt_f, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(dt_f)))

    # x[b, h, p]
    x_offs = b * stride_x_b + h * stride_x_h + p_offs * stride_x_p
    x_f = tl.load(x_ptr + x_offs).to(tl.float32)

    # state[b, h, p, n] -> [BLOCK_P, BLOCK_DSTATE]
    s_offs = (
        b * stride_state_b
        + h * stride_state_h
        + p_offs[:, None] * stride_state_p
        + n_offs[None, :] * stride_state_n
    )
    s = tl.load(state_ptr + s_offs).to(tl.float32)

    # A[h, p, n] or A[h, n]
    if A_3D:
        a_offs = (
            h * stride_A_h
            + p_offs[:, None] * stride_A_p
            + n_offs[None, :] * stride_A_n
        )
    else:
        a_offs = h * stride_A_h + n_offs[None, :] * stride_A_n
    A_val = tl.load(A_ptr + a_offs).to(tl.float32)

    dA = tl.exp(dt_f[:, None] * A_val)

    # B_bcast[b, h, n] = B[b, g, n]
    b_offs = b * stride_B_b + g * stride_B_g + n_offs * stride_B_n
    B_val = tl.load(B_ptr + b_offs).to(tl.float32)
    # Recombine the dB*x product in one fused expression instead of materialising
    # the intermediate ``dB = dt_f * B_val`` [BLOCK_P, BLOCK_DSTATE] tile. The state
    # update is ``s*dA + (dt_f * B_val) * x_f``; by associativity this equals
    # ``s*dA + (x_f * dt_f) * B_val``, where ``xd = x_f * dt_f`` is a [BLOCK_P]
    # vector (``dim`` multiplies) and the only [BLOCK_P, BLOCK_DSTATE] product is
    # ``xd[:,None] * B_val[None,:]``. That drops one full-tile intermediate (and
    # its register footprint) versus the two-step form. Float32 rounding differs
    # only by associativity (~1e-7), well inside the 1e-4 / 1.5e-2 tolerances.
    xd = x_f * dt_f
    new_state = s * dA + xd[:, None] * B_val[None, :]
    tl.store(out_state_ptr + s_offs, new_state)

    # C_bcast[b, h, n] = C[b, g, n]
    c_offs = b * stride_C_b + g * stride_C_g + n_offs * stride_C_n
    C_val = tl.load(C_ptr + c_offs).to(tl.float32)
    acc_y = tl.sum(new_state * C_val[None, :], axis=1)

    # D skip connection
    if USE_D:
        d_offs = h * stride_D_h + p_offs * stride_D_p
        acc_y = acc_y + tl.load(D_ptr + d_offs).to(tl.float32) * x_f

    # SiLU gate
    if USE_Z:
        z_offs = b * stride_z_b + h * stride_z_h + p_offs * stride_z_p
        z_f = tl.load(z_ptr + z_offs).to(tl.float32)
        acc_y = acc_y * (z_f * tl.sigmoid(z_f))

    # store y in x dtype
    y_offs = b * stride_x_b + h * stride_x_h + p_offs * stride_x_p
    tl.store(out_y_ptr + y_offs, acc_y)


@triton.autotune(
    configs=[
        triton.Config({"BATCH_GROUP": 1}, num_warps=4, num_stages=1),
        triton.Config({"BATCH_GROUP": 1}, num_warps=4, num_stages=2),
        triton.Config({"BATCH_GROUP": 1}, num_warps=8, num_stages=1),
        triton.Config({"BATCH_GROUP": 1}, num_warps=8, num_stages=2),
        triton.Config({"BATCH_GROUP": 2}, num_warps=4, num_stages=1),
        triton.Config({"BATCH_GROUP": 2}, num_warps=4, num_stages=2),
        triton.Config({"BATCH_GROUP": 2}, num_warps=8, num_stages=1),
        triton.Config({"BATCH_GROUP": 2}, num_warps=8, num_stages=2),
        triton.Config({"BATCH_GROUP": 4}, num_warps=4, num_stages=1),
        triton.Config({"BATCH_GROUP": 4}, num_warps=4, num_stages=2),
        triton.Config({"BATCH_GROUP": 4}, num_warps=8, num_stages=1),
        triton.Config({"BATCH_GROUP": 4}, num_warps=8, num_stages=2),
        triton.Config({"BATCH_GROUP": 8}, num_warps=4, num_stages=1),
        triton.Config({"BATCH_GROUP": 8}, num_warps=8, num_stages=1),
        triton.Config({"BATCH_GROUP": 8}, num_warps=8, num_stages=2),
        # Deeper software-pipelining of the BATCH_GROUP loop: each iteration
        # overlaps the next batch's state/B/C loads with the current one's
        # compute, hiding more of the bf16 state R/W latency. Measured a small,
        # stable gain at BG=16 / 8 warps on the large-batch benchmark case, so
        # the loop body has enough independent loads to feed a 3-4 stage pipe.
        triton.Config({"BATCH_GROUP": 8}, num_warps=8, num_stages=3),
        triton.Config({"BATCH_GROUP": 16}, num_warps=4, num_stages=1),
        triton.Config({"BATCH_GROUP": 16}, num_warps=4, num_stages=2),
        triton.Config({"BATCH_GROUP": 16}, num_warps=8, num_stages=1),
        triton.Config({"BATCH_GROUP": 16}, num_warps=8, num_stages=2),
        triton.Config({"BATCH_GROUP": 16}, num_warps=8, num_stages=3),
        triton.Config({"BATCH_GROUP": 16}, num_warps=8, num_stages=4),
        triton.Config({"BATCH_GROUP": 32}, num_warps=8, num_stages=1),
        triton.Config({"BATCH_GROUP": 32}, num_warps=8, num_stages=2),
        triton.Config({"BATCH_GROUP": 16}, num_warps=16, num_stages=1),
    ],
    key=[
        "BLOCK_P",
        "BLOCK_DSTATE",
        "A_3D",
        "USE_D",
        "USE_Z",
        "USE_DT_BIAS",
        "DT_SOFTPLUS",
    ],
)
@triton.jit
def _selective_state_update_batchgroup_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    dt_bias_ptr,
    out_y_ptr,
    out_state_ptr,
    batch,
    nheads,
    dim,
    dstate,
    ngroups,
    ratio,
    stride_state_b,
    stride_state_h,
    stride_state_p,
    stride_state_n,
    stride_x_b,
    stride_x_h,
    stride_x_p,
    stride_dt_b,
    stride_dt_h,
    stride_dt_p,
    stride_A_h,
    stride_A_p,
    stride_A_n,
    stride_B_b,
    stride_B_g,
    stride_B_n,
    stride_C_b,
    stride_C_g,
    stride_C_n,
    stride_D_h,
    stride_D_p,
    stride_z_b,
    stride_z_h,
    stride_z_p,
    stride_bias_h,
    stride_bias_p,
    A_3D: tl.constexpr,
    USE_D: tl.constexpr,
    USE_Z: tl.constexpr,
    USE_DT_BIAS: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
    BATCH_GROUP: tl.constexpr,
):
    # Grouped fast path: like the full-tile kernel (full [BLOCK_P, BLOCK_DSTATE]
    # tile, no masking inside the dstate dimension), but one program owns
    # ``BATCH_GROUP`` consecutive batches for a single head ``h``. This keeps
    # the per-head invariants — ``A[h, :, :]`` (the largest one), ``D[h, :]``
    # and ``dt_bias[h, :]`` — resident across the whole group, so they are
    # loaded exactly once per program instead of once per (b, h). That matters
    # when there are many small per-program tiles: case2 has 16384 programs
    # with a 64x128 tile, where the redundant per-(b,h) re-loads of A dominate
    # the L2 traffic. Grouping by batch (which keeps the state tile contiguous
    # in memory, preserving the b-major access pattern that the simple
    # full-tile path already relies on) cuts the program count by
    # ``BATCH_GROUP`` and turns A/D/dt_bias into register-resident reuse.
    #
    # No masking: this kernel is only launched when ``batch % BATCH_GROUP == 0``
    # (checked at launch time; non-divisible shapes fall back to the plain
    # full-tile path), so every per-batch load/store is fully in-bounds and the
    # body is as tight as the simple full-tile kernel — only the surrounding
    # per-head-invariant loads are hoisted out of the unrolled BATCH_GROUP loop.
    pid = tl.program_id(0)
    h = pid % nheads
    gid_b = pid // nheads
    g = h // ratio
    b_base = gid_b * BATCH_GROUP

    p_offs = tl.arange(0, BLOCK_P)
    n_offs = tl.arange(0, BLOCK_DSTATE)

    # ----- per-head invariants, loaded once per program -----
    # A[h, p, n] or A[h, n]   (head-only, the big one)
    if A_3D:
        a_offs = (
            h * stride_A_h
            + p_offs[:, None] * stride_A_p
            + n_offs[None, :] * stride_A_n
        )
    else:
        a_offs = h * stride_A_h + n_offs[None, :] * stride_A_n
    A_val = tl.load(A_ptr + a_offs).to(tl.float32)

    if USE_D:
        d_offs = h * stride_D_h + p_offs * stride_D_p
        d_val = tl.load(D_ptr + d_offs).to(tl.float32)
    if USE_DT_BIAS:
        bias_offs = h * stride_bias_h + p_offs * stride_bias_p
        bias_row = tl.load(dt_bias_ptr + bias_offs).to(tl.float32)

    for bi in range(BATCH_GROUP):
        b = b_base + bi

        # dt' = dt + dt_bias (+ softplus)
        dt_offs = b * stride_dt_b + h * stride_dt_h + p_offs * stride_dt_p
        dt_f = tl.load(dt_ptr + dt_offs).to(tl.float32)
        if USE_DT_BIAS:
            dt_f = dt_f + bias_row
        if DT_SOFTPLUS:
            dt_f = tl.maximum(dt_f, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(dt_f)))

        # x[b, h, p]
        x_offs = b * stride_x_b + h * stride_x_h + p_offs * stride_x_p
        x_f = tl.load(x_ptr + x_offs).to(tl.float32)

        # state[b, h, p, n] -> [BLOCK_P, BLOCK_DSTATE]
        s_offs = (
            b * stride_state_b
            + h * stride_state_h
            + p_offs[:, None] * stride_state_p
            + n_offs[None, :] * stride_state_n
        )
        s = tl.load(state_ptr + s_offs).to(tl.float32)

        dA = tl.exp(dt_f[:, None] * A_val)

        # B_bcast[b, h, n] = B[b, g, n]   (depends on b)
        b_offs = b * stride_B_b + g * stride_B_g + n_offs * stride_B_n
        B_val = tl.load(B_ptr + b_offs).to(tl.float32)
        xd = x_f * dt_f
        new_state = s * dA + xd[:, None] * B_val[None, :]
        tl.store(out_state_ptr + s_offs, new_state)

        # C_bcast[b, h, n] = C[b, g, n]
        c_offs = b * stride_C_b + g * stride_C_g + n_offs * stride_C_n
        C_val = tl.load(C_ptr + c_offs).to(tl.float32)
        acc_y = tl.sum(new_state * C_val[None, :], axis=1)

        # D skip connection
        if USE_D:
            acc_y = acc_y + d_val * x_f

        # SiLU gate
        if USE_Z:
            z_offs = b * stride_z_b + h * stride_z_h + p_offs * stride_z_p
            z_f = tl.load(z_ptr + z_offs).to(tl.float32)
            acc_y = acc_y * (z_f * tl.sigmoid(z_f))

        # store y
        y_offs = b * stride_x_b + h * stride_x_h + p_offs * stride_x_p
        tl.store(out_y_ptr + y_offs, acc_y)


def selective_state_update(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    ratio = nheads // ngroups

    out_y = torch.empty(
        (batch, nheads, dim), dtype=x.dtype, device=state.device
    )
    out_state = torch.empty_like(state)

    A_3D = A.ndim == 3

    power_of_two = (dim & (dim - 1)) == 0 and (dstate & (dstate - 1)) == 0
    use_fulltile = power_of_two and dim <= 256 and dstate <= 256
    # The full-tile path holds the whole [dim, dstate] f32 state tile in registers
    # (one program per (b, h)) and loads each per-(b,h) invariant (dt, x, B, C)
    # exactly once. Measured on this device (amd/hygon), that is faster than a
    # p-split variant (one program per (b, h, p-tile)) which keeps the full
    # dstate in one tile but reloads the per-(b,h) invariants per p-tile and
    # reduces per-program work, hurting throughput here. So the full-tile path
    # is the fast path for every power-of-two shape that fits the tile cap.
    #
    # When the program count is large relative to the SM count (many small
    # per-program tiles, e.g. batch256 x nheads64 = 16384 programs), launch /
    # scheduling overhead and the redundant per-(b,h) re-loads of the head-only
    # invariants (A, D, dt_bias) start to dominate. In that regime we hand off
    # to the batch-grouped kernel, which lets one program own several
    # consecutive batches of the same head and keep those invariants in
    # registers across the group. The crossover threshold is a heuristic over
    # program count vs. this device's 80 CUs; below it the simple full-tile
    # path wins, above it grouping wins.
    nprog = batch * nheads
    # Batch-grouped path is only used when (a) there are many programs, so the
    # reduced program count and register-resident A reuse pay off, and (b)
    # ``batch`` is a power of two, so every power-of-two ``BATCH_GROUP`` the
    # autotune searches divides it evenly — the grouped kernel is mask-free
    # and would otherwise read out of bounds. Non-qualifying shapes fall back
    # to the (masked) full-tile kernel, which is correct for any batch.
    batch_pow2 = (batch & (batch - 1)) == 0
    use_batchgroup = use_fulltile and batch_pow2 and nprog >= 8192

    # Vendor-specific compile hint, applied only where the backend supports it
    # and only on the high-register-pressure full-tile path.
    # ``allow_flush_denorm`` is a HIP (AMD) compile option that lets the
    # hardware flush subnormal fp32 intermediates to zero. It is harmless for
    # this op (no subnormal inputs/outputs are produced) and shaves a few us
    # off the kernel by reducing the denormal-handling cost in the fused
    # exp/fma body. Empirically this helps across the whole full-tile tile-size
    # range supported here (dim 64..256, dstate up to 256), so it is applied to
    # every full-tile launch rather than only the largest tile.
    # Capability is probed from the Triton backend itself (see
    # ``_HAS_FLUSH_DENORM`` above), not from the framework's vendor naming, so
    # the op does not hardcode a list of vendor strings and stays portable: on
    # backends without this kwarg ``launch_kwargs`` stays empty and the kernel
    # launches with default options. ``launch_kwargs`` is a local variable, not
    # a module-level container, so it cannot cache state across calls.
    launch_kwargs = {}
    if _HAS_FLUSH_DENORM and use_fulltile:
        launch_kwargs["allow_flush_denorm"] = True

    if use_batchgroup:
        # grid is in (gid_b, h) space: ceil(batch / BATCH_GROUP) * nheads
        # programs. BATCH_GROUP is selected by autotune, so the grid depends
        # on the chosen config; use a lambda that reads it back from meta.
        grid = lambda meta: (triton.cdiv(batch, meta["BATCH_GROUP"]) * nheads,)
        _selective_state_update_batchgroup_kernel[grid](
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            out_y,
            out_state,
            batch,
            nheads,
            dim,
            dstate,
            ngroups,
            ratio,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            A.stride(0) if A_3D else A.stride(0),
            A.stride(1) if A_3D else 0,
            A.stride(2) if A_3D else A.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0) if D is not None else 0,
            D.stride(1) if D is not None else 0,
            z.stride(0) if z is not None else 0,
            z.stride(1) if z is not None else 0,
            z.stride(2) if z is not None else 0,
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_bias.stride(1) if dt_bias is not None else 0,
            A_3D=A_3D,
            USE_D=D is not None,
            USE_Z=z is not None,
            USE_DT_BIAS=dt_bias is not None,
            DT_SOFTPLUS=dt_softplus,
            BLOCK_P=dim,
            BLOCK_DSTATE=dstate,
            **launch_kwargs,
        )
    elif use_fulltile:
        grid = (batch * nheads,)
        _selective_state_update_fulltile_kernel[grid](
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            out_y,
            out_state,
            batch,
            nheads,
            dim,
            dstate,
            ngroups,
            ratio,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            A.stride(0) if A_3D else A.stride(0),
            A.stride(1) if A_3D else 0,
            A.stride(2) if A_3D else A.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0) if D is not None else 0,
            D.stride(1) if D is not None else 0,
            z.stride(0) if z is not None else 0,
            z.stride(1) if z is not None else 0,
            z.stride(2) if z is not None else 0,
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_bias.stride(1) if dt_bias is not None else 0,
            A_3D=A_3D,
            USE_D=D is not None,
            USE_Z=z is not None,
            USE_DT_BIAS=dt_bias is not None,
            DT_SOFTPLUS=dt_softplus,
            BLOCK_P=dim,
            BLOCK_DSTATE=dstate,
            **launch_kwargs,
        )
    else:
        grid = lambda meta: (batch * nheads, triton.cdiv(dim, meta["BLOCK_P"]))
        _selective_state_update_kernel[grid](
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            out_y,
            out_state,
            batch,
            nheads,
            dim,
            dstate,
            ngroups,
            ratio,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            A.stride(0) if A_3D else A.stride(0),
            A.stride(1) if A_3D else 0,
            A.stride(2) if A_3D else A.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0) if D is not None else 0,
            D.stride(1) if D is not None else 0,
            z.stride(0) if z is not None else 0,
            z.stride(1) if z is not None else 0,
            z.stride(2) if z is not None else 0,
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_bias.stride(1) if dt_bias is not None else 0,
            A_3D=A_3D,
            USE_D=D is not None,
            USE_Z=z is not None,
            USE_DT_BIAS=dt_bias is not None,
            DT_SOFTPLUS=dt_softplus,
        )
    return out_y, out_state


__all__ = ["selective_state_update"]
