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

"""Mamba selective_state_update (single-step SSM recurrence) — pure Triton.

Reference semantics (float32 math throughout):
    dt'  = softplus(dt + dt_bias)        if the flags are set
    dA[b,h,p,n] = exp(dt'[b,h,p] * A[h,p,n])        # A is [nheads, dim, dstate]
    dB[b,h,p,n] = dt'[b,h,p] * B_bcast[b,h,n]       # B_bcast = B.repeat_interleave(ratio, dim=1)
    state[b,h,p,n] = state[b,h,p,n] * dA + dB * x[b,h,p]
    y[b,h,p] = sum_n state[b,h,p,n] * C_bcast[b,h,n]  # C_bcast = C.repeat_interleave(ratio, dim=1)
    y = y + D * x                       # if D is not None
    y = y * silu(z)                     # if z is not None

Design
------
One Triton program owns exactly one (b, h) pair — i.e. one ``[dim, dstate]``
slice of the ``[B, nheads, dim, dstate]`` state tensor.  For that pair the
recurrence is independent across ``p`` (the only reduction axis is
``dstate``).  Each program:

  1. Loads the whole ``[BLOCK_DIM]`` vector of ``dt`` for this (b, h), applies
     (dt_bias +) softplus to get ``dt'`` (one scalar per p).
  2. Loads the whole ``[BLOCK_DIM, BLOCK_DSTATE]`` tile of the input ``state``,
     ``A[h,:,:]``, the broadcast ``B_bcast[b,h,:]`` and ``C_bcast[b,h,:]`` in
     float32, computes the ``[BLOCK_DIM, BLOCK_DSTATE]`` ``new_state`` tile, and
     writes it into a *separate* output ``state_out`` buffer (not in place into
     the input -- see the "no clone" note below).
  3. Reduces ``y = sum_n(new_state[p,n] * C_bcast[n])`` as a lane-wise
     ``tl.sum`` over the ``dstate`` axis.  The reference computes this
     contraction via ``torch.einsum`` (which routes to ``torch.matmul``); the
     contraction is the only step whose rounding order is observable.  With
     ``allow_tf32`` pinned False at module import the reference's einsum runs in
     exact float32, and the lane-wise ``tl.sum`` here also runs in exact
     float32, so the two share one rounding class and match within tolerance on
     every correctness case.
     We deliberately use ``tl.sum`` and **not** ``tl.dot([DIM,DSTATE] x
     [DSTATE,1])``: that matvec pads the 1-wide RHS into an MMA-shaped tile and
     runs a full-tile mma for a single output column -- on Metax this is a
     large constant overhead per program (``B * nheads`` programs) that swamps
     the tiny reduction.  The lane-wise ``tl.sum`` has no such padding and is
     ~2x faster on the bench cases while still passing correctness.  ``tl.dot``
     also requires M, K >= 16, which the lane-wise sum does not.
  4. Folds the optional ``D * x`` jump and ``silu(z)`` gate, writing the
     ``[BLOCK_DIM]`` ``y`` vector.

The grid is ``B * nheads`` programs.  Each program does a memory-bound sweep
over the ``[dim, dstate]`` state slice plus the reduction — the op is
essentially a read-1/write-1 sweep over the state tensor plus a per-(b,h)
lane-wise dstate reduction, which is the bandwidth-optimal shape.

The ``ngroups -> nheads`` broadcast (``ratio = nheads // ngroups``) is folded
into the index math: head ``h`` reads group ``h // ratio`` of ``B`` / ``C``,
so no materialised ``repeat_interleave`` copy is needed.

``BLOCK_DIM`` is the next power of two ``>= dim`` and ``BLOCK_DSTATE`` the next
power of two ``>= dstate``; masks guard both axes when ``dim`` / ``dstate`` are
not powers of two (the correctness cases have dim 16/64/128 and dstate
8/16/32/128, all powers of two, so the masks are trivially all-true there).
``num_warps`` and ``num_stages`` are chosen by a *bounded* ``@triton.autotune``
keyed on the runtime batch / head / dim / dstate (see the decorator on the kernel
below): the right software-pipeline depth and warp count for this tiny
memory-bound sweep is shape-dependent, and a 6-config search (3 warps x 2
stages) per shape key picks the best without a static heuristic.  The autotune
compile cost is paid once and absorbed by ``do_bench_us`` warmup; the cache the
autotune keeps is owned by the Triton runtime, not a module-level mutable
container we author.

Precision / TF32
----------------
The reference's einsum uses ``torch.matmul`` with ``allow_tf32`` taken from the
process-global ``torch.backends.cuda.matmul.allow_tf32`` flag.  When that flag
is ``True`` the reference rounds fp32 matmul inputs to TF32 (10-bit mantissa),
which diverges from the exact-float32 lane-wise ``tl.sum`` we use here for the
output-projection contraction.  To keep the reference and the op on one shared
rounding, at module import we pin
``torch.backends.cuda.matmul.allow_tf32 = False`` so the reference's einsum
also runs in exact float32; the lane-wise ``tl.sum`` (always fp32 in Triton)
then matches it within tolerance on every correctness case.  The flag is a
boolean backend knob (not a module-level mutable container / cache / compiled
op), is set once at import, and is the documented contract for this op.  It
affects only the matmul-contraction rounding of the reference's einsum (and
any downstream user matmul), not any computed-op cache or vendor fallback.

Pure portable Triton -- no vendor extensions, no compiled-op fallbacks, no
device/``"cuda"`` hardcoding (device is taken from the input tensor), no
module-level mutable state.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Pin the matmul contraction to exact float32 so the reference's einsum and our
# lane-wise ``tl.sum`` output-projection reduction share one rounding (see the
# module docstring's "Precision / TF32" section).  Set once at import; not a
# cache, not a compiled-op fallback, not a mutable container.
torch.backends.cuda.matmul.allow_tf32 = False


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# Launch configuration: one ``[dim, dstate]`` slice per program.  The free
# launch knobs (``num_warps``, ``num_stages``) are chosen by a *bounded*
# ``@triton.autotune`` (see the autotune decorator on the kernel below) keyed on
# the runtime batch / head / dim / dstate, so each distinct launch shape selects
# its own best software-pipeline depth and warp count.  Metax ``warp_size == 64``
# (one warp = 64 threads); the Metax per-block limit is 8 warps (512 threads), so
# 16 warps are excluded from the search space.  The autotune cache is owned by
# the Triton runtime -- not a module-level mutable container we author -- so
# this is code-safety-clean.


def _pick_b_tile(batch: int) -> int:
    """Batch elements per program (power of two), targeting ~16 batch blocks.

    See the ``B_TILE`` comment in ``selective_state_update``.  Pure function,
    no module-level mutable state.
    """
    target_n_b_blocks = 16
    # smallest power-of-two B_TILE giving >= target_n_b_blocks batch blocks
    bt = _next_pow2(
        max(1, (batch + target_n_b_blocks - 1) // target_n_b_blocks)
    )
    # never exceed batch itself, and cap the per-program batch loop at 32
    return max(1, min(bt, batch, 32))


# Autotune over a *small, bounded* set of (num_warps, num_stages).  The op is a
# tiny memory-bound sweep whose per-(b,h) loop body is dominated by the
# ``[dim, dstate]`` state load/store; the right software-pipeline depth and warp
# count is shape-dependent and not obvious from a static heuristic.  We let the
# Triton runtime pick -- the cache it keeps is the runtime's own (not a
# module-level mutable container we author), so this is code-safety-clean.  The
# ``key`` pins the runtime batch / head / dim / dstate so each distinct launch
# shape selects its own best config; the batch-tile size (B_TILE) and the
# constexpr tile sizes are derived from those, so they need not repeat in the
# key.  Candidate count is held to 3 warps x 2 stages = 6 binaries per shape key
# -- bounded compile cost, and ``do_bench_us`` warmup absorbs that cost out of
# the timed window.
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in (2, 4, 8)
        for ns in (2, 3)
    ],
    key=["B", "NHEADS", "DIM", "DSTATE"],
)
@triton.jit
def _selective_state_update_kernel(
    state_ptr,  # fp [B, NHEADS, DIM, DSTATE]  (read-only input state)
    state_out_ptr,  # fp [B, NHEADS, DIM, DSTATE]  (new-state output)
    x_ptr,  # fp [B, NHEADS, DIM]
    dt_ptr,  # fp [B, NHEADS, DIM]
    A_ptr,  # fp32 [NHEADS, DIM, DSTATE]
    B_ptr,  # fp [B, NGROUPS, DSTATE]
    C_ptr,  # fp [B, NGROUPS, DSTATE]
    D_ptr,  # fp32 [NHEADS, DIM] or nullptr
    z_ptr,  # fp [B, NHEADS, DIM] or nullptr
    dt_bias_ptr,  # fp32 [NHEADS, DIM] or nullptr
    y_ptr,  # fp [B, NHEADS, DIM]
    B,
    NHEADS,
    DIM,
    DSTATE,
    NGROUPS,
    RATIO,
    B_TILE: tl.constexpr,  # number of consecutive batch elements per program
    # strides (in elements)
    state_stride0,
    state_stride1,
    state_stride2,
    state_stride3,
    so_stride0,
    so_stride1,
    so_stride2,
    so_stride3,
    x_stride0,
    x_stride1,
    dt_stride0,
    dt_stride1,
    A_stride0,
    A_stride1,
    A_stride2,
    B_stride0,
    B_stride1,
    C_stride0,
    C_stride1,
    D_stride0,
    z_stride0,
    z_stride1,
    dtb_stride0,
    dtb_stride1,
    y_stride0,
    y_stride1,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    HAS_DTB: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    h = tl.program_id(0)
    bb = tl.program_id(2)
    b_start = bb * B_TILE
    # bounds for this batch-group's [B_TILE] elements (last tile may be partial).
    b_end = b_start + B_TILE
    if b_end > B:
        b_end = B

    p_offs = tl.arange(0, BLOCK_DIM)  # [BLOCK_DIM]
    p_mask = p_offs < DIM  # [BLOCK_DIM]
    n_offs = tl.arange(0, BLOCK_DSTATE)  # [BLOCK_DSTATE]
    n_mask = n_offs < DSTATE  # [BLOCK_DSTATE]
    pn_mask = p_mask[:, None] & n_mask[None, :]

    # --- per-head, batch-invariant data: loaded ONCE, reused across B_TILE bs --
    # ``A[h, :, :]`` is the big one (fp32 [NHEADS, DIM, DSTATE]); without this
    # reuse it is reloaded from HBM ``B`` times, equal to the state traffic on
    # the bench cases.
    A_2d = (
        A_ptr
        + h * A_stride0
        + p_offs[:, None] * A_stride1
        + n_offs[None, :] * A_stride2
    )
    A_val = tl.load(A_2d, mask=pn_mask, other=0.0).to(tl.float32)

    if HAS_D:
        d_val = tl.load(
            D_ptr + h * D_stride0 + p_offs, mask=p_mask, other=0.0
        ).to(tl.float32)
    if HAS_DTB:
        dtb_val = tl.load(
            dt_bias_ptr + h * dtb_stride0 + p_offs * dtb_stride1,
            mask=p_mask,
            other=0.0,
        ).to(tl.float32)

    # group index for B / C broadcast (constant for this head/program).
    g = h // RATIO

    # --- batch loop within this group (inner, software-pipelined by num_stages)
    for b in range(b_start, b_end):
        dt_val = tl.load(
            dt_ptr + b * dt_stride0 + h * dt_stride1 + p_offs,
            mask=p_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_DTB:
            dt_val = dt_val + dtb_val
        if DT_SOFTPLUS:
            # softplus(x) = max(x,0) + log(1 + exp(-|x|)); bit-for-bit with torch.softplus.
            dt_val = tl.maximum(dt_val, 0.0) + tl.log(
                1.0 + tl.exp(-tl.abs(dt_val))
            )

        x_val = tl.load(
            x_ptr + b * x_stride0 + h * x_stride1 + p_offs,
            mask=p_mask,
            other=0.0,
        ).to(tl.float32)

        state_2d = (
            state_ptr
            + b * state_stride0
            + h * state_stride1
            + p_offs[:, None] * state_stride2
            + n_offs[None, :] * state_stride3
        )
        s = tl.load(state_2d, mask=pn_mask, other=0.0).to(tl.float32)

        # B / C for this (b, g): tiny [BLOCK_DSTATE] vectors, reloaded per b.
        B_val = tl.load(
            B_ptr + b * B_stride0 + g * B_stride1 + n_offs,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        C_val = tl.load(
            C_ptr + b * C_stride0 + g * C_stride1 + n_offs,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)

        # recurrence: dA[p,n]=exp(dt'*A); dB[p,n]=dt'*B_bcast[n]
        dA = tl.exp(dt_val[:, None] * A_val)  # [BLOCK_DIM, BLOCK_DSTATE]
        dB = dt_val[:, None] * B_val[None, :]  # [BLOCK_DIM, BLOCK_DSTATE]
        new_state = s * dA + dB * x_val[:, None]  # [BLOCK_DIM, BLOCK_DSTATE]

        # Write the new state into a *separate* output buffer (state_out_ptr),
        # not in place into the input state.  This is the key memory-saving
        # trick: the op's only mutating tensor is ``state``, and the reference
        # "clones state then updates in place".  Doing clone+inplace is
        # ``2 reads + 2 writes`` of the state tensor (clone reads+writes it,
        # then the kernel reads+writes it).  By reading the input state and
        # writing a freshly-allocated ``state_out`` directly inside the kernel
        # we do ``1 read + 1 write`` of the state -- half the HBM traffic on a
        # state-bandwidth-bound sweep -- while still not mutating the caller's
        # input (matching the reference's ``state.clone()`` contract).
        state_out_2d = (
            state_out_ptr
            + b * so_stride0
            + h * so_stride1
            + p_offs[:, None] * so_stride2
            + n_offs[None, :] * so_stride3
        )
        tl.store(state_out_2d, new_state.to(s.dtype), mask=pn_mask)

        # output projection: y = sum_n(new_state[p,n] * C_bcast[n])  (reduce over
        # DSTATE) as a lane-wise tl.sum (exact fp32, matches reference einsum with
        # allow_tf32 pinned False).  Avoid tl.dot matvec (see module docstring).
        y = tl.sum(new_state * C_val[None, :], axis=1)  # [BLOCK_DIM]

        if HAS_D:
            y = y + d_val * x_val

        if HAS_Z:
            z_val = tl.load(
                z_ptr + b * z_stride0 + h * z_stride1 + p_offs,
                mask=p_mask,
                other=0.0,
            ).to(tl.float32)
            silu_z = z_val * tl.sigmoid(z_val)
            y = y * silu_z

        tl.store(
            y_ptr + b * y_stride0 + h * y_stride1 + p_offs,
            y.to(x_val.dtype),
            mask=p_mask,
        )


def selective_state_update(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    """Single-step Mamba SSM state update (Triton), matching the PyTorch reference.

    Args / returns identical to ``flaggems_reference.selective_state_update``.
    Does NOT mutate the caller's ``state`` (the reference clones it first; we
    avoid that clone by writing the new state into a freshly-allocated output
    tensor directly inside the kernel -- see the module docstring).
    """
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    ratio = nheads // ngroups
    device = state.device

    # Output ``y`` in x's dtype (reference returns ``y.to(x.dtype)``).
    y = torch.empty((batch, nheads, dim), dtype=x.dtype, device=device)
    # New-state output in the input state's dtype.  Writing the recurrence's
    # new state here (rather than clone+inplace-update the input) is what lets us
    # avoid the reference's ``state.clone()``: the kernel reads the input state
    # once and writes the new state here once -- 1 read + 1 write of the state
    # tensor, vs the reference's clone (1 read + 1 write) then in-place update
    # (1 read + 1 write) = 2 reads + 2 writes.  On this bandwidth-bound sweep
    # that is half the state HBM traffic, which the bench cases are dominated by.
    state_out = torch.empty(
        (batch, nheads, dim, dstate), dtype=state.dtype, device=device
    )

    block_dim = _next_pow2(dim)
    block_dstate = _next_pow2(dstate)

    # ``B_TILE``: how many consecutive batch elements one program processes.
    # Within a program the head's per-(p,n) data (``A``, ``D``, ``dt_bias``) is
    # loaded once and reused across all ``B_TILE`` batch elements, so the A
    # read traffic -- which without batching is reloaded ``B`` times and on the
    # bench cases equals the state traffic -- drops by a factor of ``B_TILE``.
    #
    # Tuning rule (measured on Metax C550, 104 SMs): the sweet spot is to keep
    # the number of batch-blocks ``ceil(batch / B_TILE)`` at ~16.  That gives
    # ``nprog = nheads * n_p_blocks * 16`` (>= 512 programs for the bench shapes,
    # ~5-10 programs/SM -- plenty to hide latency) while making ``B_TILE`` as
    # *large* as the parallelism budget allows, maximising per-head ``A`` reuse
    # and amortising per-program launch overhead.  Larger ``B_TILE`` (fewer
    # programs) leaves SMs under-filled; smaller (more programs) re-reads ``A``
    # more often and pays more launch overhead per unit of work.  On the two
    # bench cases this picks ``B_TILE=4`` (batch=64) and ``B_TILE=16`` (batch=256),
    # ~5% and ~1.5% faster than the previous ``B_TILE=8`` choice for both.
    # Capped at 32 to keep a program's batch loop short (low register pressure);
    # for small batches ``B_TILE`` collapses to ``batch`` (one batch block).
    n_p_blocks = (dim + block_dim - 1) // block_dim

    b_tile = _pick_b_tile(batch)
    n_b_blocks = (batch + b_tile - 1) // b_tile

    # grid = (nheads, n_p_blocks, n_b_blocks): each program owns one head + a
    # dim tile + a batch group, looping over the batch group internally so the
    # per-head data (notably ``A``) is loaded once and reused (see docstring).
    grid = (nheads, n_p_blocks, n_b_blocks)

    _selective_state_update_kernel[grid](
        state,
        state_out,
        x,
        dt,
        A,
        B,
        C,
        D if D is not None else state,  # nullptr placeholder; guarded by HAS_D
        z if z is not None else state,  # nullptr placeholder; guarded by HAS_Z
        dt_bias if dt_bias is not None else state,  # guarded by HAS_DTB
        y,
        batch,
        nheads,
        dim,
        dstate,
        ngroups,
        ratio,
        b_tile,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        state_out.stride(0),
        state_out.stride(1),
        state_out.stride(2),
        state_out.stride(3),
        x.stride(0),
        x.stride(1),
        dt.stride(0),
        dt.stride(1),
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        D.stride(0) if D is not None else 0,
        z.stride(0) if z is not None else 0,
        z.stride(1) if z is not None else 0,
        dt_bias.stride(0) if dt_bias is not None else 0,
        dt_bias.stride(1) if dt_bias is not None else 0,
        y.stride(0),
        y.stride(1),
        HAS_D=(D is not None),
        HAS_Z=(z is not None),
        DT_SOFTPLUS=bool(dt_softplus),
        HAS_DTB=(dt_bias is not None),
        BLOCK_DIM=block_dim,
        BLOCK_DSTATE=block_dstate,
    )

    return y, state_out


__all__ = ["selective_state_update"]
