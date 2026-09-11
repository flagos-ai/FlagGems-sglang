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

"""Operator: mamba/state_passing

Triton implementation of the Mamba2 SSD inter-chunk state-passing scan.

For each (batch, head) the recurrence is sequential over chunks:

    cur_{c+1} = cur_c * exp(dA_last[b, c, h]) + states[b, c, h]
    out[b, c, h] = cur_c            (pre-update state, cast to states.dtype)
    final_states[b, h] = cur_{nchunks}

where dA_last[b, c, h] = dA_cumsum[b, h, c, -1].

The recurrence couples the chunks but the ``dim`` axis is fully independent:
every element of the length-``dim`` state vector is multiplied by the same
per-(b,h,c) scalar decay and incremented by the corresponding element of the
state increment.  We therefore tile the ``dim`` axis across multiple programs
(grid = (B*nheads, n_dim_tiles)) so the whole grid fits the GPU's parallelism
even for small (B*nheads) cases, while keeping the sequential chunk scan inside
each program in registers.

Pure portable Triton; no vendor-private ops, no module-level mutable state.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # The op is bandwidth-bound: essential traffic = read states + write out
        # + write final_states (each element touched once).  Both production
        # shapes run at 84-90 % of peak HBM bandwidth, so the only remaining
        # lever is keeping enough programs in flight to saturate the memory
        # subsystem while the sequential chunk recurrence (FMA chain) hides
        # behind the streaming vector loads/stores.
        #
        # Empirical sweep on the production shapes shows BLOCK_DIM=2048 is the
        # sweet spot for *both* cases: with dim=8192 it yields 4 dim tiles per
        # (batch,head) -> 4*B*nheads programs (1024 for case1, 8192 for case2),
        # which fills the 78 SMs of the H20 with enough in-flight programs to
        # overlap the per-chunk load/store chain.  The per-case best num_warps
        # differs (fewer warps for the low-parallelism case1, more warps for the
        # high-parallelism case2) and is selected by the autotune key=nchunks.
        # The config set is kept small on purpose: with >25 configs the
        # autotune's own do_bench noise picks sub-optimal winners, and a small
        # set centred on the measured sweet spot is both faster to compile and
        # more reliably optimal.
        triton.Config({"BLOCK_DIM": 2048}, num_warps=1, num_stages=3),
        triton.Config({"BLOCK_DIM": 2048}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_DIM": 2048}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_DIM": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_DIM": 4096}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_DIM": 4096}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_DIM": 8192}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_DIM": 8192}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_DIM": 8192}, num_warps=8, num_stages=2),
    ],
    key=["nchunks", "dim_full"],
)
@triton.jit
def _state_passing_kernel(
    states_ptr,
    out_ptr,
    final_ptr,
    dA_ptr,  # dA_cumsum[b, h, c, L-1] read directly with strides
    init_ptr,  # initial_states[b, h, :] float32, or dummy (has_init=False)
    # states [B, nchunks, nheads, dim] strides:
    s_b_stride,
    s_c_stride,
    s_h_stride,
    # out   [B, nchunks, nheads, dim] strides:
    o_b_stride,
    o_c_stride,
    o_h_stride,
    # dA_cumsum [B, nheads, nchunks, L] strides; we read [b, h, c, L-1]:
    dA_b_stride,
    dA_h_stride,
    dA_c_stride,
    L_LAST,  # index of last element within L (= L - 1)
    # initial_states [B, nheads, dim] strides:
    init_b_stride,
    init_h_stride,
    # final_states [B, nheads, dim] strides:
    f_b_stride,
    f_h_stride,
    nheads,
    has_init: tl.constexpr,
    nchunks: tl.constexpr,
    dim_full: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    # Compile-time flag: every dim tile is full when dim divides evenly into
    # BLOCK_DIM.  Letting the compiler evaluate this constexpr drops *all*
    # per-element boundary checks for the even-dim shapes (both production
    # shapes have dim=8192 and every autotune BLOCK_DIM divides it).
    EVEN_DIM: tl.constexpr = (dim_full % BLOCK_DIM) == 0

    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    b = pid_bh // nheads
    h = pid_bh % nheads

    # Offset into the dim tile this program owns.
    dim_off = pid_d * BLOCK_DIM
    offs = dim_off + tl.arange(0, BLOCK_DIM)

    # Base pointers for this (b, h).
    states_bh = states_ptr + b * s_b_stride + h * s_h_stride
    out_bh = out_ptr + b * o_b_stride + h * o_h_stride
    dA_bh = dA_ptr + b * dA_b_stride + h * dA_h_stride + L_LAST

    # Hint the dim axis is contiguous & aligned so 128-bit vector loads/stores
    # are emitted (portable Triton, no vendor-private ops).
    s_ptr_c = states_bh + offs
    o_ptr_c = out_bh + offs
    tl.multiple_of(s_ptr_c, 16)
    tl.multiple_of(o_ptr_c, 16)

    out_ty = out_ptr.dtype.element_ty
    final_bh = final_ptr + b * f_b_stride + h * f_h_stride

    if EVEN_DIM:
        # Initialize running state cur (float32).
        if has_init:
            init_bh = init_ptr + b * init_b_stride + h * init_h_stride
            cur = tl.load(init_bh + offs).to(tl.float32)
        else:
            cur = tl.zeros([BLOCK_DIM], dtype=tl.float32)

        for c in tl.static_range(0, nchunks):
            # Record current state (pre-update) as the output for this chunk.
            # out is written once per chunk and never re-read -> .cg (streaming)
            # so it passes through L2 without evicting the hot running state.
            tl.store(
                o_ptr_c + c * o_c_stride, cur.to(out_ty), cache_modifier=".cg"
            )

            # Per-(b,h,c) decay factor: exp(dA_cumsum[b, h, c, -1]).  Scalar;
            # .ca keeps the nchunks scalar loads on-chip across the dim tile.
            dA_last = tl.load(
                dA_bh + c * dA_c_stride, cache_modifier=".ca"
            ).to(tl.float32)
            decay = tl.exp(dA_last)

            # state increment for this chunk (float32 for accuracy).  Read once
            # per chunk, no reuse across the recurrence -> .cg streaming load.
            s_c = tl.load(s_ptr_c + c * s_c_stride, cache_modifier=".cg").to(
                tl.float32
            )

            cur = cur * decay + s_c

        # Final state (float32). final_states is [B, nheads, dim].
        tl.store(final_bh + offs, cur)
    else:
        dim_mask = offs < dim_full
        if has_init:
            init_bh = init_ptr + b * init_b_stride + h * init_h_stride
            cur = tl.load(init_bh + offs, mask=dim_mask, other=0.0).to(
                tl.float32
            )
        else:
            cur = tl.zeros([BLOCK_DIM], dtype=tl.float32)

        for c in tl.static_range(0, nchunks):
            tl.store(
                o_ptr_c + c * o_c_stride,
                cur.to(out_ty),
                mask=dim_mask,
                cache_modifier=".cg",
            )
            dA_last = tl.load(
                dA_bh + c * dA_c_stride, cache_modifier=".ca"
            ).to(tl.float32)
            decay = tl.exp(dA_last)
            s_c = tl.load(
                s_ptr_c + c * s_c_stride,
                mask=dim_mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            cur = cur * decay + s_c

        tl.store(final_bh + offs, cur, mask=dim_mask)


def state_passing(states, dA_cumsum, initial_states=None):
    batch, nchunks, nheads, dim = states.shape

    out = torch.empty(
        (batch, nchunks, nheads, dim), device=states.device, dtype=states.dtype
    )
    final_states = torch.empty(
        (batch, nheads, dim), device=states.device, dtype=torch.float32
    )

    # dA_cumsum: [B, nheads, nchunks, L]. We read the last element along L directly
    # via stride indexing, so no copy/contiguous() is needed.
    L = dA_cumsum.shape[-1]
    L_LAST = L - 1

    dA_b_stride, dA_h_stride, dA_c_stride, _ = dA_cumsum.stride()

    s_b_stride, s_c_stride, s_h_stride, _ = states.stride()
    o_b_stride, o_c_stride, o_h_stride, _ = out.stride()
    # final_states is contiguous [B, nheads, dim] (just allocated); use its strides
    # so the store is correct regardless of how the caller laid it out internally.
    f_b_stride, f_h_stride, _ = final_states.stride()

    has_init = initial_states is not None
    if has_init:
        init_ptr = initial_states
        init_b_stride, init_h_stride, _ = initial_states.stride()
        # reference: cur = initial_states.float().clone()  -> we load & promote in kernel
    else:
        init_ptr = states  # dummy valid pointer; has_init=False guards loads
        init_b_stride = 0
        init_h_stride = 0

    grid = lambda meta: (batch * nheads, triton.cdiv(dim, meta["BLOCK_DIM"]))
    _state_passing_kernel[grid](
        states,
        out,
        final_states,
        dA_cumsum,
        init_ptr,
        s_b_stride,
        s_c_stride,
        s_h_stride,
        o_b_stride,
        o_c_stride,
        o_h_stride,
        dA_b_stride,
        dA_h_stride,
        dA_c_stride,
        L_LAST,
        init_b_stride,
        init_h_stride,
        f_b_stride,
        f_h_stride,
        nheads,
        has_init,
        nchunks,
        dim,
    )
    return out, final_states


__all__ = ["state_passing"]
