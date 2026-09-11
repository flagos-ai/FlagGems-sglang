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

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _select_config(dim: int):

    block_d = triton.next_power_of_2(max(dim, 16))

    if block_d > 8192:
        block_d = 8192
    if block_d >= 4096:
        num_warps = 8
    elif block_d >= 1024:
        num_warps = 8
    elif block_d >= 256:
        num_warps = 4
    else:
        num_warps = 2
    num_stages = 2
    return block_d, num_warps, num_stages


@triton.jit
def _state_passing_kernel(
    states_ptr,
    out_ptr,
    init_ptr,
    dA_cumsum_ptr,
    final_ptr,
    s_str_b,
    s_str_c,
    s_str_h,
    o_str_b,
    o_str_c,
    o_str_h,
    init_str_b,
    init_str_h,
    da_str_b,
    da_str_h,
    da_str_c,
    da_str_l,
    fin_str_b,
    fin_str_h,
    nheads,
    nchunks,
    dim,
    L_LAST,  # physical index of the last time-step within a chunk (L-1)
    BLOCK_D: tl.constexpr,
    HAS_INIT: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_bh // nheads
    h = pid_bh % nheads

    d_off = pid_d * BLOCK_D
    d_idx = d_off + tl.arange(0, BLOCK_D)
    d_mask = d_idx < dim

    # cur = initial_states[b, h, :] (float32), or zeros if None.
    if HAS_INIT:
        cur = tl.load(
            init_ptr + b * init_str_b + h * init_str_h + d_idx,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        cur = tl.zeros((BLOCK_D,), dtype=tl.float32)

    da_bh_base = b * da_str_b + h * da_str_h + L_LAST * da_str_l

    # Sequential cross-chunk recurrence (per-lane serial order == reference).
    for c in range(nchunks):
        # Snapshot the *input* state of this chunk: out[b, c, h, :] = cur.
        tl.store(
            out_ptr + b * o_str_b + c * o_str_c + h * o_str_h + d_idx,
            cur.to(OUT_DTYPE),
            mask=d_mask,
        )

        # decay = exp(dA_cumsum[b, h, c, L-1])  (scalar, fused extraction)
        da = tl.load(dA_cumsum_ptr + da_bh_base + c * da_str_c)
        decay = tl.exp(da.to(tl.float32))

        # cur = cur * decay + states[b, c, h, :].float()
        s = tl.load(
            states_ptr + b * s_str_b + c * s_str_c + h * s_str_h + d_idx,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        cur = cur * decay + s

    # final_states[b, h, :] = cur  (float32)
    tl.store(
        final_ptr + b * fin_str_b + h * fin_str_h + d_idx,
        cur,
        mask=d_mask,
    )


def state_passing(states, dA_cumsum, initial_states=None):
    batch, nchunks, nheads, dim = states.shape

    out = torch.empty(
        (batch, nchunks, nheads, dim),
        device=states.device,
        dtype=states.dtype,
    )

    has_init = initial_states is not None
    if has_init:
        init = initial_states.to(torch.float32)
    else:

        init = states

    final = torch.empty(
        (batch, nheads, dim),
        device=states.device,
        dtype=torch.float32,
    )

    block_d, num_warps, num_stages = _select_config(dim)
    num_d_blocks = triton.cdiv(dim, block_d)
    grid = (batch * nheads, num_d_blocks)

    # Output dtype as a Triton dtype constexpr.
    out_triton_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[states.dtype]

    da_str_b, da_str_h, da_str_c, da_str_l = dA_cumsum.stride()
    L_last = dA_cumsum.shape[-1] - 1

    _state_passing_kernel[grid](
        states,
        out,
        init,
        dA_cumsum,
        final,
        states.stride(0),
        states.stride(1),
        states.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        init.stride(0) if has_init else 0,
        init.stride(1) if has_init else 0,
        da_str_b,
        da_str_h,
        da_str_c,
        da_str_l,
        final.stride(0),
        final.stride(1),
        nheads,
        nchunks,
        dim,
        L_last,
        BLOCK_D=block_d,
        HAS_INIT=has_init,
        OUT_DTYPE=out_triton_dtype,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out, final


__all__ = ["state_passing"]
