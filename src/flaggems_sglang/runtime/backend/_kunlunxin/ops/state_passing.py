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
def _state_passing_4t_kernel(
    out_ptr,  # [B, nchunks, nheads, dim]   (states.dtype)
    final_ptr,  # [B, nheads, dim]            (float32)
    states_ptr,  # [B, nchunks, nheads, dim]   (states.dtype)
    dA_cumsum_ptr,  # [B, nheads, nchunks, L]     (float32)
    init_ptr,  # [B, nheads, dim] float32
    has_init,  # tl.int1: 1 if initial_states given else 0
    nheads,
    dim,
    L,
    stride_ob,
    stride_oc,
    stride_oh,
    stride_od,  # out
    stride_fb,
    stride_fh,
    stride_fd,  # final
    stride_sb,
    stride_sc,
    stride_sh,
    stride_sd,  # states
    stride_db,
    stride_dh,
    stride_dc,
    stride_dl,  # dA
    stride_ib,
    stride_ih,
    stride_id,  # init
    BLOCK_D: tl.constexpr,
    NCHUNKS: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # lane = batch * nheads + head
    pid_dg = tl.program_id(1)  # tile-group index (each group = 4 dim-tiles)

    b = pid_bh // nheads
    h = pid_bh % nheads

    base_d = pid_dg * (BLOCK_D * 4)
    offs = base_d + tl.arange(0, BLOCK_D)
    d0 = offs
    d1 = offs + BLOCK_D
    d2 = offs + 2 * BLOCK_D
    d3 = offs + 3 * BLOCK_D
    m0 = d0 < dim
    m1 = d1 < dim
    m2 = d2 < dim
    m3 = d3 < dim

    # Four independent fp32 cur tiles, kept in registers across the loop.
    if has_init:
        cur0 = tl.load(
            init_ptr + b * stride_ib + h * stride_ih + d0 * stride_id,
            mask=m0,
            other=0.0,
        ).to(tl.float32)
        cur1 = tl.load(
            init_ptr + b * stride_ib + h * stride_ih + d1 * stride_id,
            mask=m1,
            other=0.0,
        ).to(tl.float32)
        cur2 = tl.load(
            init_ptr + b * stride_ib + h * stride_ih + d2 * stride_id,
            mask=m2,
            other=0.0,
        ).to(tl.float32)
        cur3 = tl.load(
            init_ptr + b * stride_ib + h * stride_ih + d3 * stride_id,
            mask=m3,
            other=0.0,
        ).to(tl.float32)
    else:
        cur0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        cur1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        cur2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        cur3 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    last_l = L - 1

    for c in tl.static_range(0, NCHUNKS):
        # decay = exp(dA_cumsum[b, h, c, L-1])  (scalar, broadcast over dim)
        decay = tl.exp(
            tl.load(
                dA_cumsum_ptr
                + b * stride_db
                + h * stride_dh
                + c * stride_dc
                + last_l * stride_dl
            )
        )

        tl.store(
            out_ptr
            + b * stride_ob
            + c * stride_oc
            + h * stride_oh
            + d0 * stride_od,
            cur0.to(states_ptr.dtype.element_ty),
            mask=m0,
        )
        tl.store(
            out_ptr
            + b * stride_ob
            + c * stride_oc
            + h * stride_oh
            + d1 * stride_od,
            cur1.to(states_ptr.dtype.element_ty),
            mask=m1,
        )
        tl.store(
            out_ptr
            + b * stride_ob
            + c * stride_oc
            + h * stride_oh
            + d2 * stride_od,
            cur2.to(states_ptr.dtype.element_ty),
            mask=m2,
        )
        tl.store(
            out_ptr
            + b * stride_ob
            + c * stride_oc
            + h * stride_oh
            + d3 * stride_od,
            cur3.to(states_ptr.dtype.element_ty),
            mask=m3,
        )

        # Then load the four states[:, c] tiles as a single load group.
        st0 = tl.load(
            states_ptr
            + b * stride_sb
            + c * stride_sc
            + h * stride_sh
            + d0 * stride_sd,
            mask=m0,
            other=0.0,
        ).to(tl.float32)
        st1 = tl.load(
            states_ptr
            + b * stride_sb
            + c * stride_sc
            + h * stride_sh
            + d1 * stride_sd,
            mask=m1,
            other=0.0,
        ).to(tl.float32)
        st2 = tl.load(
            states_ptr
            + b * stride_sb
            + c * stride_sc
            + h * stride_sh
            + d2 * stride_sd,
            mask=m2,
            other=0.0,
        ).to(tl.float32)
        st3 = tl.load(
            states_ptr
            + b * stride_sb
            + c * stride_sc
            + h * stride_sh
            + d3 * stride_sd,
            mask=m3,
            other=0.0,
        ).to(tl.float32)

        # Advance each of the four independent cur chains.
        cur0 = cur0 * decay + st0
        cur1 = cur1 * decay + st1
        cur2 = cur2 * decay + st2
        cur3 = cur3 * decay + st3

    tl.store(
        final_ptr + b * stride_fb + h * stride_fh + d0 * stride_fd,
        cur0,
        mask=m0,
    )
    tl.store(
        final_ptr + b * stride_fb + h * stride_fh + d1 * stride_fd,
        cur1,
        mask=m1,
    )
    tl.store(
        final_ptr + b * stride_fb + h * stride_fh + d2 * stride_fd,
        cur2,
        mask=m2,
    )
    tl.store(
        final_ptr + b * stride_fb + h * stride_fh + d3 * stride_fd,
        cur3,
        mask=m3,
    )


@triton.jit
def _state_passing_1t_kernel(
    out_ptr,  # [B, nchunks, nheads, dim]   (states.dtype)
    final_ptr,  # [B, nheads, dim]            (float32)
    states_ptr,  # [B, nchunks, nheads, dim]   (states.dtype)
    dA_cumsum_ptr,  # [B, nheads, nchunks, L]     (float32)
    init_ptr,  # [B, nheads, dim] float32
    has_init,  # tl.int1
    nheads,
    dim,
    L,
    stride_ob,
    stride_oc,
    stride_oh,
    stride_od,
    stride_fb,
    stride_fh,
    stride_fd,
    stride_sb,
    stride_sc,
    stride_sh,
    stride_sd,
    stride_db,
    stride_dh,
    stride_dc,
    stride_dl,
    stride_ib,
    stride_ih,
    stride_id,
    BLOCK_D: tl.constexpr,
    NCHUNKS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_bh // nheads
    h = pid_bh % nheads

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < dim

    if has_init:
        cur = tl.load(
            init_ptr + b * stride_ib + h * stride_ih + d_off * stride_id,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        cur = tl.zeros((BLOCK_D,), dtype=tl.float32)

    last_l = L - 1
    for c in tl.static_range(0, NCHUNKS):
        tl.store(
            out_ptr
            + b * stride_ob
            + c * stride_oc
            + h * stride_oh
            + d_off * stride_od,
            cur.to(states_ptr.dtype.element_ty),
            mask=d_mask,
        )
        decay = tl.exp(
            tl.load(
                dA_cumsum_ptr
                + b * stride_db
                + h * stride_dh
                + c * stride_dc
                + last_l * stride_dl
            )
        )
        st = tl.load(
            states_ptr
            + b * stride_sb
            + c * stride_sc
            + h * stride_sh
            + d_off * stride_sd,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        cur = cur * decay + st

    tl.store(
        final_ptr + b * stride_fb + h * stride_fh + d_off * stride_fd,
        cur,
        mask=d_mask,
    )


@triton.jit
def _state_passing_4t_nomask_kernel(
    out_ptr,  # [B, nchunks, nheads, dim]   (states.dtype)
    final_ptr,  # [B, nheads, dim]            (float32)
    states_ptr,  # [B, nchunks, nheads, dim]   (states.dtype)
    dA_cumsum_ptr,  # [B, nheads, nchunks, L]     (float32)
    init_ptr,  # [B, nheads, dim] float32
    has_init,  # tl.int1: 1 if initial_states given else 0
    nheads,
    dim,
    L,
    stride_ob,
    stride_oc,
    stride_oh,
    stride_od,  # out
    stride_fb,
    stride_fh,
    stride_fd,  # final
    stride_sb,
    stride_sc,
    stride_sh,
    stride_sd,  # states
    stride_db,
    stride_dh,
    stride_dc,
    stride_dl,  # dA
    stride_ib,
    stride_ih,
    stride_id,  # init
    BLOCK_D: tl.constexpr,
    NCHUNKS: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # lane = batch * nheads + head
    pid_dg = tl.program_id(1)  # tile-group index (each group = 4 dim-tiles)

    b = pid_bh // nheads
    h = pid_bh % nheads

    base_d = pid_dg * (BLOCK_D * 4)
    offs = base_d + tl.arange(0, BLOCK_D)
    d0 = offs
    d1 = offs + BLOCK_D
    d2 = offs + 2 * BLOCK_D
    d3 = offs + 3 * BLOCK_D

    if has_init:
        cur0 = tl.load(
            init_ptr + b * stride_ib + h * stride_ih + d0 * stride_id
        ).to(tl.float32)
        cur1 = tl.load(
            init_ptr + b * stride_ib + h * stride_ih + d1 * stride_id
        ).to(tl.float32)
        cur2 = tl.load(
            init_ptr + b * stride_ib + h * stride_ih + d2 * stride_id
        ).to(tl.float32)
        cur3 = tl.load(
            init_ptr + b * stride_ib + h * stride_ih + d3 * stride_id
        ).to(tl.float32)
    else:
        cur0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        cur1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        cur2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        cur3 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    last_l = L - 1

    for c in tl.static_range(0, NCHUNKS):
        decay = tl.exp(
            tl.load(
                dA_cumsum_ptr
                + b * stride_db
                + h * stride_dh
                + c * stride_dc
                + last_l * stride_dl
            )
        )
        tl.store(
            out_ptr
            + b * stride_ob
            + c * stride_oc
            + h * stride_oh
            + d0 * stride_od,
            cur0.to(states_ptr.dtype.element_ty),
        )
        tl.store(
            out_ptr
            + b * stride_ob
            + c * stride_oc
            + h * stride_oh
            + d1 * stride_od,
            cur1.to(states_ptr.dtype.element_ty),
        )
        tl.store(
            out_ptr
            + b * stride_ob
            + c * stride_oc
            + h * stride_oh
            + d2 * stride_od,
            cur2.to(states_ptr.dtype.element_ty),
        )
        tl.store(
            out_ptr
            + b * stride_ob
            + c * stride_oc
            + h * stride_oh
            + d3 * stride_od,
            cur3.to(states_ptr.dtype.element_ty),
        )

        st0 = tl.load(
            states_ptr
            + b * stride_sb
            + c * stride_sc
            + h * stride_sh
            + d0 * stride_sd
        ).to(tl.float32)
        st1 = tl.load(
            states_ptr
            + b * stride_sb
            + c * stride_sc
            + h * stride_sh
            + d1 * stride_sd
        ).to(tl.float32)
        st2 = tl.load(
            states_ptr
            + b * stride_sb
            + c * stride_sc
            + h * stride_sh
            + d2 * stride_sd
        ).to(tl.float32)
        st3 = tl.load(
            states_ptr
            + b * stride_sb
            + c * stride_sc
            + h * stride_sh
            + d3 * stride_sd
        ).to(tl.float32)

        cur0 = cur0 * decay + st0
        cur1 = cur1 * decay + st1
        cur2 = cur2 * decay + st2
        cur3 = cur3 * decay + st3

    tl.store(final_ptr + b * stride_fb + h * stride_fh + d0 * stride_fd, cur0)
    tl.store(final_ptr + b * stride_fb + h * stride_fh + d1 * stride_fd, cur1)
    tl.store(final_ptr + b * stride_fb + h * stride_fh + d2 * stride_fd, cur2)
    tl.store(final_ptr + b * stride_fb + h * stride_fh + d3 * stride_fd, cur3)


_BLOCK_D_4T = 2048

_NUM_WARPS_4T = 2
_NUM_STAGES_4T = 2


_BLOCK_D_1T_MAX = 2048
_NUM_WARPS_1T = 4
_NUM_STAGES_1T = 1


def state_passing(states, dA_cumsum, initial_states=None):
    """Mamba2 SSD cross-chunk state passing (Triton).

    Signature matches the PyTorch ``reference(states, dA_cumsum,
    initial_states=None)`` exactly.
    """
    batch, nchunks, nheads, dim = states.shape
    L = dA_cumsum.shape[-1]

    out = torch.empty(
        batch, nchunks, nheads, dim, device=states.device, dtype=states.dtype
    )
    final_states = torch.empty(
        batch, nheads, dim, device=states.device, dtype=torch.float32
    )

    # Decay source promoted to float32 once (matches the reference's .float()).
    dA_f = (
        dA_cumsum
        if dA_cumsum.dtype == torch.float32
        else dA_cumsum.to(torch.float32)
    )

    if initial_states is None:
        init_ptr = states  # unused when has_init is False
        has_init = False
        stride_ib = stride_ih = stride_id = 0
    else:
        if initial_states.dtype == torch.float32:
            init_ptr = initial_states
        else:
            init_ptr = initial_states.to(torch.float32)
        has_init = True
        stride_ib = init_ptr.stride(0)
        stride_ih = init_ptr.stride(1)
        stride_id = init_ptr.stride(2)

    if dim >= 4 * _BLOCK_D_4T:
        grid = (batch * nheads, triton.cdiv(dim, _BLOCK_D_4T * 4))
        if dim % (4 * _BLOCK_D_4T) == 0:
            _state_passing_4t_nomask_kernel[grid](
                out,
                final_states,
                states,
                dA_f,
                init_ptr,
                has_init,
                nheads,
                dim,
                L,
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out.stride(3),
                final_states.stride(0),
                final_states.stride(1),
                final_states.stride(2),
                states.stride(0),
                states.stride(1),
                states.stride(2),
                states.stride(3),
                dA_f.stride(0),
                dA_f.stride(1),
                dA_f.stride(2),
                dA_f.stride(3),
                stride_ib,
                stride_ih,
                stride_id,
                num_warps=_NUM_WARPS_4T,
                num_stages=_NUM_STAGES_4T,
                BLOCK_D=_BLOCK_D_4T,
                NCHUNKS=nchunks,
            )
        else:
            _state_passing_4t_kernel[grid](
                out,
                final_states,
                states,
                dA_f,
                init_ptr,
                has_init,
                nheads,
                dim,
                L,
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out.stride(3),
                final_states.stride(0),
                final_states.stride(1),
                final_states.stride(2),
                states.stride(0),
                states.stride(1),
                states.stride(2),
                states.stride(3),
                dA_f.stride(0),
                dA_f.stride(1),
                dA_f.stride(2),
                dA_f.stride(3),
                stride_ib,
                stride_ih,
                stride_id,
                num_warps=_NUM_WARPS_4T,
                num_stages=_NUM_STAGES_4T,
                BLOCK_D=_BLOCK_D_4T,
                NCHUNKS=nchunks,
            )
    else:
        block_d = min(triton.next_power_of_2(dim), _BLOCK_D_1T_MAX)
        grid = (batch * nheads, triton.cdiv(dim, block_d))
        _state_passing_1t_kernel[grid](
            out,
            final_states,
            states,
            dA_f,
            init_ptr,
            has_init,
            nheads,
            dim,
            L,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            final_states.stride(0),
            final_states.stride(1),
            final_states.stride(2),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            dA_f.stride(0),
            dA_f.stride(1),
            dA_f.stride(2),
            dA_f.stride(3),
            stride_ib,
            stride_ih,
            stride_id,
            num_warps=_NUM_WARPS_1T,
            num_stages=_NUM_STAGES_1T,
            BLOCK_D=block_d,
            NCHUNKS=nchunks,
        )
    return out, final_states


__all__ = ["state_passing"]
