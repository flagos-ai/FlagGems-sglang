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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 32


@triton.jit
def _chunk_state_varlen_fused_ieee_kernel(
    x_ptr,
    b_ptr,
    states_ptr,
    dt_ptr,
    da_ptr,
    cu_ptr,
    hdim,
    dstate,
    chunk_size,
    nheads_ngroups_ratio,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_b_seqlen,
    stride_b_group,
    stride_b_dstate,
    stride_states_batch,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_da_head,
    stride_da_chunk,
    stride_da_csize,
    NCHUNKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    start_idx = tl.load(cu_ptr + pid_b)
    end_idx = tl.load(cu_ptr + pid_b + 1)
    pid_c = tl.minimum(
        tl.maximum((end_idx - 1) // chunk_size, 0),
        NCHUNKS - 1,
    )
    chunk_base = pid_c * chunk_size
    seg_end = tl.minimum(tl.maximum(end_idx - chunk_base, 0), chunk_size)
    seg_start = tl.minimum(tl.maximum(start_idx - chunk_base, 0), seg_end)

    seq_len = seg_end - seg_start

    token_base = chunk_base + seg_start
    x_ptr += token_base * stride_x_seqlen + pid_h * stride_x_head
    b_ptr += (
        token_base * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_group
    )
    dt_ptr += (
        pid_h * stride_dt_head
        + pid_c * stride_dt_chunk
        + seg_start * stride_dt_csize
    )
    da_chunk_ptr = da_ptr + pid_h * stride_da_head + pid_c * stride_da_chunk
    da_ptr = da_chunk_ptr + seg_start * stride_da_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    da_last = tl.load(
        da_chunk_ptr + tl.maximum(seg_end - 1, 0) * stride_da_csize
    ).to(tl.float32)
    da_ptrs = da_ptr + offs_k * stride_da_csize

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, seq_len, BLOCK_SIZE_K):
        valid_k = offs_k < seq_len - k
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & valid_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=valid_k[:, None] & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dt = tl.load(dt_ptrs, mask=valid_k, other=0.0).to(tl.float32)
        da = tl.load(da_ptrs, mask=valid_k, other=0.0).to(tl.float32)
        scale = tl.exp(da_last - da) * dt
        b *= scale[:, None]
        acc += tl.dot(x.to(tl.float32), b, input_precision="ieee")

        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        da_ptrs += BLOCK_SIZE_K * stride_da_csize

    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim
        + offs_n[None, :] * stride_states_dstate
    )
    states = acc.to(states_ptr.dtype.element_ty)
    tl.store(
        states_ptrs,
        states,
        mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate),
    )


def chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.numel() - 1
    ratio = nheads // ngroups

    states = torch.empty(
        (batch, nheads, headdim, dstate),
        device=x.device,
        dtype=chunk_states.dtype,
    )
    grid = (
        triton.cdiv(headdim, _BLOCK_M) * triton.cdiv(dstate, _BLOCK_N),
        batch,
        nheads,
    )
    _chunk_state_varlen_fused_ieee_kernel[grid](
        x,
        B,
        states,
        dt,
        dA_cumsum,
        cu_seqlens,
        headdim,
        dstate,
        chunk_size,
        ratio,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        dt.stride(0),
        dt.stride(1),
        dt.stride(2),
        dA_cumsum.stride(0),
        dA_cumsum.stride(1),
        dA_cumsum.stride(2),
        nchunks,
        _BLOCK_M,
        _BLOCK_N,
        _BLOCK_K,
        num_warps=4,
        num_stages=1,
    )
    return states


__all__ = ["chunk_state_varlen"]
