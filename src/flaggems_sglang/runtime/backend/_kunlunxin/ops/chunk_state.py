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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""chunk_state — Mamba2 SSM per-chunk hidden-state accumulation.

Computes, for each (batch, chunk, head), the ``[headdim, dstate]`` state block

    states[b, c, h, p, n] = sum_t  x[b, c, t, h, p] * B[b, c, t, g(h), n]
                                   * exp(dA_last[b,h,c] - dA_cumsum[b,h,c,t])
                                   * dt[b,h,c,t]

i.e. the einsum ``bcthp,bcthn->bchpn`` of ``x_c`` with a decay/dt-scaled ``B``.

Semantics (matches the PyTorch reference exactly):

    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size      = dt.shape
    _, _, ngroups, dstate          = B.shape
    ratio = nheads // ngroups

    x_c  = x.reshape(batch, nchunks, chunk_size, nheads, headdim).float()
    B_c  = B.reshape(batch, nchunks, chunk_size, ngroups, dstate).float()
    B_c  = B_c.repeat_interleave(ratio, dim=3)
    decay = exp(dA_cumsum[..., -1:] - dA_cumsum).float()
    scale = (decay * dt.float()).permute(0, 2, 3, 1)      # [b, c, t, h]
    Bs    = B_c * scale.unsqueeze(-1)
    states = einsum("bcthp,bcthn->bchpn", x_c, Bs)

Shapes:
    B         : [batch, seqlen, ngroups, dstate]   (SSM state-projection matrix)
    x         : [batch, seqlen, nheads, headdim]
    dt        : [batch, nheads, nchunks, chunk_size]
    dA_cumsum : [batch, nheads, nchunks, chunk_size]
    states    : [batch, nchunks, nheads, headdim, dstate]  (output, float32)

Scope:
    - seqlen == nchunks * chunk_size (chunk_size a power of 2)
    - head grouping: head h reads group  h // (nheads // ngroups)
    - inputs may be float32 / bfloat16 / float16; accumulation and output float32

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator (no torch.einsum, no
``_compiled`` handles, no torch.ops.* fused matmul) — the whole contraction is
done inside the Triton kernel so it is portable across supported chips.

Kunlun P800 (昆仑芯 XPU) backend notes:
    - XPU is exposed as CUDA devices (cc 8.6), triton 3.0.0 XPU backend.
    - ``warp_size = 1`` (no warps): ``num_warps`` is ignored, so it is set to 1.
    - ``tl.dot`` with fp32 operands (``allow_tf32=False``) works and is used for
      the contraction; the fp16 SDNN path is not needed for correctness here.
    - Two measured backend bugs this file works around:
      1. A fp32 vector that is *computed in-kernel* (``tl.exp`` of dt/dA) and
         then consumed by the 2D broadcast ``B * scale[:, None]`` corrupts
         (relative error ~O(1) once the decay > 1); the same scale vector
         *loaded from global memory* broadcasts bit-exactly.  The scale is
         therefore computed by a separate 1D kernel into a scratch buffer and
         the dot kernel loads it back.
      2. Element-wise vector ops (notably ``tl.exp``) on <= 32-wide vectors
         return wrong data for the tail lanes when multiple programs are in
         flight, while >= 64-wide is bit-exact.  The scale kernel therefore
         always works on a >= 64-lane tile (masked to ``chunk_size``), never on
         a ``chunk_size``-wide tile when ``chunk_size`` is 16/32.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_state_scale_kernel(
    dt_ptr,
    dA_ptr,
    scale_ptr,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_cs,
    stride_dA_batch,
    stride_dA_head,
    stride_dA_chunk,
    stride_dA_cs,
    nheads,
    nchunks,
    chunk_size: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Compute scale[b, c, h, t] = exp(dA_last - dA_cumsum[t]) * dt[t].

    Grid: (batch * nchunks * nheads,).  Pure 1D: each program loads the
    dt/dA vectors for one (b, c, h), computes the decay scale in fp32, and
    writes it to a contiguous scratch row.  This is a *separate* kernel from
    the dot so that the dot can consume the scale via an exact
    load-from-global-memory (see module docstring).  BLOCK_C >= 64 is always
    used (masked to chunk_size) to stay out of the backend's broken <=32-lane
    element-wise-op path.
    """
    pid = tl.program_id(0)
    h = pid % nheads
    tmp = pid // nheads
    c = tmp % nchunks
    b = tmp // nchunks

    rv = tl.arange(0, BLOCK_C)
    mask = rv < chunk_size

    dt_base = b * stride_dt_batch + h * stride_dt_head + c * stride_dt_chunk
    dA_base = b * stride_dA_batch + h * stride_dA_head + c * stride_dA_chunk

    dA_last = tl.load(dA_ptr + dA_base + (chunk_size - 1) * stride_dA_cs).to(
        tl.float32
    )
    dt_vec = tl.load(
        dt_ptr + dt_base + rv * stride_dt_cs, mask=mask, other=0.0
    ).to(tl.float32)
    dA_vec = tl.load(
        dA_ptr + dA_base + rv * stride_dA_cs, mask=mask, other=0.0
    ).to(tl.float32)

    scale = tl.exp(dA_last - dA_vec) * dt_vec
    tl.store(scale_ptr + pid * BLOCK_C + rv, scale, mask=mask)


@triton.jit
def _chunk_state_fwd_kernel(
    x_ptr,
    b_ptr,
    scale_ptr,
    states_ptr,
    headdim,
    dstate,
    chunk_size: tl.constexpr,
    nheads,
    nchunks,
    ratio,
    # x strides [batch, seqlen, nheads, headdim]
    stride_x_batch,
    stride_x_seq,
    stride_x_head,
    stride_x_hdim,
    # B strides [batch, seqlen, ngroups, dstate]
    stride_b_batch,
    stride_b_seq,
    stride_b_group,
    stride_b_dstate,
    # states strides [batch, nchunks, nheads, headdim, dstate]
    stride_s_batch,
    stride_s_chunk,
    stride_s_head,
    stride_s_hdim,
    stride_s_dstate,
    BLOCK_M: tl.constexpr,  # headdim tile
    BLOCK_N: tl.constexpr,  # dstate tile
    BLOCK_K: tl.constexpr,  # chunk_size (cumsum axis) tile
    BLOCK_C: tl.constexpr,  # scale scratch row width (>= chunk_size, >= 64)
):
    """One program computes a [BLOCK_M, BLOCK_N] tile of states[b, c, h].

    The per-token decay scale is *loaded* from the scratch written by
    _chunk_state_scale_kernel (never computed in-kernel — see module
    docstring), then row-broadcast over the B tile before the dot.
    """
    pid_bch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Decode the flattened (batch, chunk, head) index.
    h = pid_bch % nheads
    tmp = pid_bch // nheads
    c = tmp % nchunks
    b = tmp // nchunks
    g = h // ratio  # group index for this head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # headdim
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # dstate
    offs_k = tl.arange(0, BLOCK_K)  # position within chunk

    seq0 = c * chunk_size  # first token of this chunk in the seqlen axis

    # Base pointers for this (b, c, h) / group.
    x_base = b * stride_x_batch + seq0 * stride_x_seq + h * stride_x_head
    b_base = b * stride_b_batch + seq0 * stride_b_seq + g * stride_b_group

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, chunk_size, BLOCK_K):
        kk = k + offs_k
        mask_k = kk < chunk_size

        # x tile as [headdim(M), t(K)]  ->  A for the matmul
        x_ptrs = (
            x_ptr
            + x_base
            + offs_m[:, None] * stride_x_hdim
            + kk[None, :] * stride_x_seq
        )
        a = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < headdim) & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        # B tile as [t(K), dstate(N)]
        b_ptrs = (
            b_ptr
            + b_base
            + kk[:, None] * stride_b_seq
            + offs_n[None, :] * stride_b_dstate
        )
        bmat = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)

        # scale[t] = exp(dA_last - dA_cumsum[t]) * dt[t], precomputed by the
        # separate scale kernel; load + row-broadcast (exact on this backend).
        scale = tl.load(
            scale_ptr + pid_bch * BLOCK_C + kk,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        bmat = bmat * scale[:, None]

        acc += tl.dot(a, bmat, allow_tf32=False)

    # Store states[b, c, h, offs_m, offs_n]
    s_ptrs = (
        states_ptr
        + b * stride_s_batch
        + c * stride_s_chunk
        + h * stride_s_head
        + offs_m[:, None] * stride_s_hdim
        + offs_n[None, :] * stride_s_dstate
    )
    tl.store(
        s_ptrs,
        acc,
        mask=(offs_m[:, None] < headdim) & (offs_n[None, :] < dstate),
    )


def chunk_state(B, x, dt, dA_cumsum):
    """Mamba2 chunk-state: accumulate x * (decay*dt-scaled B) over each chunk.

    Args:
        B:         [batch, seqlen, ngroups, dstate] SSM state-projection matrix.
        x:         [batch, seqlen, nheads, headdim].
        dt:        [batch, nheads, nchunks, chunk_size].
        dA_cumsum: [batch, nheads, nchunks, chunk_size].

    Returns:
        [batch, nchunks, nheads, headdim, dstate] float32 tensor.
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    ratio = nheads // ngroups

    states = torch.empty(
        (batch, nchunks, nheads, headdim, dstate),
        device=x.device,
        dtype=torch.float32,
    )

    BLOCK_M = 64
    BLOCK_N = 64
    # BLOCK_K must be >= 16 for tl.dot; chunk_size is a power of 2 (>=16 here).
    BLOCK_K = chunk_size if chunk_size <= 64 else 64
    # Scale tile: always >= 64 lanes to avoid the backend's broken <=32-lane
    # element-wise-op path (chunk_size is a power of two, so this is 64 for
    # chunk_size 16/32 and chunk_size itself otherwise).
    BLOCK_C = chunk_size if chunk_size >= 64 else 64

    # Per-(b, c, h) decay scale, computed by a separate 1D kernel into a
    # contiguous scratch so the dot kernel can load it back exactly.
    scale_scratch = torch.empty(
        batch * nchunks * nheads,
        BLOCK_C,
        device=x.device,
        dtype=torch.float32,
    )
    _chunk_state_scale_kernel[(batch * nchunks * nheads,)](
        dt,
        dA_cumsum,
        scale_scratch,
        dt.stride(0),
        dt.stride(1),
        dt.stride(2),
        dt.stride(3),
        dA_cumsum.stride(0),
        dA_cumsum.stride(1),
        dA_cumsum.stride(2),
        dA_cumsum.stride(3),
        nheads,
        nchunks,
        chunk_size,
        BLOCK_C=BLOCK_C,
        num_warps=1,  # ignored on Kunlun (warp_size = 1)
    )

    grid = (
        batch * nchunks * nheads,
        triton.cdiv(headdim, BLOCK_M),
        triton.cdiv(dstate, BLOCK_N),
    )

    _chunk_state_fwd_kernel[grid](
        x,
        B,
        scale_scratch,
        states,
        headdim,
        dstate,
        chunk_size,
        nheads,
        nchunks,
        ratio,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        B.stride(3),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        states.stride(4),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=1,  # ignored on Kunlun (warp_size = 1)
    )
    return states


__all__ = ["chunk_state"]
