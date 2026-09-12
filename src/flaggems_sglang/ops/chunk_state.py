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
    - seqlen == nchunks * chunk_size (chunk_size a power of 2, >= 16)
    - head grouping: head h reads group  h // (nheads // ngroups)
    - inputs may be float32 / bfloat16 / float16; accumulation and output float32

Optimization notes (v3):
    - Native-dtype tensor-core tl.dot (bf16/fp16 MMA, fp32 accumulator) with the
      decay*dt scale fused on *x* (the [M, K] tile): with headdim <= dstate the
      scaling/rounding work is minimized and B is consumed in native dtype with
      no conversion at all (kept from v2).
    - Launch-config re-tune from an on-target sweep (MetaX C550, 64 KiB smem/SM):
      a *uniform* (BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=2)
      is best across every production shape, including chunk_size=256 where v2's
      (BLOCK_K=64, 8 warps) was 1.35-1.42x slower.  Small BLOCK_K keeps the tl.dot
      smem footprint at ~24 KiB so 2 CTAs fit per SM (latency hiding), and the
      short K loop software-pipelines (num_stages=2, which the MACA backend does
      not apply by default).
    - headdim >= 128 is handled by splitting M into two 64-wide tiles (measured
      equal-or-better than a single whole-chunk 128-wide dot, which needs 64 KiB
      of dot staging and runs one CTA/SM with no pipelining).
    - Innermost tensor strides are known to be 1 for the contiguous tensors
      produced by this launch path and are passed as constexpr, letting Triton
      prove contiguity and emit vectorized 128-bit loads.
    - Dead ends measured on this chip (kept out): grouping several heads per
      program to reuse the B tile in smem costs far more in occupancy than it
      saves in B traffic; eviction-policy hints change nothing.

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator (no torch.einsum, no
``_compiled`` handles, no torch.ops.* fused matmul) — the whole contraction is
done inside the Triton kernel so it is portable across supported chips.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_state_fwd_kernel(
    x_ptr,
    b_ptr,
    dt_ptr,
    dA_ptr,
    states_ptr,
    # sizes
    headdim,
    dstate,
    nheads,
    nchunks,
    ratio,
    # x strides [batch, seqlen, nheads, headdim]
    stride_x_batch,
    stride_x_seq,
    stride_x_head,
    # B strides [batch, seqlen, ngroups, dstate]
    stride_b_batch,
    stride_b_seq,
    stride_b_group,
    # dt / dA strides [batch, nheads, nchunks, chunk_size]
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    # states strides [batch, nchunks, nheads, headdim, dstate]
    stride_s_batch,
    stride_s_chunk,
    stride_s_head,
    stride_s_hdim,
    # compile-time geometry
    chunk_size: tl.constexpr,
    BLOCK_M: tl.constexpr,  # headdim tile
    BLOCK_N: tl.constexpr,  # dstate tile
    BLOCK_K: tl.constexpr,  # chunk axis tile
    DOT_DTYPE: tl.constexpr,  # input dtype of tl.dot (native input dtype)
):
    """One program computes a [BLOCK_M, BLOCK_N] tile of states[b, c, h]."""
    pid_bch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Decode the flattened (batch, chunk, head) index.
    h = pid_bch % nheads
    tmp = pid_bch // nheads
    c = tmp % nchunks
    b = tmp // nchunks
    g = h // ratio  # group index for this head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # headdim (stride 1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # dstate (stride 1)
    offs_k = tl.arange(0, BLOCK_K)  # position within chunk

    seq0 = c * chunk_size  # first token of this chunk in the seqlen axis

    # Base pointers for this (b, c, h) / group.
    x_base = b * stride_x_batch + seq0 * stride_x_seq + h * stride_x_head
    b_base = b * stride_b_batch + seq0 * stride_b_seq + g * stride_b_group
    dt_base = b * stride_dt_batch + h * stride_dt_head + c * stride_dt_chunk

    # dA_last = dA_cumsum[b, h, c, chunk_size - 1]  (scalar for this program)
    dA_last = tl.load(dA_ptr + dt_base + (chunk_size - 1) * 1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, chunk_size, BLOCK_K):
        kk = k + offs_k
        mask_k = kk < chunk_size

        # x tile as [headdim(M), t(K)]  ->  A of the matmul (contig in M).
        x_ptrs = (
            x_ptr + x_base + offs_m[:, None] * 1 + kk[None, :] * stride_x_seq
        )
        a = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < headdim) & mask_k[None, :],
            other=0.0,
        )

        # B tile as [t(K), dstate(N)] (contig in N), native dtype.
        b_ptrs = (
            b_ptr + b_base + kk[:, None] * stride_b_seq + offs_n[None, :] * 1
        )
        bmat = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & (offs_n[None, :] < dstate),
            other=0.0,
        )

        # scale[t] = exp(dA_last - dA_cumsum[t]) * dt[t]   (fp32, [BLOCK_K])
        dt_vals = tl.load(
            dt_ptr + dt_base + kk * 1, mask=mask_k, other=0.0
        ).to(tl.float32)
        dA_vals = tl.load(
            dA_ptr + dt_base + kk * 1, mask=mask_k, other=0.0
        ).to(tl.float32)
        scale = tl.exp(dA_last - dA_vals) * dt_vals

        if DOT_DTYPE == tl.float32:
            # fp32 inputs: keep everything in fp32, IEEE matmul.
            a = a.to(tl.float32) * scale[None, :]
            bmat = bmat.to(tl.float32)
            acc = tl.dot(a, bmat, acc, allow_tf32=False)
        else:
            # bf16/fp16 inputs: scale x in fp32 (reference-exact products),
            # round once back to native dtype, and use the tensor-core MMA
            # path (fp32 accumulator). B stays in native dtype untouched.
            a = (a.to(tl.float32) * scale[None, :]).to(DOT_DTYPE)
            bmat = bmat.to(DOT_DTYPE)
            acc = tl.dot(a, bmat, acc)

    # Store states[b, c, h, offs_m, offs_n] (contig in dstate).
    s_ptrs = (
        states_ptr
        + b * stride_s_batch
        + c * stride_s_chunk
        + h * stride_s_head
        + offs_m[:, None] * stride_s_hdim
        + offs_n[None, :] * 1
    )
    tl.store(
        s_ptrs,
        acc,
        mask=(offs_m[:, None] < headdim) & (offs_n[None, :] < dstate),
    )


def _pick_config(headdim, dstate, chunk_size, is_fp32):
    """Pick (BLOCK_M, BLOCK_N, BLOCK_K, num_warps) tuned for the target.

    Swept on the MetaX C550 (bf16, fp32 accum) across all production shapes:
    the uniform (M64, N128, K32, 4 warps) wins everywhere.  Small K tiles keep
    the tl.dot smem footprint small (2 CTAs/SM on this 64 KiB/SM chip) and let
    the K loop software-pipeline; larger tiles / more warps measured 1.35-1.42x
    slower on chunk_size=256 and no faster on chunk_size=128.
    """
    block_m = min(64, headdim)
    block_n = 128 if dstate >= 128 else (64 if dstate >= 64 else dstate)
    block_n = min(block_n, dstate)
    block_k = min(32, chunk_size)
    num_warps = 4
    # Never tile beyond the problem size (tl.dot needs M, N, K >= 16).
    block_m = max(block_m, 16)
    block_n = max(block_n, 16)
    block_k = max(block_k, 16)
    if is_fp32:
        # fp32 dot operands are staged in shared memory at 4B/elem; keep two
        # num_stages=2 buffers under the 64 KiB hardware limit:
        #   8 * block_k * (block_m + block_n) <= 65536.
        max_k = 8192 // (block_m + block_n)
        block_k = min(block_k, max_k - (max_k % 16))
    return block_m, block_n, block_k, num_warps


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

    tl_dtype = {
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
    }.get(x.dtype)
    if tl_dtype is None:
        raise ValueError(f"unsupported input dtype {x.dtype}")

    block_m, block_n, block_k, num_warps = _pick_config(
        headdim, dstate, chunk_size, tl_dtype == tl.float32
    )

    grid = (
        batch * nchunks * nheads,
        triton.cdiv(headdim, block_m),
        triton.cdiv(dstate, block_n),
    )

    _chunk_state_fwd_kernel[grid](
        x,
        B,
        dt,
        dA_cumsum,
        states,
        headdim,
        dstate,
        nheads,
        nchunks,
        ratio,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        dt.stride(0),
        dt.stride(1),
        dt.stride(2),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        chunk_size=chunk_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        DOT_DTYPE=tl_dtype,
        num_warps=num_warps,
        num_stages=2,
    )
    return states


__all__ = ["chunk_state"]
