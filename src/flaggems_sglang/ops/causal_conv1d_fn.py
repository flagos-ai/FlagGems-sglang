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

"""Generic depthwise causal 1-D convolution over continuous-batched
sequences (Triton).

Computes, for each sequence segment ``[start, end)`` of a packed ``x`` of
shape ``(dim, total_t)`` (segments given by ``query_start_loc``), each
channel ``d`` and each within-sequence position ``t``:

    conv[d, t] = sum_{k=0}^{WIDTH-1} weight[d, k] * x_pad[d, t + k]
    out[d, t]  = silu(conv[d, t] + bias[d])      # activation optional

where ``x_pad`` is the segment's channel row left-padded with
``WIDTH-1`` zeros (causal -- no future or cross-sequence lookback). All
math is done in fp32 and cast back to the input dtype, matching the
reference.

This is the chip-agnostic generic fallback: a simple 3-D grid
``(n_seqs, dim/BLOCK_D, time/BLOCK_T)`` with one program per
``(sequence, channel-block, time-block)``. Vendor tiers may override
it with backend-specialized kernels.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def causal_conv1d_fn_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    query_start_loc_ptr,
    out_ptr,
    TOTAL_T: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_ACTIVATION: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_t = tl.program_id(2)

    # All channels in this program share the same continuous-batching
    # segment.
    start = tl.load(query_start_loc_ptr + pid_seq)
    end = tl.load(query_start_loc_ptr + pid_seq + 1)
    seq_len = end - start

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)[:, None]
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)[None, :]
    valid_d = offs_d < DIM
    valid_out = valid_d & (offs_t < seq_len)

    acc = tl.zeros((BLOCK_D, BLOCK_T), dtype=tl.float32)
    for k in range(WIDTH):
        # Output position t reads x[t - (WIDTH-1) + k], zero-padded where
        # that index falls left of the segment start (causal padding).
        valid_in = valid_out & (offs_t + k >= WIDTH - 1)
        src_t = start + offs_t + k - (WIDTH - 1)
        x = tl.load(
            x_ptr + offs_d * TOTAL_T + src_t,
            mask=valid_in,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            weight_ptr + offs_d * WIDTH + k,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        acc += x * w

    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_d, mask=valid_d, other=0.0).to(tl.float32)
        acc += b

    if HAS_ACTIVATION:
        acc = acc / (1.0 + tl.exp(-acc))

    tl.store(
        out_ptr + offs_d * TOTAL_T + start + offs_t,
        acc,
        mask=valid_out,
    )


def causal_conv1d_fn(
    x, weight, bias, query_start_loc, seq_lens_cpu, activation="silu"
):
    """Gated causal depthwise conv1d. Signature matches the reference."""
    dim, total_t = x.shape
    width = weight.shape[1]
    n_seqs = len(seq_lens_cpu)
    if isinstance(seq_lens_cpu, torch.Tensor):
        max_seq_len = int(seq_lens_cpu.max().item())
    else:
        max_seq_len = int(max(seq_lens_cpu))

    block_t = 128
    block_d = 4
    out = torch.empty_like(x)

    causal_conv1d_fn_kernel[
        (
            n_seqs,
            triton.cdiv(dim, block_d),
            triton.cdiv(max_seq_len, block_t),
        )
    ](
        x,
        weight,
        x if bias is None else bias,
        query_start_loc,
        out,
        TOTAL_T=total_t,
        DIM=dim,
        WIDTH=width,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        HAS_BIAS=bias is not None,
        HAS_ACTIVATION=activation in ("silu", "swish"),
        num_warps=4,
    )
    return out


__all__ = ["causal_conv1d_fn"]
