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

"""Hybrid-layout causal 1-D convolution (Kunlun XPU).

v3: Hybrid layout causal_conv1d for Kunlun XPU.

Profiling on this backend revealed a sharp asymmetry for *scattered*
reads vs writes: writing 2048 short rows of 128 elements costs ~64us,
but *reading* the same 2048 short rows costs ~476us (~7x more).  v2
processes one channel row per program (grid = dim x t_blocks), so when
total_len is small each program issues only a few hundred contiguous
elements and the reads are effectively scattered across 2048 programs
-- which is exactly the slow case.  The bf16-1seq-128 benchmark
(dim=2048, total_len=128) therefore ran at ~0.5x of the PyTorch
reference.

v3 fixes the small-total_len case with a *transposed* layout:

* For small total_len we transpose x (and the output) so that the
  channel axis becomes contiguous.  A program then handles one time
  position ``t`` and a BLOCK_D-wide block of channels, and every load
  ``x_t[t+k, d0:d0+BLOCK_D]`` (and the weight
  ``w_t[k, d0:d0+BLOCK_D]``, and the store) is a single long stride-1
  contiguous access.  The depthwise conv is still computed entirely
  inside the Triton kernel; the transpose is pure data movement (the
  same category as the F.pad that v2 already used), not a torch
  conv/activ op.

* For large total_len the channel rows are already long enough that the
  row-major v2 kernel reads/writes efficiently, so we keep it unchanged
  (the transpose would add ~2 full copies with no benefit there).

Both paths: all loads/stores are 1-D stride-1 affine (base + arange +
const).  No reshape / trans / 2D load / block pointer / int64.  The
causal boundary for multi-sequence inputs is enforced with a
precomputed ``lp`` (local position) mask applied via tl.where on the
loaded value -- never on the address -- so loads stay affine.  The
single-sequence fast path skips the mask entirely.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# total_len below this uses the transposed kernel (contiguous channel
# reads); at/above it the row-major kernel already reads long contiguous
# rows.
_TRANSPOSE_MAX_LEN = 2048
# Cap the channel-block width of the transposed kernel (wide blocks
# saturate the memory system; capping keeps registers/SRAM in check).
_TRANSPOSE_MAX_BLOCK_D = 2048


@triton.jit
def _cc1d_kernel(
    xp_ptr,  # [dim, total_len + WIDTH - 1] flat, left-padded, contiguous
    w_ptr,  # [dim, WIDTH] flat, contiguous
    bias_ptr,  # [dim] float32
    lp_ptr,  # [total_len] int32: local position within its sequence
    out_ptr,  # [dim, total_len] flat
    total_len,
    row_len,  # int = total_len + WIDTH - 1 (stride of a padded row)
    WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_ACT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    d = tl.program_id(0)  # channel
    pid_t = tl.program_id(1)  # time block
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < total_len

    if HAS_MASK:
        lp = tl.load(lp_ptr + t_offs, mask=t_mask, other=0)

    xbase = xp_ptr + d * row_len
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)
    for k in tl.static_range(WIDTH):
        w_k = tl.load(w_ptr + d * WIDTH + k).to(tl.float32)
        # affine stride-1 read of the padded row; always in-bounds
        # thanks to the host-side left pad (never a negative / clamped
        # address).
        xv = tl.load(xbase + t_offs + k, mask=t_mask, other=0.0).to(tl.float32)
        if HAS_MASK:
            valid = (lp + k) >= (WIDTH - 1)
            xv = tl.where(valid, xv, 0.0)
        acc = acc + w_k * xv

    if HAS_BIAS:
        acc = acc + tl.load(bias_ptr + d).to(tl.float32)
    if HAS_ACT:
        acc = acc * tl.sigmoid(acc)

    tl.store(
        out_ptr + d * total_len + t_offs,
        acc.to(xp_ptr.dtype.element_ty),
        mask=t_mask,
    )


@triton.jit
def _cc1d_t_kernel(
    x_t_ptr,  # [total_len + WIDTH - 1, dim] flat (transposed, padded x)
    w_t_ptr,  # [WIDTH, dim] flat (transposed weight)
    bias_ptr,  # [dim] float32
    lp_ptr,  # [total_len] int32: local position within its sequence
    out_t_ptr,  # [total_len, dim] flat (transposed output)
    dim,
    total_len,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_ACT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    t = tl.program_id(0)  # time position
    pid_d = tl.program_id(1)  # channel block
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < dim

    if HAS_MASK:
        lp = tl.load(lp_ptr + t)  # scalar local position

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k in tl.static_range(WIDTH):
        # contiguous stride-1 reads across channels (the whole point of
        # the transposed layout); base + arange + const, so affine.
        w_k = tl.load(w_t_ptr + k * dim + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        xv = tl.load(
            x_t_ptr + (t + k) * dim + d_offs, mask=d_mask, other=0.0
        ).to(tl.float32)
        if HAS_MASK:
            valid = (lp + k) >= (WIDTH - 1)
            xv = tl.where(valid, xv, 0.0)
        acc = acc + w_k * xv

    if HAS_BIAS:
        acc = acc + tl.load(bias_ptr + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
    if HAS_ACT:
        acc = acc * tl.sigmoid(acc)

    tl.store(
        out_t_ptr + t * dim + d_offs,
        acc.to(x_t_ptr.dtype.element_ty),
        mask=d_mask,
    )


def _make_lp(lens, total_len, device):
    """Local position within its sequence for every global position."""
    parts = [
        torch.arange(0, int(seq_len), dtype=torch.int32) for seq_len in lens
    ]
    return torch.cat(parts).to(device)


def causal_conv1d_fn(
    x, weight, bias, query_start_loc, seq_lens_cpu, activation="silu"
):
    """Triton depthwise causal 1-D convolution for continuous-batching
    sequences."""
    assert x.is_contiguous(), "x must be contiguous"
    dim, total_len = x.shape
    width = weight.shape[1]
    weight = weight.contiguous()

    if isinstance(seq_lens_cpu, torch.Tensor):
        lens = seq_lens_cpu.cpu().tolist()
    else:
        lens = [int(seq_len) for seq_len in seq_lens_cpu]
    n_seqs = len(lens)

    has_mask = n_seqs > 1
    if has_mask:
        lp = _make_lp(lens, total_len, x.device)
    else:
        # unused when HAS_MASK=False; a valid 1-element tensor keeps the
        # pointer argument well-formed without ever being dereferenced.
        lp = torch.zeros(1, dtype=torch.int32, device=x.device)

    if bias is not None:
        bias_f32 = bias.float().contiguous()
        has_bias = True
    else:
        bias_f32 = torch.zeros(dim, dtype=torch.float32, device=x.device)
        has_bias = False

    has_act = activation in ("silu", "swish")

    if total_len < _TRANSPOSE_MAX_LEN:
        # Transposed layout: contiguous channel reads for short rows.
        BLOCK_D = 1
        while BLOCK_D < dim:
            BLOCK_D *= 2
        BLOCK_D = min(BLOCK_D, _TRANSPOSE_MAX_BLOCK_D)

        xp = F.pad(x, (width - 1, 0))
        x_t = xp.t().contiguous()
        w_t = weight.t().contiguous()
        out_t = torch.empty(total_len, dim, device=x.device, dtype=x.dtype)

        dim_blocks = triton.cdiv(dim, BLOCK_D)
        grid = (total_len, dim_blocks)
        _cc1d_t_kernel[grid](
            x_t,
            w_t,
            bias_f32,
            lp,
            out_t,
            dim,
            total_len,
            WIDTH=width,
            BLOCK_D=BLOCK_D,
            HAS_ACT=has_act,
            HAS_BIAS=has_bias,
            HAS_MASK=has_mask,
            num_warps=4,
            num_stages=1,
        )
        return out_t.t().contiguous()

    # Row-major layout: a whole (wide) channel row per program.
    xp = F.pad(x, (width - 1, 0))
    row_len = total_len + width - 1

    out = torch.empty_like(x)

    BLOCK_T = 1024
    while BLOCK_T < total_len and BLOCK_T < 8192:
        BLOCK_T *= 2

    n_t_blocks = triton.cdiv(total_len, BLOCK_T)
    grid = (dim, n_t_blocks)

    _cc1d_kernel[grid](
        xp,
        weight,
        bias_f32,
        lp,
        out,
        total_len,
        row_len,
        WIDTH=width,
        BLOCK_T=BLOCK_T,
        HAS_ACT=has_act,
        HAS_BIAS=has_bias,
        HAS_MASK=has_mask,
        num_warps=4,
        num_stages=1,
    )

    return out


__all__ = ["causal_conv1d_fn"]
