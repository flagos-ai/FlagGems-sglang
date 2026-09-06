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

"""Depthwise causal 1-D convolution over continuous-batched sequences
(Triton) -- Huawei Ascend NPU specialization.

Computes, for each sequence segment ``[start, end)`` of a packed ``x`` of
shape ``(dim, total_t)`` (segments given by ``query_start_loc``), each
channel ``d`` and each within-segment position ``t``:

    conv[d, t] = sum_{k=0}^{WIDTH-1} weight[d, k] * x_pad[d, t + k]
    out[d, t]  = silu(conv[d, t] + bias[d])          # activation optional

where ``x_pad`` is the segment's channel row left-padded with
``WIDTH-1`` zeros (causal -- no future or cross-sequence lookback). All
math is done in fp32 and cast back to the input dtype, matching the
reference.

This is the chip-agnostic Triton baseline: the kernel targets
portability across accelerators; this box (Ascend 910B4 NPU) is only
used to measure performance. Optimization uses generic Triton levers
(launch config, memory access, tile/block choice, fused compute) and
does not call any vendor fused op.

Grid design (why time is on grid.x, not n_seqs)
-----------------------------------------------
A naive 3-D grid ``(n_seqs, dim/BLOCK_D, time/BLOCK_T)`` overflows
backend grid caps and compiles slowly. Instead we grid over the
**packed time axis**, which is dense regardless of how the batch
splits into sequences:

* ``grid.x = cdiv(total_t, BLOCK_T)`` -- packed time, the naturally big
  axis.
* ``grid.y = cdiv(dim, BLOCK_D)`` -- channel blocks.

Sequence boundaries do not live in the grid. The host builds
``seg_start`` -- a ``(total_t,)`` int32 vector giving, for every packed
position, the ``start`` of the segment it belongs to -- so the kernel
applies causal (left) zero-padding per segment with a simple
``src >= seg_start`` test. This costs one cheap ``repeat_interleave`` on
the host and one extra ``(BLOCK_T,)`` load per program.

Ascend notes
------------
* **Keep tiles small enough for the unified buffer (UB, ~192 KB).** A
  ``BLOCK_D x BLOCK_T`` fp32 accumulator plus its transient loads
  (times the multi-buffer factor) must fit the UB; oversized tiles fail
  with "ub overflow". ``BLOCK_D=32, BLOCK_T=128`` (16 KB acc) is a
  safe, portable starting point.
* **Keep the time-domain math 1-D until the final load.** Build ``src``
  / ``valid_t`` as 1-D ``(BLOCK_T,)`` vectors and expand to 2-D only
  right before the ``tl.load``; broadcasting a 1-D value inside
  ``tl.where`` on a 2-D condition can silently produce a 3-D tile that
  fails to compile.

``seq_lens_cpu`` may be a ``torch.Tensor`` or a plain Python ``list``.

``TOTAL_T`` and ``DIM`` are passed as **runtime** args
(``do_not_specialize``), not ``constexpr``, so a single compiled kernel
is reused across all batch layouts and channel counts for a given
``(WIDTH, BLOCK_T, BLOCK_D, bias, activation)`` -- without this, every
distinct shape would trigger a fresh (slow) Ascend compile.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["TOTAL_T", "DIM"])
def _causal_conv1d_fn_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    seg_start_ptr,
    out_ptr,
    TOTAL_T,  # runtime: packed token count (do_not_specialize)
    DIM,  # runtime: channel count (do_not_specialize)
    WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_ACTIVATION: tl.constexpr,
):
    # grid = (num_t_blocks, num_d_blocks): packed time on grid.x,
    # channel blocks on grid.y. Each program owns a BLOCK_D x BLOCK_T
    # tile.
    pid_t = tl.program_id(0)  # packed-time block
    pid_d = tl.program_id(1)  # channel block

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # (BLOCK_T,) pos
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # (BLOCK_D,)
    mask_t = offs_t < TOTAL_T
    mask_d = offs_d < DIM

    # seg_start[t] = start of the segment that packed position t belongs
    # to.
    seg_start = tl.load(seg_start_ptr + offs_t, mask=mask_t, other=0)

    acc = tl.zeros((BLOCK_D, BLOCK_T), dtype=tl.float32)
    for k in range(WIDTH):
        # Output position t reads x[..., t - (WIDTH-1) + k], zero-padded
        # where that index falls left of the segment start (causal
        # padding). Keep 1-D; expand to 2-D only at the load below.
        src = offs_t - (WIDTH - 1) + k  # (BLOCK_T,)
        valid_t = mask_t & (src >= seg_start)  # (BLOCK_T,)
        src = tl.where(valid_t, src, 0)  # keep masked lanes in-bounds

        addr = offs_d[:, None] * TOTAL_T + src[None, :]  # (D, T)
        valid = valid_t[None, :] & mask_d[:, None]  # (D, T)
        x = tl.load(x_ptr + addr, mask=valid, other=0.0).to(tl.float32)
        w = tl.load(
            weight_ptr + offs_d * WIDTH + k,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        acc += x * w[:, None]

    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        acc += b[:, None]

    if HAS_ACTIVATION:
        acc = acc / (1.0 + tl.exp(-acc))  # silu(x) = x / (1 + exp(-x))

    store_mask = mask_t[None, :] & mask_d[:, None]
    tl.store(
        out_ptr + offs_d[:, None] * TOTAL_T + offs_t[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=store_mask,
    )


def causal_conv1d_fn(
    x, weight, bias, query_start_loc, seq_lens_cpu, activation="silu"
):
    """Fused causal depthwise conv1d + optional silu.

    Signature matches the reference.
    """
    dim, total_t = x.shape
    width = weight.shape[1]

    # seq_lens_cpu may be a torch.Tensor or a plain Python list.
    if isinstance(seq_lens_cpu, torch.Tensor):
        lens = seq_lens_cpu.to(x.device, torch.int32)
    else:
        lens = torch.tensor(seq_lens_cpu, device=x.device, dtype=torch.int32)

    # Per packed-position segment start: repeat each segment's start by
    # its length.
    seg_start = torch.repeat_interleave(
        query_start_loc[:-1].to(torch.int32), lens
    ).to(torch.int32)

    out = torch.empty_like(x)
    if total_t == 0 or dim == 0:
        return out

    # Small, UB-safe tile: BLOCK_D x BLOCK_T fp32 accumulator = 16 KB,
    # well within the ~192 KB Ascend unified buffer even with
    # multi-buffering.
    block_t = 128
    block_d = 32
    grid = (triton.cdiv(total_t, block_t), triton.cdiv(dim, block_d))

    _causal_conv1d_fn_kernel[grid](
        x,
        weight,
        x if bias is None else bias,
        seg_start,
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
