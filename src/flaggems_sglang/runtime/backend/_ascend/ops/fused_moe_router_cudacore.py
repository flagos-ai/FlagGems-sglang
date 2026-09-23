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

"""Triton kernels for moe/fused_moe_router_cudacore (Ascend).

Two launches, matching the reference semantics exactly: a tiled GEMM kernel
writes the ``[BS, E]`` fp32 logits with the soft-cap and correction bias fused
into its epilogue, then a per-token kernel does the global softmax over all
experts plus the top-k selection and weight gather.

The logits kernel is a ``tl.dot`` GEMM over ``[BLOCK_M, BLOCK_E]`` output
tiles, replacing the previous broadcast-multiply reduction
(``tl.sum(x[:, None, :] * w[None, :, :], axis=2)`` over ``BLOCK_H = 8`` slices,
which needed 512 K-iterations for the ``H = 4096`` bench shape). Three Ascend
constraints shape the port:

  * Feeding the dot native bf16/f16 tiles trips a hardware MTE fault, so the
    tiles are upcast to float32 and the precision is selected with
    ``input_precision``: ``"ieee"`` for float32 input (the reference runs a
    strict fp32 matmul and the tolerance is 1e-4), ``"hf32"`` otherwise. The
    source data is bf16/f16 (<=8 mantissa bits), so hf32's ~10 mantissa bits
    discard nothing the input dtype had not already dropped, and it runs
    substantially faster. Same approach as ``sgemm_lora_a`` and the sibling
    ``fused_moe_router_tensorcore``.
  * ``BLOCK_H`` (the K tile) is capped at 128 once ``BLOCK_E`` covers a full
    256-wide expert axis — the wider dot does not fit the unified buffer and
    fails to compile — and at 256 otherwise.
  * ``tl.range(..., num_stages=N)`` asserts ``N <= 2`` on this backend, so the
    K-loop pipelining depth is capped at ``_MAX_KLOOP_STAGES``.

``w`` is loaded as a coalesced ``[BLOCK_E, BLOCK_H]`` tile (hidden dim
innermost) and transposed for the dot rather than gathered column-wise with a
stride-``H`` access.

Tile sizes come from ``_logits_config``, a pure function of the shapes; the
tiles follow the sibling tensorcore GEMM's measured sweep on this device for
the same ``[BS, 4096] @ [4096, 256]`` shape: an N-split tile for the
compute-bound ``BS > 512`` regime, a full-expert-axis ``[BLOCK_M, 256]`` tile
with ``BLOCK_H = 128`` below it. Unlike that kernel this one carries the
epilogue, so the row tile is held one step below its measured optimum (see
``_logits_config``) — the per-shape tuning is inherited, not re-measured, since
this machine has no Ascend device attached.

Soft-cap uses the sigmoid form of tanh (``cap * (2*sigmoid(2z/cap) - 1)``);
``tl.tanh`` is not available on this backend. Top-k breaks ties toward the
lowest expert index (``min`` over the argmax set) to match ``torch.topk``.
"""

import torch
import triton
import triton.language as tl

# Deepest loop-level software pipelining this backend's Triton accepts for
# ``tl.range(..., num_stages=N)``: it asserts ``num_stages <= 2`` on a range
# iterator, so anything deeper is a compile error, not a slow kernel.
_MAX_KLOOP_STAGES = 2

# Ascend unified-buffer budget in bytes (~192KB). The tile footprint picked by
# ``_logits_config`` is kept inside this; overflowing it fails to compile.
_UB_BYTES = 1572864 // 8


@triton.jit
def _logits_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    logits_ptr,
    BS: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    STRIDE_X_B: tl.constexpr,
    STRIDE_W_E: tl.constexpr,
    CAP: tl.constexpr,
    HAS_CAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_IEEE: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KLOOP_STAGES: tl.constexpr,
):
    """``logits[m, e] = x[m, :] @ w[e, :].T`` (+ soft-cap, + bias).

    Both inputs are contiguous (the launcher enforces it), so the hidden-dim
    stride is 1 and only the row strides are passed.
    """
    pid_m = tl.program_id(0)
    pid_e = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    offs_h0 = tl.arange(0, BLOCK_H)
    m_live = offs_m < BS
    e_live = offs_e < E

    acc = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)
    for h0 in tl.range(0, H, BLOCK_H, num_stages=KLOOP_STAGES):
        offs_h = h0 + offs_h0
        if EVEN:
            # Every tile is fully in range: no mask arithmetic at all.
            x_tile = tl.load(
                x_ptr + offs_m[:, None] * STRIDE_X_B + offs_h[None, :],
            ).to(tl.float32)
            w_tile = tl.load(
                w_ptr + offs_e[:, None] * STRIDE_W_E + offs_h[None, :],
            ).to(tl.float32)
        else:
            h_live = offs_h < H
            x_tile = tl.load(
                x_ptr + offs_m[:, None] * STRIDE_X_B + offs_h[None, :],
                mask=m_live[:, None] & h_live[None, :],
                other=0.0,
            ).to(tl.float32)
            w_tile = tl.load(
                w_ptr + offs_e[:, None] * STRIDE_W_E + offs_h[None, :],
                mask=e_live[:, None] & h_live[None, :],
                other=0.0,
            ).to(tl.float32)
        # Native bf16/f16 tiles fault the Ascend MMA; the tiles are fp32 above
        # and the precision flag picks the schedule.
        if USE_IEEE:
            acc += tl.dot(x_tile, tl.trans(w_tile), input_precision="ieee")
        else:
            acc += tl.dot(x_tile, tl.trans(w_tile), input_precision="hf32")

    if HAS_CAP:
        # tanh(z/cap) * cap, written with sigmoid (no tl.tanh on this backend).
        acc = (2.0 * tl.sigmoid(2.0 * (acc / CAP)) - 1.0) * CAP
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=e_live, other=0.0).to(
            tl.float32
        )
        acc += bias[None, :]

    tl.store(
        logits_ptr + offs_m[:, None] * E + offs_e[None, :],
        acc,
        mask=m_live[:, None] & e_live[None, :],
    )


@triton.jit
def _topk_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NEG: tl.constexpr,
):
    token = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    e_live = offs_e < E
    logits = tl.load(logits_ptr + token * E + offs_e, mask=e_live, other=NEG)
    logits = tl.where(e_live, logits, NEG)
    row_max = tl.max(logits, axis=0)
    inv = 1.0 / tl.sum(tl.exp(logits - row_max), axis=0)
    remaining = logits
    for slot in range(0, TOPK):
        selected = tl.max(remaining, axis=0)
        expert = tl.min(
            tl.where(remaining == selected, offs_e, BLOCK_E), axis=0
        )
        tl.store(ids_ptr + token * TOPK + slot, expert)
        tl.store(
            weights_ptr + token * TOPK + slot, tl.exp(selected - row_max) * inv
        )
        remaining = tl.where(offs_e == expert, NEG, remaining)


def _next_pow2(value):
    result = 1
    while result < value:
        result *= 2
    return result


def _logits_config(bs, experts, hidden):
    """Pick (BLOCK_M, BLOCK_E, BLOCK_H, num_warps, num_stages, kloop_stages).

    Pure function of the shapes. The expert axis is kept in a single tile
    whenever it fits (``BLOCK_E = next_pow2(E) <= 256``) so the grid stays
    small; the row tile is the lever.

      * ``bs > 512`` (compute-bound): an N-split ``[128, 128]`` tile with
        ``BLOCK_H = 128``. The sibling tensorcore GEMM measured ``[256, 128]``
        as the sweet spot (~199us) with ``[128, 256]`` next (~206us), but that
        kernel has no epilogue — here the soft-cap's fp32 temporaries are
        another couple of ``[BLOCK_M, BLOCK_E]`` tiles sharing the same budget,
        so the row tile is held at 128. Splitting the expert axis keeps the
        program count up either way.
      * ``bs <= 512`` (launch-bound): one full-axis ``[BLOCK_M, 256]`` tile,
        ``BLOCK_M`` scaled to the actual rows (padded rows are cheap here, an
        extra program launch is not).

    Whatever the preference above picks, ``BLOCK_H`` is then shrunk until the
    tile footprint fits ``_UB_BYTES``: the accumulator, the epilogue's fp32
    temporaries and the staged ``x`` / ``w`` operand tiles all share the
    unified buffer, and overflowing it is a compile failure on this backend,
    not a slow kernel. That fit check is what folds in the donor tuning
    table's ``BLOCK_K = 1024 / 512`` entries — they came from a device with a
    far larger tile budget.
    """
    block_e = _next_pow2(max(experts, 16))
    if block_e > 256:
        # Very large E: fall back to a classic 128-wide expert split.
        block_e = 128
    block_h = _next_pow2(max(hidden, 16))
    if block_h > 256:
        block_h = 256

    if bs > 512:
        block_m = 128
        block_e = min(block_e, 128)
        block_h = min(block_h, 128)
    elif bs <= 64:
        block_m = 64 if bs <= 16 else 32
    else:
        block_m = 64

    if hidden >= 256 and block_h >= 64:
        num_warps, num_stages = 8, 3
    else:
        num_warps, num_stages = 4, 2

    # K-loop pipelining overlaps the next K-tile's loads with the dot. The
    # backend caps loop-level staging at _MAX_KLOOP_STAGES.
    kloop_stages = _MAX_KLOOP_STAGES

    # Shrink BLOCK_H (then, if still needed, the kernel-wide staging) until the
    # tile footprint fits the unified buffer: the [BLOCK_M, BLOCK_E] fp32
    # accumulator, the soft-cap epilogue's temporaries over that same tile, and
    # ``num_stages`` copies of the two [.., BLOCK_H] operand tiles.
    tile_bytes = block_m * block_e * 4 * 2
    while (
        tile_bytes + num_stages * (block_m * block_h + block_e * block_h) * 4
        > _UB_BYTES
    ):
        if block_h > 16:
            block_h //= 2
        elif num_stages > 1:
            num_stages -= 1
        else:
            break
    kloop_stages = min(kloop_stages, num_stages)
    return block_m, block_e, block_h, num_warps, num_stages, kloop_stages


def fused_moe_router_cudacore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    x = x.contiguous()
    weights = router_weight.contiguous()
    bs, hidden = x.shape
    experts = weights.shape[0]
    topk = int(topk)
    cap = float(moe_softcapping)
    has_cap = cap != 0.0
    has_bias = correction_bias is not None
    logits = torch.empty((bs, experts), dtype=torch.float32, device=x.device)
    topk_weights = torch.empty(
        (bs, topk), dtype=torch.float32, device=x.device
    )
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
    # bias_ptr is unread when HAS_BIAS is false; any valid pointer will do.
    bias_arg = correction_bias if has_bias else x
    neg = -3.0e38

    block_m, block_e, block_h, num_warps, num_stages, kloop_stages = (
        _logits_config(bs, experts, hidden)
    )
    even = (
        bs % block_m == 0 and experts % block_e == 0 and hidden % block_h == 0
    )
    grid = (
        (bs + block_m - 1) // block_m,
        (experts + block_e - 1) // block_e,
    )
    _logits_kernel[grid](
        x,
        weights,
        bias_arg,
        logits,
        bs,
        hidden,
        experts,
        x.stride(0),
        weights.stride(0),
        cap,
        has_cap,
        has_bias,
        x.dtype == torch.float32,
        even,
        block_m,
        block_e,
        block_h,
        kloop_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _topk_kernel[(bs,)](
        logits,
        topk_weights,
        topk_ids,
        experts,
        topk,
        _next_pow2(experts),
        neg,
        num_warps=1,
        num_stages=1,
    )
    return topk_weights, topk_ids


__all__ = ["fused_moe_router_cudacore"]
