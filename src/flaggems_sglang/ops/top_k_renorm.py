# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""Triton implementation of sampling/top_k_renorm.

Keeps every probability at or above the row's k-th largest value, zeroes the
rest, and renormalizes the retained mass in FP32. Ties at the cutoff are
retained, so a row can keep more than ``k`` entries -- which is exactly what
sglang's ``top_k_renorm_probs`` does (it masks with ``probs >= pivot``, not
with a rank).

The cutoff is found without sorting. Every probability is non-negative, so
interpreting the float32 bit pattern as a signed integer preserves order over
the whole domain this operator needs; a bisection on that integer then finds
the largest pivot whose retained count still reaches ``k`` in O(log(range))
sweeps instead of the O(vocab log vocab) a full sort would cost. This is the
same approach ``top_p_renorm`` takes for its threshold, kept consistent with
it rather than pulling a second sorting strategy into the same family.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _top_k_select(
    P,
    T,
    O,
    V: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    TS: tl.constexpr,
    IS_TENSOR: tl.constexpr,
    SINGLE: tl.constexpr,
    PV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)

    if IS_TENSOR:
        ti = 0 if SINGLE else row * TS
        k = tl.load(T + ti).to(tl.int32)
    else:
        k = tl.full((), PV, tl.int32)

    # Pass 1: the range of the row, as ordered integer bit patterns. The
    # sentinels are the widest possible int32 values, so any real probability
    # replaces them; there is always at least one valid column.
    lo_bits = tl.full((), 2147483647, tl.int32)
    hi_bits = tl.full((), 0, tl.int32)
    for start in range(triton.cdiv(V, BLOCK)):
        cols = start * BLOCK + offsets
        valid = cols < V
        x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(tl.float32)
        b = x.to(tl.int32, bitcast=True)
        lo_bits = tl.minimum(
            lo_bits, tl.min(tl.where(valid, b, 2147483647), 0)
        )
        hi_bits = tl.maximum(hi_bits, tl.max(tl.where(valid, b, 0), 0))

    # Pass 2: bisect for the largest pivot that still retains at least k
    # entries. The midpoint is rounded up so the loop always makes progress.
    # A k below 1 drives the pivot to the row maximum and a k above the row
    # length leaves it at the row minimum, matching clamping k to
    # [1, vocab_size] on the other side without an explicit branch.
    lo = lo_bits
    hi = hi_bits
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2
        pivot = mid.to(tl.float32, bitcast=True)
        count = tl.full((), 0, tl.int32)
        for start in range(triton.cdiv(V, BLOCK)):
            cols = start * BLOCK + offsets
            valid = cols < V
            x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(
                tl.float32
            )
            count += tl.sum(tl.where(valid & (x >= pivot), 1, 0), 0)
        if count >= k:
            lo = mid
        else:
            hi = mid - 1
    pivot = lo.to(tl.float32, bitcast=True)

    # Pass 3: the retained mass, then the write. A row whose retained mass is
    # zero would divide zero by zero, so the denominator gets a floor: it can
    # only be hit when every retained value is also zero, keeping the quotient
    # at zero instead of nan.
    denominator = tl.full((), 0.0, tl.float32)
    for start in range(triton.cdiv(V, BLOCK)):
        cols = start * BLOCK + offsets
        valid = cols < V
        x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(tl.float32)
        denominator += tl.sum(tl.where(valid & (x >= pivot), x, 0.0), 0)
    safe = tl.maximum(denominator, 1.0e-38)

    for start in range(triton.cdiv(V, BLOCK)):
        cols = start * BLOCK + offsets
        valid = cols < V
        x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(tl.float32)
        y = tl.where(x >= pivot, x / safe, 0.0)
        tl.store(O + row * V + cols, y, valid)


def top_k_renorm(probs, top_k):
    """Keep the top-k mass of each probability row and renormalize it.

    Args:
        probs: ``(batch, vocab)`` probability rows. Any floating dtype; the
            computation is done in FP32.
        top_k: either a Python integer shared by every row, or a tensor with
            one value per row (a one-element tensor is broadcast).

    Returns:
        A new FP32 ``(batch, vocab)`` tensor. Entries below each row's cutoff
        are exactly zero and the retained entries sum to 1, except for a row
        with no mass at all, which stays zero.
    """
    if probs.ndim != 2:
        raise ValueError(
            f"probs must be 2D (batch, vocab), got shape={tuple(probs.shape)}"
        )
    batch, vocab = probs.shape

    out = torch.empty((batch, vocab), dtype=torch.float32, device=probs.device)

    is_tensor = isinstance(top_k, torch.Tensor)
    if is_tensor:
        values = top_k.to(device=probs.device, dtype=torch.int32).reshape(-1)
        if values.numel() == 1:
            single = True
        elif values.numel() == batch:
            single = False
        else:
            raise ValueError(
                f"top_k must be a scalar or have one value per row, got "
                f"{values.numel()} values for {batch} rows"
            )
        stride = values.stride(0)
        scalar = 0
        k_arg = values
    else:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError(
                f"top_k must be an int or a tensor, got {top_k!r}"
            )
        values = probs
        single = True
        stride = 0
        scalar = top_k
        k_arg = probs

    if batch > 0 and vocab > 0:
        # One block covers as much of the vocabulary as fits, and the operand
        # loops over the rest; a whole-vocab in-register buffer is what makes
        # the bisection cheap enough to be worth doing on the device.
        block = min(32768, triton.next_power_of_2(vocab))
        _top_k_select[(batch,)](
            probs,
            k_arg,
            out,
            vocab,
            probs.stride(0),
            probs.stride(1),
            stride,
            is_tensor,
            single,
            scalar,
            block,
            num_warps=min(16, max(4, block // 1024)),
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["top_k_renorm"]
