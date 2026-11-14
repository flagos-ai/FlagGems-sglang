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

"""merge_state — online-softmax merge of two split-KV attention states.

GCU (Enflame L600, 48 SP) tuning notes:
* The op is memory-bound (2 bf16 tile loads + 1 store + tiny fp32 lse I/O).
  A flat-1D rewrite is not viable on this backend: every mechanism that
  broadcasts the per-row scales onto a flat vector (value reshape, address
  reshape, broadcast_to+reshape) lowers to a slow layout conversion
  (4-6x), and per-lane gather scales need per-lane transcendentals.
  The 2D [RPP, HEAD_DIM] tile keeps the broadcast native and is the only
  fast form.
* The lse loads ([RPP] fp32 vectors) sit on the FMA dependency chain and
  their latency is not hidden by the tile streams, costing ~30us on the
  4096-token shape. The pipelined arm prefetches chunk k+1's lse and tiles
  before consuming chunk k, which recovers most of that cost.
* Keeping the tile FMA in the input dtype (scales cast to the tile dtype)
  halves buffer DSM so the double-buffered pipeline stays under the 448KB
  dual-program threshold; fp32 tiles push it over and lock single-program
  scheduling (~1.6x slower). fp32 inputs are unaffected (cast is a no-op).
* Plain arm tiers (any shape, masked): rpp 32/64/256 by row_count, nw=1,
  with the [rpp, BLOCK] tile capped at 32768 lanes (large head dims shrink
  rpp to stay inside the DSM budget).
* Pipe arm (pow2 head_dim <= 128, row_count % (256*K) == 0, >= 16384 rows):
  rpp=256, K=2 up to 65536 rows then K=4, nw=1.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _merge_state_kernel(
    prefix_output,
    prefix_lse,
    suffix_output,
    suffix_lse,
    output,
    output_lse,
    row_count,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    columns = tl.arange(0, BLOCK_SIZE)
    row_mask = rows < row_count

    prefix_lse_value = tl.load(
        prefix_lse + rows, mask=row_mask, other=-math.inf
    ).to(tl.float32)
    suffix_lse_value = tl.load(
        suffix_lse + rows, mask=row_mask, other=-math.inf
    ).to(tl.float32)
    prefix_lse_value = tl.where(
        prefix_lse_value == math.inf, -math.inf, prefix_lse_value
    )
    suffix_lse_value = tl.where(
        suffix_lse_value == math.inf, -math.inf, suffix_lse_value
    )

    prefix_is_high = prefix_lse_value >= suffix_lse_value
    high_lse = tl.maximum(prefix_lse_value, suffix_lse_value)
    delta = tl.minimum(prefix_lse_value, suffix_lse_value) - high_lse
    low_se = tl.exp(delta)
    inverse_sum = 1.0 / (1.0 + low_se)
    low_scale = low_se * inverse_sum
    log_sum = tl.log(1.0 + low_se)
    prefix_scale = tl.where(prefix_is_high, inverse_sum, low_scale)
    suffix_scale = tl.where(prefix_is_high, low_scale, inverse_sum)
    tl.store(output_lse + rows, high_lse + log_sum, mask=row_mask)

    offsets = rows[:, None] * HEAD_DIM + columns[None, :]
    mask = row_mask[:, None] & (columns[None, :] < HEAD_DIM)
    prefix_value = tl.load(prefix_output + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    suffix_value = tl.load(suffix_output + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    merged = (
        prefix_value * prefix_scale[:, None]
        + suffix_value * suffix_scale[:, None]
    )
    tl.store(output + offsets, merged, mask=mask)


@triton.jit
def _merge_state_pipe_kernel(
    prefix_output,
    prefix_lse,
    suffix_output,
    suffix_lse,
    output,
    output_lse,
    row_count,
    HEAD_DIM: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    K: tl.constexpr,
):
    columns = tl.arange(0, HEAD_DIM)
    base_row = tl.program_id(0) * (ROWS_PER_PROGRAM * K)

    rows0 = base_row + tl.arange(0, ROWS_PER_PROGRAM)
    prefix_lse_cur = tl.load(prefix_lse + rows0).to(tl.float32)
    suffix_lse_cur = tl.load(suffix_lse + rows0).to(tl.float32)
    offsets0 = rows0[:, None] * HEAD_DIM + columns[None, :]
    prefix_cur = tl.load(prefix_output + offsets0)
    suffix_cur = tl.load(suffix_output + offsets0)

    for k in tl.static_range(K):
        if k + 1 < K:
            rows_next = (
                base_row
                + (k + 1) * ROWS_PER_PROGRAM
                + tl.arange(0, ROWS_PER_PROGRAM)
            )
            prefix_lse_next = tl.load(prefix_lse + rows_next).to(tl.float32)
            suffix_lse_next = tl.load(suffix_lse + rows_next).to(tl.float32)
            offsets_next = rows_next[:, None] * HEAD_DIM + columns[None, :]
            prefix_next = tl.load(prefix_output + offsets_next)
            suffix_next = tl.load(suffix_output + offsets_next)

        prefix_lse_value = tl.where(
            prefix_lse_cur == math.inf, -math.inf, prefix_lse_cur
        )
        suffix_lse_value = tl.where(
            suffix_lse_cur == math.inf, -math.inf, suffix_lse_cur
        )
        prefix_is_high = prefix_lse_value >= suffix_lse_value
        high_lse = tl.maximum(prefix_lse_value, suffix_lse_value)
        delta = tl.minimum(prefix_lse_value, suffix_lse_value) - high_lse
        low_se = tl.exp(delta)
        inverse_sum = 1.0 / (1.0 + low_se)
        low_scale = low_se * inverse_sum
        log_sum = tl.log(1.0 + low_se)
        prefix_scale = tl.where(prefix_is_high, inverse_sum, low_scale)
        suffix_scale = tl.where(prefix_is_high, low_scale, inverse_sum)

        rows_cur = (
            base_row + k * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
        )
        tl.store(output_lse + rows_cur, high_lse + log_sum)
        merged = (
            prefix_cur * prefix_scale.to(prefix_cur.dtype)[:, None]
            + suffix_cur * suffix_scale.to(suffix_cur.dtype)[:, None]
        )
        offsets_cur = rows_cur[:, None] * HEAD_DIM + columns[None, :]
        tl.store(output + offsets_cur, merged)

        if k + 1 < K:
            prefix_lse_cur = prefix_lse_next
            suffix_lse_cur = suffix_lse_next
            prefix_cur = prefix_next
            suffix_cur = suffix_next


def merge_state(prefix_output, prefix_lse, suffix_output, suffix_lse):
    output_elements = prefix_output.numel()
    lse_storage_elements = (
        prefix_lse.numel()
        * prefix_lse.element_size()
        // prefix_output.element_size()
    )
    storage = torch.empty(
        output_elements + lse_storage_elements,
        device=prefix_output.device,
        dtype=prefix_output.dtype,
    )
    output = storage[:output_elements].view_as(prefix_output)
    output_lse = (
        storage[output_elements:].view(prefix_lse.dtype).view_as(prefix_lse)
    )

    head_dim = prefix_output.shape[-1]
    row_count = output_elements // head_dim
    head_dim_is_pow2 = head_dim > 0 and (head_dim & (head_dim - 1)) == 0

    pipelined = False
    if head_dim_is_pow2 and row_count >= 16384 and 256 * head_dim <= 32768:
        chunks = 4 if row_count > 65536 else 2
        if row_count % (256 * chunks) == 0:
            _merge_state_pipe_kernel[(row_count // (256 * chunks),)](
                prefix_output,
                prefix_lse,
                suffix_output,
                suffix_lse,
                output,
                output_lse,
                row_count,
                HEAD_DIM=head_dim,
                ROWS_PER_PROGRAM=256,
                K=chunks,
                num_warps=1,
            )
            pipelined = True

    if not pipelined:
        block_size = triton.next_power_of_2(head_dim)
        if row_count <= 512:
            rows_per_program = 32
        elif row_count <= 4096:
            rows_per_program = 64
        else:
            rows_per_program = 256
        while rows_per_program > 1 and rows_per_program * block_size > 32768:
            rows_per_program //= 2
        _merge_state_kernel[(triton.cdiv(row_count, rows_per_program),)](
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output,
            output_lse,
            row_count,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            ROWS_PER_PROGRAM=rows_per_program,
            num_warps=1,
        )
    return output, output_lse


__all__ = ["merge_state"]
