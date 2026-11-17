# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

# One program per SP (GCU400 has 48 SPs). The GCU400 dispatch floor is
# ~52 ns/CTA, so the previous (num_tokens, cdiv(H, BLOCK_N)) launch spent
# 28672 * 52 ns ~= 1.5 ms in dispatch alone on (4096, 8, 7168) - 82% of the
# 1.82 ms baseline. A persistent grid capped at 48 programs removes that
# floor and lets each program stride over the flattened tile space.
_GRID_TARGET = 48
# Token-walking kernel only for launches with enough tokens to amortize
# the coarser per-program granularity; at T=64 the flat tile launch still
# measures marginally faster (9.49 vs 9.60 us), from T=96 up the walk form
# wins (-5.5% at T=96, -2% at T=512/4096).
_TOKEN_WALK_MIN_TOKENS = 96


@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    CONTIG_H: tl.constexpr,
):
    # 1-D affine accesses only (the (BLOCK_K, BLOCK_N) 2-D tile + tl.sum form
    # is both the known-slow tree path and a memref_ext compile-regression
    # risk). The top-k reduction is a constexpr static_range unroll: the k
    # loads issue in parallel and accumulate in ascending k order, which is
    # bitwise equal to the fp32 reference sum. EXACT drops the hidden
    # predicate when BLOCK_H divides hidden_size; CONTIG_H drops the
    # h * stride vector multiplies; SCALE is a per-value constexpr so the
    # runtime-float-scalar-in-loop tar_load pathology cannot trigger.
    NUM_H_BLOCKS: tl.constexpr = (hidden_size + BLOCK_H - 1) // BLOCK_H
    EXACT: tl.constexpr = hidden_size % BLOCK_H == 0
    total_tiles = NUM_H_BLOCKS * num_tokens
    for flat_idx in range(tl.program_id(0), total_tiles, tl.num_programs(0)):
        block_idx = flat_idx % NUM_H_BLOCKS
        token_idx = flat_idx // NUM_H_BLOCKS
        hidden_offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
        hidden_mask = hidden_offsets < hidden_size
        if CONTIG_H:
            input_base = (
                input_ptr + token_idx * input_stride_token + hidden_offsets
            )
            output_pos = (
                output_ptr + token_idx * output_stride_token + hidden_offsets
            )
        else:
            input_base = (
                input_ptr
                + token_idx * input_stride_token
                + hidden_offsets * input_stride_hidden
            )
            output_pos = (
                output_ptr
                + token_idx * output_stride_token
                + hidden_offsets * output_stride_hidden
            )
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for k in tl.static_range(top_k):
            if EXACT:
                values = tl.load(input_base + k * input_stride_topk)
            else:
                values = tl.load(
                    input_base + k * input_stride_topk,
                    mask=hidden_mask,
                    other=0.0,
                )
            acc += values.to(tl.float32)
        acc = acc * SCALE
        if EXACT:
            tl.store(output_pos, acc.to(output_ptr.dtype.element_ty))
        else:
            tl.store(
                output_pos,
                acc.to(output_ptr.dtype.element_ty),
                mask=hidden_mask,
            )


@triton.jit
def _moe_sum_reduce_walk_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    CONTIG_H: tl.constexpr,
    I64_IDX: tl.constexpr,
):
    # Same arithmetic and k-ascending order as _moe_sum_reduce_kernel, but
    # each program strides over tokens and walks the hidden space as
    # NUM_FULL whole blocks plus one ragged tail block, all unrolled
    # statically inside the token body. Within a token the block sequence
    # is a compile-time fact, so the whole blocks carry no hidden predicate
    # at all and no runtime full/tail branch is needed: a per-tile runtime
    # branch measurably breaks the stride-loop software pipeline (+13% on
    # (4096, 8, 7168)), while the uniform masked form pays a ~10% predicate
    # tax (exact-vs-masked same-shape A/B: 2.18 vs 1.97 TB/s). Per-tile
    # arithmetic, accumulation order and addresses are unchanged, so every
    # element is bitwise equal to the single-loop kernel.
    NUM_FULL: tl.constexpr = hidden_size // BLOCK_H
    HAS_TAIL: tl.constexpr = hidden_size % BLOCK_H != 0
    nprog = tl.num_programs(0)
    for token_idx in range(tl.program_id(0), num_tokens, nprog):
        if I64_IDX:
            token_idx2 = token_idx.to(tl.int64)
        else:
            token_idx2 = token_idx
        for b in tl.static_range(NUM_FULL):
            hidden_offsets = b * BLOCK_H + tl.arange(0, BLOCK_H)
            if CONTIG_H:
                input_base = (
                    input_ptr
                    + token_idx2 * input_stride_token
                    + hidden_offsets
                )
                output_pos = (
                    output_ptr
                    + token_idx2 * output_stride_token
                    + hidden_offsets
                )
            else:
                input_base = (
                    input_ptr
                    + token_idx2 * input_stride_token
                    + hidden_offsets * input_stride_hidden
                )
                output_pos = (
                    output_ptr
                    + token_idx2 * output_stride_token
                    + hidden_offsets * output_stride_hidden
                )
            acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for k in tl.static_range(top_k):
                values = tl.load(input_base + k * input_stride_topk)
                acc += values.to(tl.float32)
            acc = acc * SCALE
            tl.store(output_pos, acc.to(output_ptr.dtype.element_ty))
        if HAS_TAIL:
            # Ragged tail block (the remainder may be smaller than BLOCK_H):
            # the hidden predicate lives here only.
            hidden_offsets = NUM_FULL * BLOCK_H + tl.arange(0, BLOCK_H)
            hidden_mask = hidden_offsets < hidden_size
            if CONTIG_H:
                input_base = (
                    input_ptr
                    + token_idx2 * input_stride_token
                    + hidden_offsets
                )
                output_pos = (
                    output_ptr
                    + token_idx2 * output_stride_token
                    + hidden_offsets
                )
            else:
                input_base = (
                    input_ptr
                    + token_idx2 * input_stride_token
                    + hidden_offsets * input_stride_hidden
                )
                output_pos = (
                    output_ptr
                    + token_idx2 * output_stride_token
                    + hidden_offsets * output_stride_hidden
                )
            acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for k in tl.static_range(top_k):
                values = tl.load(
                    input_base + k * input_stride_topk,
                    mask=hidden_mask,
                    other=0.0,
                )
                acc += values.to(tl.float32)
            acc = acc * SCALE
            tl.store(
                output_pos,
                acc.to(output_ptr.dtype.element_ty),
                mask=hidden_mask,
            )


def _max_linear_offset(tensor) -> int:
    offset = 0
    for dim, stride in zip(tensor.shape, tensor.stride()):
        offset += (dim - 1) * stride
    return offset


def moe_sum_reduce(
    input: torch.Tensor, routed_scaling_factor: float
) -> torch.Tensor:
    num_tokens, top_k, hidden_dim = input.shape
    output = torch.empty(
        (num_tokens, hidden_dim), dtype=input.dtype, device=input.device
    )
    if num_tokens == 0 or hidden_dim == 0:
        return output
    # BLOCK_H sweep on (T, 8, 7168) bf16: 1024/2048/4096/8192 @ nw=4 =>
    # 545/282/268/269 us @ T=4096 and 73/40/39/40 us @ T=512 - load width
    # dominates over predicate elimination (12.5% masked waste is cheaper
    # than halving the DMA width), so 4096 is the cap once the grid is
    # saturated. For num_tokens < 64 the launch has fewer tiles than SPs,
    # and the extra SP-level parallelism of 2048-wide blocks wins
    # (T=1: 4.08 vs 4.48 us, T=8: 4.63 vs 4.79 us). nw=4 keeps every warp
    # at <=1024 elements (compile boundary: >4096 bf16 elems/warp hangs).
    width_cap = 2048 if num_tokens < 64 else 4096
    block_h = min(width_cap, triton.next_power_of_2(hidden_dim))
    num_warps = min(4, max(1, block_h // 128))
    contiguous = input.stride(2) == 1 and output.stride(1) == 1
    num_h_blocks = triton.cdiv(hidden_dim, block_h)
    total_tiles = num_h_blocks * num_tokens
    grid = (min(total_tiles, _GRID_TARGET),)
    if num_tokens >= _TOKEN_WALK_MIN_TOKENS:
        # Token-major addressing overflows int32 once num_tokens * stride
        # passes 2**31 (e.g. T > ~37k at (8, 7168) layout); widen the token
        # index only on those (rare) launches so hot shapes keep i32 math.
        i64_idx = (
            _max_linear_offset(input) >= 2**31
            or _max_linear_offset(output) >= 2**31
        )
        _moe_sum_reduce_walk_kernel[grid](
            input,
            output,
            num_tokens,
            hidden_dim,
            top_k,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            output.stride(0),
            output.stride(1),
            SCALE=float(routed_scaling_factor),
            BLOCK_H=block_h,
            CONTIG_H=contiguous,
            I64_IDX=i64_idx,
            num_warps=num_warps,
        )
    else:
        _moe_sum_reduce_kernel[grid](
            input,
            output,
            num_tokens,
            hidden_dim,
            top_k,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            output.stride(0),
            output.stride(1),
            SCALE=float(routed_scaling_factor),
            BLOCK_H=block_h,
            CONTIG_H=contiguous,
            num_warps=num_warps,
        )
    return output


__all__ = ["moe_sum_reduce"]
