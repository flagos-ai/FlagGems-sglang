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

import torch
import triton
import triton.language as tl


@triton.jit
def _sgemm_lora_b_kernel(
    x,
    weights,
    seg_indptr,
    weight_indices,
    lora_ranks,
    scalings,
    permutation,
    base_output,
    output,
    rank: tl.constexpr,
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_weight_adapter: tl.constexpr,
    stride_weight_output: tl.constexpr,
    stride_base_row: tl.constexpr,
    has_permutation: tl.constexpr,
    use_dot: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    bf16_epi: tl.constexpr,
    even_k: tl.constexpr,
    even_n: tl.constexpr,
    hoist_base: tl.constexpr,
):
    segment_tile = tl.program_id(0)
    output_tile = tl.program_id(1)
    tiles_per_segment: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment

    start = tl.load(seg_indptr + segment)
    end = tl.load(seg_indptr + segment + 1)
    adapter = tl.load(weight_indices + segment)
    active = tl.load(lora_ranks + adapter) != 0
    scaling = tl.load(scalings + adapter)

    offsets_m = tile_m * block_m + tl.arange(0, block_m)
    offsets_n = output_tile * block_n + tl.arange(0, block_n)
    sequence_positions = start + offsets_m
    mask_m = sequence_positions < end
    compute_m = mask_m & active
    mask_n = offsets_n < output_size
    if has_permutation:
        rows = tl.load(permutation + sequence_positions, mask=mask_m, other=0)
    else:
        rows = sequence_positions

    output_offsets = rows[:, None] * stride_base_row + offsets_n[None, :]
    if even_n:
        output_mask = mask_m[:, None]
    else:
        output_mask = mask_m[:, None] & mask_n[None, :]
    if hoist_base:
        if bf16_epi:
            base = tl.load(
                base_output + output_offsets,
                mask=output_mask,
                other=0.0,
            )
        else:
            base_fp32 = tl.load(
                base_output + output_offsets,
                mask=output_mask,
                other=0.0,
            ).to(tl.float32)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    if use_dot:
        for offset_k in range(0, rank, block_k):
            offsets_k = offset_k + tl.arange(0, block_k)
            if even_k:
                x_tile = tl.load(
                    x + rows[:, None] * stride_x_row + offsets_k[None, :],
                    mask=compute_m[:, None],
                    other=0.0,
                )
            else:
                mask_k = offsets_k < rank
                x_tile = tl.load(
                    x + rows[:, None] * stride_x_row + offsets_k[None, :],
                    mask=compute_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
            if even_k and even_n:
                weight_tile = tl.load(
                    weights
                    + adapter * stride_weight_adapter
                    + offsets_k[:, None]
                    + offsets_n[None, :] * stride_weight_output
                )
            elif even_k:
                weight_tile = tl.load(
                    weights
                    + adapter * stride_weight_adapter
                    + offsets_k[:, None]
                    + offsets_n[None, :] * stride_weight_output,
                    mask=mask_n[None, :],
                    other=0.0,
                )
            else:
                mask_k = offsets_k < rank
                weight_tile = tl.load(
                    weights
                    + adapter * stride_weight_adapter
                    + offsets_k[:, None]
                    + offsets_n[None, :] * stride_weight_output,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                )
            accumulator += tl.dot(x_tile, weight_tile)
    else:
        for offset_k in range(0, rank):
            x_values = tl.load(
                x + rows * stride_x_row + offset_k,
                mask=compute_m,
                other=0.0,
            ).to(tl.float32)
            weight_values = tl.load(
                weights
                + adapter * stride_weight_adapter
                + offsets_n * stride_weight_output
                + offset_k,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            accumulator += x_values[:, None] * weight_values[None, :]

    if not hoist_base:
        if bf16_epi:
            base = tl.load(
                base_output + output_offsets,
                mask=output_mask,
                other=0.0,
            )
        else:
            base_fp32 = tl.load(
                base_output + output_offsets,
                mask=output_mask,
                other=0.0,
            ).to(tl.float32)
    if bf16_epi:
        tl.store(
            output + output_offsets,
            base + (scaling * accumulator).to(base.dtype),
            mask=output_mask,
        )
    else:
        tl.store(
            output + output_offsets,
            base_fp32 + accumulator * scaling,
            mask=output_mask,
        )


@triton.jit
def _serial_kernel(
    x,
    weights,
    seg_indptr,
    weight_indices,
    lora_ranks,
    scalings,
    base_output,
    output,
    rank: tl.constexpr,
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_weight_adapter: tl.constexpr,
    stride_weight_output: tl.constexpr,
    stride_base_row: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    segment_tile = tl.program_id(0)
    output_tile = tl.program_id(1)
    tiles_per_segment: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment
    start = tl.load(seg_indptr + segment)
    end = tl.load(seg_indptr + segment + 1)
    adapter = tl.load(weight_indices + segment)
    active = tl.load(lora_ranks + adapter) != 0
    scaling = tl.load(scalings + adapter)
    offsets_m = tile_m * block_m + tl.arange(0, block_m)
    offsets_n = output_tile * block_n + tl.arange(0, block_n)
    rows = start + offsets_m
    mask_m = rows < end
    compute_m = mask_m & active
    mask_n = offsets_n < output_size
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for offset_k in tl.static_range(0, rank):
        x_values = tl.load(
            x + rows * stride_x_row + offset_k,
            mask=compute_m,
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(
            weights
            + adapter * stride_weight_adapter
            + offsets_n * stride_weight_output
            + offset_k,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        accumulator += x_values[:, None] * weight_values[None, :]
    output_offsets = rows[:, None] * stride_base_row + offsets_n[None, :]
    base = tl.load(
        base_output + output_offsets,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + output_offsets,
        base + accumulator * scaling,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _serial_permuted_kernel(
    x,
    weights,
    seg_indptr,
    weight_indices,
    lora_ranks,
    scalings,
    permutation,
    base_output,
    output,
    rank: tl.constexpr,
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_weight_adapter: tl.constexpr,
    stride_weight_output: tl.constexpr,
    stride_base_row: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    segment_tile = tl.program_id(0)
    output_tile = tl.program_id(1)
    tiles_per_segment: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment
    start = tl.load(seg_indptr + segment)
    end = tl.load(seg_indptr + segment + 1)
    adapter = tl.load(weight_indices + segment)
    active = tl.load(lora_ranks + adapter) != 0
    scaling = tl.load(scalings + adapter)
    offsets_m = tile_m * block_m + tl.arange(0, block_m)
    offsets_n = output_tile * block_n + tl.arange(0, block_n)
    sequence_positions = start + offsets_m
    mask_m = sequence_positions < end
    compute_m = mask_m & active
    mask_n = offsets_n < output_size
    rows = tl.load(permutation + sequence_positions, mask=mask_m, other=0)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for offset_k in tl.static_range(0, rank):
        x_values = tl.load(
            x + rows * stride_x_row + offset_k,
            mask=compute_m,
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(
            weights
            + adapter * stride_weight_adapter
            + offsets_n * stride_weight_output
            + offset_k,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        accumulator += x_values[:, None] * weight_values[None, :]
    output_offsets = rows[:, None] * stride_base_row + offsets_n[None, :]
    base = tl.load(
        base_output + output_offsets,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + output_offsets,
        base + accumulator * scaling,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _launch_config(
    x,
    weights,
    batch_info,
    base_output,
    block_m,
    block_n,
    block_k,
    num_warps,
    num_stages,
    bf16_epi=False,
):
    _, rank = x.shape
    output_size = weights.shape[1]
    batch_size = batch_info.bs
    max_segment_length = max(1, batch_info.max_len)
    output = torch.empty_like(base_output)
    grid = (
        batch_size * triton.cdiv(max_segment_length, block_m),
        triton.cdiv(output_size, block_n),
    )
    permutation = batch_info.permutation
    _sgemm_lora_b_kernel[grid](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        x if permutation is None else permutation,
        base_output,
        output,
        rank,
        output_size,
        max_segment_length,
        x.stride(0),
        weights.stride(0),
        weights.stride(1),
        base_output.stride(0),
        permutation is not None,
        x.dtype != torch.float32,
        block_m,
        block_n,
        block_k,
        bf16_epi=bf16_epi,
        even_k=int(rank) == int(block_k),
        even_n=output_size % block_n == 0,
        hoist_base=(
            block_m
            * block_n
            * (8 if num_warps == 1 else 4)
            + 2 * block_k * (block_m + block_n)
            + block_m * block_n * 4
            <= _DSM_DUAL_PROGRAM
        ),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def _launch_serial(x, weights, batch_info, base_output, block_m, block_n):
    _, rank = x.shape
    output_size = weights.shape[1]
    max_segment_length = max(1, batch_info.max_len)
    output = torch.empty_like(base_output)
    grid = (
        batch_info.bs * triton.cdiv(max_segment_length, block_m),
        triton.cdiv(output_size, block_n),
    )
    _serial_kernel[grid](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        base_output,
        output,
        rank,
        output_size,
        max_segment_length,
        x.stride(0),
        weights.stride(0),
        weights.stride(1),
        base_output.stride(0),
        block_m,
        block_n,
        num_warps=4,
        num_stages=1,
    )
    return output


def _launch_serial_permuted(x, weights, batch_info, base_output):
    _, rank = x.shape
    output_size = weights.shape[1]
    max_segment_length = max(1, batch_info.max_len)
    block_m = 16
    block_n = 64
    output = torch.empty_like(base_output)
    grid = (
        batch_info.bs * triton.cdiv(max_segment_length, block_m),
        triton.cdiv(output_size, block_n),
    )
    _serial_permuted_kernel[grid](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        batch_info.permutation,
        base_output,
        output,
        rank,
        output_size,
        max_segment_length,
        x.stride(0),
        weights.stride(0),
        weights.stride(1),
        base_output.stride(0),
        block_m,
        block_n,
        num_warps=4,
        num_stages=1,
    )
    return output


# gcu400 on-chip DSM per program (bytes). The kernel per-program DSM,
# measured on gcu400 across tile configs (probe on this op, 2026-11-07),
# follows the same model as the sibling gate_up_lora_b fast kernel:
#   block_m*block_n*8 + 2*block_k*(block_m + block_n)   at num_warps=1
#   block_m*block_n*4 + 2*block_k*(block_m + block_n)   at num_warps=4
# (nw4 on <=32768-element tiles measured even lower, x2) and only holds
# while block_m*block_n <= 65536: larger tiles fall back to full fp32
# materialization (the original 256x512 nw1 config requested 917632 B >
# 917504 B and failed to compile with OutOfResources).
# Hoisting the epilogue base tile ahead of the dot (hoist_base) keeps it
# live through the k loop and the compiler stages it at fp32 width, i.e.
# an extra block_m*block_n*4 bytes: 64x512 -> 299008 B (still 2-program
# co-resident, -2.7% measured), 256x256 -> 589824 B > _DSM_DUAL_PROGRAM
# (locks 24-way, +6% measured), so the host enables the hoist only under
# the dual-program budget (measured llir dsm matches both predictions).
_DSM_LIMIT = 917504
_DSM_DUAL_PROGRAM = _DSM_LIMIT // 2
# Planning keeps a safety margin below the compile hard limit: the
# formula matched llir exactly on the tuned ranks, but tile picks that
# land exactly on _DSM_LIMIT (e.g. block_k=1024 at 256x128) are left to
# the next ladder entry instead of gambling on the compile-time check.
_DSM_PLAN_LIMIT = _DSM_LIMIT - 65536

# (block_m, block_n, num_warps, num_stages), fastest first per branch,
# measured on the two core benchmark shapes (bf16, fast path with
# bf16_epi/even_k/even_n/hoist_base):
#   seg=(64,)*8 r=32  -> 64x512 nw4 st2 17.1 us (64x1024 19.2-20.2, generic 37.8*)
#   seg=(256,)*4 r=64 -> 256x128 nw2 st2 24.6 us (256x256 nw4 25.6, 128x256 nw2
#                       25.0, 128x512 nw4 26.4-26.6, generic 39.5*)
# 2026-11-08 23-config sweep vetoed nw2/nw8/wider/narrower/st on the then
# current tiles; the 2026-11-13 second round re-swept the shape-1 tile
# neighborhood: 256x128/128x256 at nw2 (tile 32768) are the only points
# that beat the 256x256 nw4 incumbent (-3.2% interleaved A/B, st1/st2/st3
# equal); wider no-hoist tiles lose the hoist overlap, and other nw2
# tiles regress (256x256 nw2 29.7, 256x64 39.5, 128x128 38.4).
# * generic numbers are the 11-07 baseline; on the 11-08 stack the
#   generic kernel no longer fresh-compiles (memref_ext.slice_pad_start
#   8 vs 7), see compiler_issues/vllm_silu_flat_slice_start.md.
_SMALL_TILES = ((64, 512, 4, 2), (64, 256, 4, 2), (64, 1024, 1, 1))
_LARGE_TILES = ((256, 128, 2, 2), (256, 256, 4, 2), (128, 128, 4, 2))


def _launch_plan(max_len, rank):
    """Pick 4096-wide kernel tiles that fit the gcu400 DSM budget."""
    block_k = triton.next_power_of_2(max(int(rank), 1))
    candidates = _SMALL_TILES if max_len <= 64 else _LARGE_TILES
    for block_m, block_n, num_warps, num_stages in candidates:
        tile = block_m * block_n
        dsm = tile * (8 if num_warps == 1 else 4) + 2 * block_k * (block_m + block_n)
        if tile <= 65536 and dsm <= _DSM_PLAN_LIMIT:
            return block_m, block_n, num_warps, num_stages
    return None


def sgemm_lora_b(x, weights, batch_info, base_output):
    if weights.shape[1] == 4096 and x.element_size() == 2:
        plan = _launch_plan(batch_info.max_len, x.shape[1])
        if plan is not None:
            block_m, block_n, num_warps, num_stages = plan
            return _launch_config(
                x,
                weights,
                batch_info,
                base_output,
                block_m,
                block_n,
                triton.next_power_of_2(max(int(x.shape[1]), 1)),
                num_warps,
                num_stages,
                bf16_epi=base_output.element_size() == 2,
            )
    return _launch_config(
        x, weights, batch_info, base_output, 32, 64, 32, 4, 2
    )


__all__ = ["sgemm_lora_b"]
