# SPDX-License-Identifier: Apache-2.0
# Vendor-derived sections retain copyright of contributors to the vLLM project.
# Copyright 2026 FlagOS Contributors
"""qsa_compress_norm_mrope_store_groups; adapted from FlagGems-vllm PR #455 dependencies.

Source: flagos-ai/FlagGems-vllm@f775b37f3066e2935742d26c7592a5dd55499159.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _is_triton_device(*tensors: torch.Tensor) -> bool:
    """Return true for same-device accelerator tensors accepted by Triton."""

    return bool(
        tensors
        and len({tensor.device for tensor in tensors}) == 1
        and all(
            tensor.device.type not in ("cpu", "meta") for tensor in tensors
        )
    )


@triton.jit
def _compress_norm_mrope_store_qsa_groups_kernel(
    raw_cache_ptr,
    rope_cache_ptr,
    raw_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    norm_weight_ptr,
    cos_sin_cache_ptr,
    compressed_cache_ptr,
    stride_raw_block,
    stride_raw_token,
    stride_raw_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_table_req,
    stride_table_page,
    stride_cos_row,
    stride_cos_dim,
    stride_compressed_block,
    stride_compressed_token,
    stride_compressed_dim,
    num_rows,
    num_raw_blocks,
    num_rope_blocks,
    num_compressed_blocks,
    num_requests,
    num_cos_rows,
    norm_eps,
    RAW_PAGE_SIZE: tl.constexpr,
    RAW_TABLE_WIDTH: tl.constexpr,
    ROPE_PAGE_SIZE: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MROPE_SECTION_T: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
    MROPE_INTERLEAVED: tl.constexpr,
    LOAD_MROPE_POSITIONS: tl.constexpr,
) -> None:
    """Pool, Gemma-normalize, MRoPE, and store one compressed QSA key."""

    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_row = (
        (row < num_rows)
        & (request >= 0)
        & (request < num_requests)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
        & (compressed_slot < num_compressed_blocks * COMPRESSED_PAGE_SIZE)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if valid_row:
        for group_offset in tl.range(0, COMPRESS_RATIO):
            position = end_position - (COMPRESS_RATIO - 1 - group_offset)
            logical_page = position // RAW_PAGE_SIZE
            page_offset = position % RAW_PAGE_SIZE
            valid = logical_page < RAW_TABLE_WIDTH
            physical_page = tl.load(
                raw_table_ptr
                + request * stride_table_req
                + tl.minimum(logical_page, RAW_TABLE_WIDTH - 1)
                * stride_table_page,
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_raw_blocks)
            accumulator += tl.load(
                raw_cache_ptr
                + tl.maximum(physical_page, 0).to(tl.int64) * stride_raw_block
                + page_offset * stride_raw_token
                + dims * stride_raw_dim,
                mask=valid & (dims < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)

    # Match the unfused BF16 materialization before Gemma RMSNorm.
    pooled = (accumulator / COMPRESS_RATIO).to(tl.bfloat16)
    pooled_fp32 = pooled.to(tl.float32)
    variance = tl.sum(pooled_fp32 * pooled_fp32, axis=0) / HEAD_DIM
    weight = tl.load(
        norm_weight_ptr + dims,
        mask=dims < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    normalized = (
        pooled_fp32 * tl.rsqrt(variance + norm_eps) * (weight + 1.0)
    ).to(tl.bfloat16)

    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_MROPE_POSITIONS:
        rope_page = first_position // ROPE_PAGE_SIZE
        rope_offset = first_position % ROPE_PAGE_SIZE
        valid_rope = valid_row & (rope_page < RAW_TABLE_WIDTH)
        rope_physical_page = tl.load(
            raw_table_ptr
            + tl.minimum(tl.maximum(request, 0), num_requests - 1)
            * stride_table_req
            + tl.minimum(rope_page, RAW_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid_rope,
            other=-1,
        )
        valid_rope &= (rope_physical_page >= 0) & (
            rope_physical_page < num_rope_blocks
        )
        axis_offsets = tl.arange(0, 4)
        axis_positions = tl.load(
            rope_cache_ptr
            + tl.maximum(rope_physical_page, 0).to(tl.int64)
            * stride_rope_block
            + rope_offset * stride_rope_token
            + axis_offsets * stride_rope_dim,
            mask=valid_rope & (axis_offsets < 3),
            other=0,
        )
        time_position = tl.max(tl.where(axis_offsets == 0, axis_positions, 0))
        height_position = tl.max(
            tl.where(axis_offsets == 1, axis_positions, 0)
        )
        width_position = tl.max(tl.where(axis_offsets == 2, axis_positions, 0))
    else:
        time_position = first_position
        height_position = first_position
        width_position = first_position

    # HEAD_DIM=128 and ROTARY_DIM=64 in the production checkpoint.  Split the
    # normalized vector without dynamic local indexing: first head half holds
    # all rotary channels, second head half is the pass-through tail.
    head_pairs = tl.permute(tl.reshape(normalized, (2, BLOCK_D // 2)), (1, 0))
    rotary_values, pass_values = tl.split(head_pairs)
    rotary_pairs = tl.permute(
        tl.reshape(rotary_values, (2, ROTARY_DIM // 2)), (1, 0)
    )
    first_half, second_half = tl.split(rotary_pairs)
    frequencies = tl.arange(0, ROTARY_DIM // 2)
    if MROPE_INTERLEAVED:
        use_height = ((frequencies % 3) == 1) & (
            frequencies < 3 * MROPE_SECTION_H
        )
        use_width = ((frequencies % 3) == 2) & (
            frequencies < 3 * MROPE_SECTION_W
        )
    else:
        height_start = MROPE_SECTION_T
        width_start = height_start + MROPE_SECTION_H
        use_height = (frequencies >= height_start) & (
            frequencies < width_start
        )
        use_width = (frequencies >= width_start) & (
            frequencies < width_start + MROPE_SECTION_W
        )
    rope_positions = tl.where(
        use_height,
        height_position,
        tl.where(use_width, width_position, time_position),
    )
    valid_position = (rope_positions >= 0) & (rope_positions < num_cos_rows)
    safe_positions = tl.minimum(
        tl.maximum(rope_positions, 0), num_cos_rows - 1
    )
    cos = tl.load(
        cos_sin_cache_ptr
        + safe_positions * stride_cos_row
        + frequencies * stride_cos_dim,
        mask=valid_row & valid_position,
        other=0.0,
    )
    sin = tl.load(
        cos_sin_cache_ptr
        + safe_positions * stride_cos_row
        + (ROTARY_DIM // 2 + frequencies) * stride_cos_dim,
        mask=valid_row & valid_position,
        other=0.0,
    )
    # SGLang apply_rotary_emb casts cache values to the input dtype and
    # materializes each product before the final sum/subtraction.
    cos = cos.to(tl.bfloat16).to(tl.float32)
    sin = sin.to(tl.bfloat16).to(tl.float32)
    left = first_half.to(tl.float32)
    right = second_half.to(tl.float32)
    left_cos = (left * cos).to(tl.bfloat16).to(tl.float32)
    right_sin = (right * sin).to(tl.bfloat16).to(tl.float32)
    right_cos = (right * cos).to(tl.bfloat16).to(tl.float32)
    left_sin = (left * sin).to(tl.bfloat16).to(tl.float32)
    rotated_first = (left_cos - right_sin).to(tl.bfloat16)
    rotated_second = (right_cos + left_sin).to(tl.bfloat16)

    compressed_block = tl.maximum(compressed_slot, 0) // COMPRESSED_PAGE_SIZE
    compressed_token = tl.maximum(compressed_slot, 0) % COMPRESSED_PAGE_SIZE
    compressed_base = (
        compressed_cache_ptr
        + compressed_block.to(tl.int64) * stride_compressed_block
        + compressed_token * stride_compressed_token
    )
    tl.store(
        compressed_base + frequencies * stride_compressed_dim,
        rotated_first,
        mask=valid_row & valid_position,
    )
    tl.store(
        compressed_base
        + (ROTARY_DIM // 2 + frequencies) * stride_compressed_dim,
        rotated_second,
        mask=valid_row & valid_position,
    )
    pass_offsets = tl.arange(0, BLOCK_D // 2)
    tl.store(
        compressed_base + (ROTARY_DIM + pass_offsets) * stride_compressed_dim,
        pass_values,
        mask=valid_row & ((ROTARY_DIM + pass_offsets) < HEAD_DIM),
    )


def qsa_compress_norm_mrope_store_groups(
    raw_cache: torch.Tensor,
    raw_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compressed_cache: torch.Tensor,
    norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int,
    norm_eps: float,
    rotary_dim: int,
    mrope_section: tuple[int, int, int],
    mrope_interleaved: bool,
    rope_cache: torch.Tensor | None = None,
) -> None:
    """Fuse QSA compression, Gemma norm, Neox MRoPE, and cache insert."""

    if not _is_triton_device(raw_cache):
        raise NotImplementedError(
            "fused QSA compression requires accelerator Triton"
        )
    if raw_cache.dtype != torch.bfloat16:
        raise ValueError("QSA compression requires BF16 raw keys")
    rows = token_to_req.numel()
    if (
        raw_cache.ndim != 4
        or raw_cache.shape[2] != 1
        or not all(raw_cache.shape)
    ):
        raise ValueError("fused QSA compression received an invalid raw cache")
    head_dim = raw_cache.shape[3]
    if head_dim != 128 or rotary_dim != 64:
        raise ValueError(
            "fused QSA compression requires head_dim=128, rotary_dim=64"
        )
    if (
        compressed_cache.ndim != 4
        or compressed_cache.shape[2:] != (1, head_dim)
        or not all(compressed_cache.shape)
        or compressed_cache.device != raw_cache.device
        or compressed_cache.dtype != raw_cache.dtype
    ):
        raise ValueError(
            "fused QSA compressed cache must match the raw key width"
        )
    if raw_block_table.ndim != 2 or not all(raw_block_table.shape):
        raise ValueError(
            "fused QSA compression requires a nonempty block table"
        )
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError(
            "fused QSA compression metadata must match token rows"
        )
    if (
        norm_weight.shape != (head_dim,)
        or norm_weight.dtype != raw_cache.dtype
        or norm_weight.device != raw_cache.device
        or norm_weight.stride(0) != 1
    ):
        raise ValueError("fused QSA compression norm weight is incompatible")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != rotary_dim
        or not cos_sin_cache.shape[0]
        or cos_sin_cache.device != raw_cache.device
        or cos_sin_cache.dtype not in (raw_cache.dtype, torch.float32)
    ):
        raise ValueError(
            "fused QSA compression received an invalid RoPE cache"
        )
    if sum(mrope_section) != rotary_dim // 2:
        raise ValueError(
            "fused QSA MRoPE sections must cover half the rotary dim"
        )
    if compress_ratio <= 0 or norm_eps <= 0:
        raise ValueError(
            "fused QSA compression ratio and epsilon must be positive"
        )
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != raw_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
        or rope_cache.device != raw_cache.device
    ):
        raise ValueError("fused QSA packed MRoPE cache is invalid")
    if not rows:
        return
    if rope_cache is None:
        rope_cache = raw_cache
        load_mrope_positions = False
    else:
        load_mrope_positions = True
    _compress_norm_mrope_store_qsa_groups_kernel[(rows,)](
        raw_cache,
        rope_cache,
        raw_block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
        norm_weight,
        cos_sin_cache,
        compressed_cache,
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        raw_block_table.stride(0),
        raw_block_table.stride(1),
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        compressed_cache.stride(0),
        compressed_cache.stride(1),
        compressed_cache.stride(3),
        rows,
        raw_cache.shape[0],
        rope_cache.shape[0],
        compressed_cache.shape[0],
        raw_block_table.shape[0],
        cos_sin_cache.shape[0],
        float(norm_eps),
        RAW_PAGE_SIZE=raw_cache.shape[1],
        RAW_TABLE_WIDTH=raw_block_table.shape[1],
        ROPE_PAGE_SIZE=rope_cache.shape[1],
        COMPRESSED_PAGE_SIZE=compressed_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        MROPE_SECTION_T=mrope_section[0],
        MROPE_SECTION_H=mrope_section[1],
        MROPE_SECTION_W=mrope_section[2],
        MROPE_INTERLEAVED=mrope_interleaved,
        LOAD_MROPE_POSITIONS=load_mrope_positions,
        num_warps=4,
        enable_fp_fusion=False,
    )


__all__ = ["qsa_compress_norm_mrope_store_groups"]
