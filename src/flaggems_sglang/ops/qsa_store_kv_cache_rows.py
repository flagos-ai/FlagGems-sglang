# SPDX-License-Identifier: Apache-2.0
# Vendor-derived sections retain copyright of contributors to the vLLM project.
# Copyright 2026 FlagOS Contributors
"""qsa_store_kv_cache_rows; adapted from FlagGems-vllm PR #455 dependencies.

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
def _store_qsa_kv_rows_kernel(
    k_cache_ptr,
    v_cache_ptr,
    slots_ptr,
    k_rows_ptr,
    v_rows_ptr,
    stride_k_cache_block,
    stride_k_cache_token,
    stride_k_cache_head,
    stride_k_cache_dim,
    stride_v_cache_block,
    stride_v_cache_token,
    stride_v_cache_head,
    stride_v_cache_dim,
    stride_k_rows_row,
    stride_k_rows_head,
    stride_k_rows_dim,
    stride_v_rows_row,
    stride_v_rows_head,
    stride_v_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Store K and V together while preserving arbitrary cache strides."""

    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    k_values = tl.load(
        k_rows_ptr
        + row * stride_k_rows_row
        + head * stride_k_rows_head
        + dims * stride_k_rows_dim,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
        other=0,
    )
    v_values = tl.load(
        v_rows_ptr
        + row * stride_v_rows_row
        + head * stride_v_rows_head
        + dims * stride_v_rows_dim,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
        other=0,
    )
    tl.store(
        k_cache_ptr
        + block * stride_k_cache_block
        + token * stride_k_cache_token
        + head * stride_k_cache_head
        + dims * stride_k_cache_dim,
        k_values,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
    )
    tl.store(
        v_cache_ptr
        + block * stride_v_cache_block
        + token * stride_v_cache_token
        + head * stride_v_cache_head
        + dims * stride_v_cache_dim,
        v_values,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
    )


def qsa_store_kv_cache_rows(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Store paired K/V rows with one cross-vendor Triton launch."""

    if not _is_triton_device(k_cache):
        raise NotImplementedError(
            "QSA K/V cache stores require an accelerator Triton backend"
        )
    if (
        k_cache.ndim != 4
        or v_cache.shape != k_cache.shape
        or key.ndim != 3
        or value.shape != key.shape
    ):
        raise ValueError("QSA K/V cache store received invalid tensor shapes")
    if key.shape != (slot_mapping.numel(), k_cache.shape[2], k_cache.shape[3]):
        raise ValueError("QSA K/V rows and cache geometry are incompatible")
    if key.dtype != k_cache.dtype or value.dtype != v_cache.dtype:
        raise ValueError("QSA K/V rows and caches must have matching dtypes")
    if not all(k_cache.shape):
        raise ValueError("QSA K/V cache dimensions must be nonzero")
    if not key.shape[0]:
        return
    _store_qsa_kv_rows_kernel[(key.shape[0], key.shape[1])](
        k_cache,
        v_cache,
        slot_mapping,
        key,
        value,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key.shape[0],
        k_cache.shape[0],
        PAGE_SIZE=k_cache.shape[1],
        NUM_HEADS=key.shape[1],
        HEAD_DIM=key.shape[2],
        BLOCK_D=triton.next_power_of_2(key.shape[2]),
        num_warps=4,
    )


__all__ = ["qsa_store_kv_cache_rows"]
