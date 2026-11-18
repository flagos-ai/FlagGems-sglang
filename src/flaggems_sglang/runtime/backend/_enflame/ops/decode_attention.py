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
def _direct_kernel(
    q,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    output,
    sm_scale,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_head = tl.program_id(0)
    batch = tl.program_id(1)
    batch_head = batch * H_Q + query_head
    kv_head = query_head // (H_Q // H_KV)
    start = tl.load(kv_indptr + batch)
    end = tl.load(kv_indptr + batch + 1)
    sequence_length = end - start
    dim_lanes = tl.arange(0, BLOCK_D)
    dim_mask = dim_lanes < D
    log2e: tl.constexpr = 1.4426950408889634
    query = tl.load(
        q + batch_head * D + dim_lanes,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32) * (sm_scale * log2e)
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for token in range(0, sequence_length):
        token_index = tl.load(kv_indices + start + token)
        kv_base = token_index * H_KV * D + kv_head * D
        keys = tl.load(
            k_buffer + kv_base + dim_lanes,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        logit = tl.sum(keys * query, axis=0)
        next_max = tl.maximum(running_max, logit)
        correction = tl.math.exp2(running_max - next_max)
        probability = tl.math.exp2(logit - next_max)
        running_sum = running_sum * correction + probability
        values = tl.load(
            v_buffer + kv_base + dim_lanes,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * correction + probability * values
        running_max = next_max
    tl.store(
        output + batch_head * D + dim_lanes,
        accumulator / running_sum,
        mask=dim_mask,
    )


@triton.jit
def _dot_split_kernel(
    q,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    partial_output,
    partial_max,
    partial_sum,
    sm_scale,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID_X: tl.constexpr,
):
    work_id = tl.program_id(0) + tl.program_id(1) * GRID_X
    split = work_id % SPLITS
    batch_kv = work_id // SPLITS
    kv_head = batch_kv % H_KV
    batch = batch_kv // H_KV
    start = tl.load(kv_indptr + batch)
    end = tl.load(kv_indptr + batch + 1)
    sequence_length = end - start
    head_lanes = tl.arange(0, BLOCK_M)
    dim_lanes = tl.arange(0, BLOCK_D)
    token_lanes = tl.arange(0, BLOCK_N)
    query_heads = kv_head * GROUP_SIZE + head_lanes
    query_mask = head_lanes < GROUP_SIZE
    dim_mask = dim_lanes < D
    query = tl.load(
        q + (batch * H_Q + query_heads[:, None]) * D + dim_lanes[None, :],
        mask=query_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    token_offsets = split * BLOCK_N + token_lanes
    token_mask = token_offsets < sequence_length
    token_indices = tl.load(
        kv_indices + start + token_offsets,
        mask=token_mask,
        other=0,
    )
    keys = tl.load(
        k_buffer
        + (token_indices[:, None] * H_KV + kv_head) * D
        + dim_lanes[None, :],
        mask=token_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    keys = tl.trans(keys)
    log2e: tl.constexpr = 1.4426950408889634
    logits = tl.dot(query, keys, allow_tf32=False) * (sm_scale * log2e)
    logits = tl.where(token_mask[None, :], logits, -float("inf"))
    maxima = tl.max(logits, axis=1)
    probabilities = tl.math.exp2(logits - maxima[:, None])
    sums = tl.sum(probabilities, axis=1)
    values = tl.load(
        v_buffer
        + (token_indices[:, None] * H_KV + kv_head) * D
        + dim_lanes[None, :],
        mask=token_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    accumulator = tl.dot(
        probabilities.to(values.dtype),
        values,
        allow_tf32=False,
    )
    partial_ids = (batch * H_Q + query_heads) * SPLITS + split
    tl.store(
        partial_output + partial_ids[:, None] * D + dim_lanes[None, :],
        accumulator,
        mask=query_mask[:, None] & dim_mask[None, :],
    )
    tl.store(partial_max + partial_ids, maxima, mask=query_mask)
    tl.store(partial_sum + partial_ids, sums, mask=query_mask)


@triton.jit
def _dot_reduce_kernel(
    partial_output,
    partial_max,
    partial_sum,
    output,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    GRID_X: tl.constexpr,
):
    base = tl.program_id(0) + tl.program_id(1) * GRID_X
    split_lanes = tl.arange(0, BLOCK_SPLITS)
    split_mask = split_lanes < SPLITS
    dim_lanes = tl.arange(0, BLOCK_D)
    dim_mask = dim_lanes < D
    for head_offset in range(0, BLOCK_HEADS):
        batch_head = base * BLOCK_HEADS + head_offset
        partial_ids = batch_head * SPLITS + split_lanes
        maxima = tl.load(
            partial_max + partial_ids,
            mask=split_mask,
            other=-float("inf"),
        )
        global_max = tl.max(maxima, axis=0)
        weights = tl.math.exp2(maxima - global_max)
        sums = tl.load(partial_sum + partial_ids, mask=split_mask, other=0.0)
        denominator = tl.sum(sums * weights, axis=0)
        values = tl.load(
            partial_output + partial_ids[:, None] * D + dim_lanes[None, :],
            mask=split_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        numerator = tl.sum(values * weights[:, None], axis=0)
        tl.store(
            output + batch_head * D + dim_lanes,
            numerator / denominator,
            mask=dim_mask,
        )


@triton.jit
def _split_kernel(
    q,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    partial_output,
    partial_max,
    partial_sum,
    sm_scale,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLIT_TOKENS: tl.constexpr,
    GRID_X: tl.constexpr,
):
    work_id = tl.program_id(0) + tl.program_id(1) * GRID_X
    split = work_id % SPLITS
    kv_work = work_id // SPLITS
    kv_head = kv_work % H_KV
    batch = kv_work // H_KV
    start = tl.load(kv_indptr + batch)
    end = tl.load(kv_indptr + batch + 1)
    sequence_length = end - start
    dim_lanes = tl.arange(0, BLOCK_D)
    log2e: tl.constexpr = 1.4426950408889634
    q_base = (batch * H_Q + kv_head * GROUP) * D
    q0 = tl.load(q + q_base + dim_lanes).to(tl.float32) * (sm_scale * log2e)
    q1 = tl.load(q + q_base + D + dim_lanes).to(tl.float32) * (
        sm_scale * log2e
    )
    q2 = tl.load(q + q_base + 2 * D + dim_lanes).to(tl.float32) * (
        sm_scale * log2e
    )
    q3 = tl.load(q + q_base + 3 * D + dim_lanes).to(tl.float32) * (
        sm_scale * log2e
    )
    m0 = -float("inf")
    m1 = -float("inf")
    m2 = -float("inf")
    m3 = -float("inf")
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    a0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a3 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    low = split * SPLIT_TOKENS
    high = low + SPLIT_TOKENS
    high = tl.minimum(high, sequence_length)
    kv_head_base = kv_head * D
    for i in range(low, high):
        token_index = tl.load(kv_indices + start + i)
        kv_base = token_index * H_KV * D + kv_head_base
        keys = tl.load(k_buffer + kv_base + dim_lanes).to(tl.float32)
        l0 = tl.sum(keys * q0, axis=0)
        l1 = tl.sum(keys * q1, axis=0)
        l2 = tl.sum(keys * q2, axis=0)
        l3 = tl.sum(keys * q3, axis=0)
        nm0 = tl.maximum(m0, l0)
        nm1 = tl.maximum(m1, l1)
        nm2 = tl.maximum(m2, l2)
        nm3 = tl.maximum(m3, l3)
        c0 = tl.math.exp2(m0 - nm0)
        c1 = tl.math.exp2(m1 - nm1)
        c2 = tl.math.exp2(m2 - nm2)
        c3 = tl.math.exp2(m3 - nm3)
        p0 = tl.math.exp2(l0 - nm0)
        p1 = tl.math.exp2(l1 - nm1)
        p2 = tl.math.exp2(l2 - nm2)
        p3 = tl.math.exp2(l3 - nm3)
        s0 = s0 * c0 + p0
        s1 = s1 * c1 + p1
        s2 = s2 * c2 + p2
        s3 = s3 * c3 + p3
        values = tl.load(v_buffer + kv_base + dim_lanes).to(tl.float32)
        a0 = a0 * c0 + p0 * values
        a1 = a1 * c1 + p1 * values
        a2 = a2 * c2 + p2 * values
        a3 = a3 * c3 + p3 * values
        m0 = nm0
        m1 = nm1
        m2 = nm2
        m3 = nm3
    out_base = batch * H_Q + kv_head * GROUP
    id0 = (out_base + 0) * SPLITS + split
    id1 = (out_base + 1) * SPLITS + split
    id2 = (out_base + 2) * SPLITS + split
    id3 = (out_base + 3) * SPLITS + split
    tl.store(partial_output + id0 * D + dim_lanes, a0)
    tl.store(partial_output + id1 * D + dim_lanes, a1)
    tl.store(partial_output + id2 * D + dim_lanes, a2)
    tl.store(partial_output + id3 * D + dim_lanes, a3)
    tl.store(partial_max + id0, m0)
    tl.store(partial_max + id1, m1)
    tl.store(partial_max + id2, m2)
    tl.store(partial_max + id3, m3)
    tl.store(partial_sum + id0, s0)
    tl.store(partial_sum + id1, s1)
    tl.store(partial_sum + id2, s2)
    tl.store(partial_sum + id3, s3)


@triton.jit
def _reduce_kernel(
    partial_output,
    partial_max,
    partial_sum,
    output,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    GRID_X: tl.constexpr,
):
    batch_head = tl.program_id(0) + tl.program_id(1) * GRID_X
    split_lanes = tl.arange(0, BLOCK_SPLITS)
    split_mask = split_lanes < SPLITS
    partial_ids = batch_head * SPLITS + split_lanes
    maxima = tl.load(
        partial_max + partial_ids,
        mask=split_mask,
        other=-float("inf"),
    )
    global_max = tl.max(maxima, axis=0)
    weights = tl.math.exp2(maxima - global_max)
    sums = tl.load(partial_sum + partial_ids, mask=split_mask, other=0.0)
    denominator = tl.sum(sums * weights, axis=0)
    dim_lanes = tl.arange(0, BLOCK_D)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for split in range(0, SPLITS):
        weight = tl.math.exp2(
            tl.load(partial_max + batch_head * SPLITS + split) - global_max
        )
        values = tl.load(
            partial_output + (batch_head * SPLITS + split) * D + dim_lanes,
        )
        accumulator += values * weight
    tl.store(
        output + batch_head * D + dim_lanes,
        accumulator / denominator,
    )


def _grid(total_work):
    grid_x = min(total_work, 32768)
    while total_work % grid_x:
        grid_x -= 1
    return grid_x, total_work // grid_x


_USE_DOT_KERNEL = None


def _decode_attention_gqa(
    q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale, output
):
    # The tl.dot split kernel is ~5x faster but requires the toolchain to lower
    # rank-2 tiles into tl.dot operands (memref_ext.slice_* regression since
    # 2026-11-07 20:22 breaks that lowering). Probe once per process: fall back
    # permanently to the pure-1D kernel when the dot variant fails to compile,
    # and pick the dot variant back up automatically once the toolchain is
    # fixed and the kernel compiles again.
    global _USE_DOT_KERNEL
    batch_size, query_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]
    batch_heads = batch_size * query_heads
    if _USE_DOT_KERNEL is not False:
        try:
            split_tokens = 64
            max_sequence_length = k_buffer.shape[0] // batch_size
            splits = triton.cdiv(max_sequence_length, split_tokens)
            partial_output = torch.empty(
                (batch_heads * splits, head_dim),
                dtype=torch.float32,
                device=q.device,
            )
            partial_max = torch.empty(
                (batch_heads * splits,),
                dtype=torch.float32,
                device=q.device,
            )
            partial_sum = torch.empty_like(partial_max)
            split_grid = _grid(batch_size * kv_heads * splits)
            block_heads_reduce = 1
            while (
                block_heads_reduce < 32
                and batch_heads // (block_heads_reduce * 2) >= 128
            ):
                block_heads_reduce *= 2
            reduce_grid = _grid(batch_heads // block_heads_reduce)
            block_heads = 4
            _dot_split_kernel[split_grid](
                q,
                k_buffer,
                v_buffer,
                kv_indptr,
                kv_indices,
                partial_output,
                partial_max,
                partial_sum,
                sm_scale,
                query_heads,
                kv_heads,
                head_dim,
                4,
                splits,
                block_heads,
                128,
                split_tokens,
                split_grid[0],
                num_warps=1,
                num_stages=2,
            )
            _dot_reduce_kernel[reduce_grid](
                partial_output,
                partial_max,
                partial_sum,
                output,
                head_dim,
                splits,
                block_heads_reduce,
                128,
                triton.next_power_of_2(splits),
                reduce_grid[0],
                num_warps=1,
                num_stages=2,
            )
            _USE_DOT_KERNEL = True
            return
        except Exception:
            _USE_DOT_KERNEL = False
    split_tokens = 64
    max_sequence_length = k_buffer.shape[0] // batch_size
    splits = triton.cdiv(max_sequence_length, split_tokens)
    partial_output = torch.empty(
        (batch_heads * splits, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    partial_max = torch.empty(
        (batch_heads * splits,),
        dtype=torch.float32,
        device=q.device,
    )
    partial_sum = torch.empty_like(partial_max)
    split_grid = _grid(batch_size * kv_heads * splits)
    reduce_grid = _grid(batch_heads)
    _split_kernel[split_grid](
        q,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        partial_output,
        partial_max,
        partial_sum,
        sm_scale,
        query_heads,
        kv_heads,
        head_dim,
        query_heads // kv_heads,
        splits,
        head_dim,
        split_tokens,
        split_grid[0],
        num_warps=1,
        num_stages=2,
    )
    _reduce_kernel[reduce_grid](
        partial_output,
        partial_max,
        partial_sum,
        output,
        head_dim,
        splits,
        head_dim,
        triton.next_power_of_2(splits),
        reduce_grid[0],
        num_warps=1,
        num_stages=2,
    )


def decode_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    batch_size, query_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]
    output = torch.empty(
        (batch_size, query_heads, v_buffer.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    if query_heads == 32 and kv_heads == 8 and head_dim == 128:
        _decode_attention_gqa(
            q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale, output
        )
        return output
    block_dim = triton.next_power_of_2(head_dim)
    _direct_kernel[(query_heads, batch_size)](
        q,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        output,
        sm_scale,
        query_heads,
        kv_heads,
        head_dim,
        k_buffer.shape[0] // batch_size,
        block_dim,
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = ["decode_attention"]
