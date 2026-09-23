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

import math

import torch
import triton
import triton.language as tl

_BLOCK_M = 64
_BLOCK_N = 128
_BLOCK_R = 64


@triton.jit
def _extend_attention_kunlunxin_pack_prefix_kernel(
    kb_ptr,
    vb_ptr,
    kt_ptr,
    vg_ptr,
    kvi_ptr,
    kvp_ptr,
    head_dim,
    keys_per_batch,
    stride_kb_token,
    stride_kb_head,
    stride_kb_dim,
    stride_vb_token,
    stride_vb_head,
    stride_vb_dim,
    stride_vg_token,
    stride_vg_head,
    stride_vg_dim,
    kt_stride_batch,
    kt_stride_head,
    kt_stride_dim,
    kt_stride_key,
    block_r: tl.constexpr,
    block_d: tl.constexpr,
    buffer_tokens: tl.constexpr,
):
    row_block = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    sequence_id = tl.program_id(2)
    prefix_start = tl.load(kvp_ptr + sequence_id)
    prefix_end = tl.load(kvp_ptr + sequence_id + 1)
    prefix_length = prefix_end - prefix_start
    rows = row_block * block_r + tl.arange(0, block_r)
    row_mask = rows < prefix_length
    dim_offsets = tl.arange(0, block_d)
    dim_mask = dim_offsets < head_dim
    tile_mask = row_mask[:, None] & dim_mask[None, :]
    token_ids = tl.load(kvi_ptr + prefix_start + rows, mask=row_mask, other=0)
    token_ids = tl.where(token_ids < 0, token_ids + buffer_tokens, token_ids)
    k = tl.load(
        kb_ptr
        + token_ids[:, None] * stride_kb_token
        + kv_head_id * stride_kb_head
        + dim_offsets[None, :] * stride_kb_dim,
        mask=tile_mask,
        other=0.0,
    )
    v = tl.load(
        vb_ptr
        + token_ids[:, None] * stride_vb_token
        + kv_head_id * stride_vb_head
        + dim_offsets[None, :] * stride_vb_dim,
        mask=tile_mask,
        other=0.0,
    )
    tl.store(
        kt_ptr
        + sequence_id * kt_stride_batch
        + kv_head_id * kt_stride_head
        + dim_offsets[None, :] * kt_stride_dim
        + rows[:, None] * kt_stride_key,
        k,
        mask=tile_mask,
    )
    tl.store(
        vg_ptr
        + (sequence_id * keys_per_batch + rows)[:, None] * stride_vg_token
        + kv_head_id * stride_vg_head
        + dim_offsets[None, :] * stride_vg_dim,
        v,
        mask=tile_mask,
    )


@triton.jit
def _extend_attention_kunlunxin_pack_extend_kernel(
    ke_ptr,
    ve_ptr,
    kt_ptr,
    vg_ptr,
    qo_ptr,
    kvp_ptr,
    head_dim,
    keys_per_batch,
    stride_ke_token,
    stride_ke_head,
    stride_ke_dim,
    stride_ve_token,
    stride_ve_head,
    stride_ve_dim,
    stride_vg_token,
    stride_vg_head,
    stride_vg_dim,
    kt_stride_batch,
    kt_stride_head,
    kt_stride_dim,
    kt_stride_key,
    block_r: tl.constexpr,
    block_d: tl.constexpr,
):
    row_block = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    sequence_id = tl.program_id(2)
    query_start = tl.load(qo_ptr + sequence_id)
    query_end = tl.load(qo_ptr + sequence_id + 1)
    extend_length = query_end - query_start
    prefix_start = tl.load(kvp_ptr + sequence_id)
    prefix_end = tl.load(kvp_ptr + sequence_id + 1)
    prefix_length = prefix_end - prefix_start
    rows = row_block * block_r + tl.arange(0, block_r)
    row_mask = rows < extend_length
    dim_offsets = tl.arange(0, block_d)
    dim_mask = dim_offsets < head_dim
    tile_mask = row_mask[:, None] & dim_mask[None, :]
    k = tl.load(
        ke_ptr
        + (query_start + rows)[:, None] * stride_ke_token
        + kv_head_id * stride_ke_head
        + dim_offsets[None, :] * stride_ke_dim,
        mask=tile_mask,
        other=0.0,
    )
    v = tl.load(
        ve_ptr
        + (query_start + rows)[:, None] * stride_ve_token
        + kv_head_id * stride_ve_head
        + dim_offsets[None, :] * stride_ve_dim,
        mask=tile_mask,
        other=0.0,
    )
    tl.store(
        kt_ptr
        + sequence_id * kt_stride_batch
        + kv_head_id * kt_stride_head
        + dim_offsets[None, :] * kt_stride_dim
        + (prefix_length + rows)[:, None] * kt_stride_key,
        k,
        mask=tile_mask,
    )
    tl.store(
        vg_ptr
        + (sequence_id * keys_per_batch + prefix_length + rows)[:, None]
        * stride_vg_token
        + kv_head_id * stride_vg_head
        + dim_offsets[None, :] * stride_vg_dim,
        v,
        mask=tile_mask,
    )


@triton.jit
def _extend_attention_kunlunxin_score_kernel(
    q_ptr,
    kt_ptr,
    score_ptr,
    qo_ptr,
    kvp_ptr,
    sm_scale,
    head_dim,
    num_key_blocks,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    kt_stride_batch,
    kt_stride_head,
    kt_stride_dim,
    kt_stride_key,
    stride_score_token,
    stride_score_head,
    stride_score_key,
    group_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    tile_id = tl.program_id(0)
    sequence_id = tl.program_id(1)
    head_id = tl.program_id(2)
    query_block = tile_id // num_key_blocks
    key_block = tile_id % num_key_blocks
    kv_head_id = head_id // group_size
    query_start = tl.load(qo_ptr + sequence_id)
    query_end = tl.load(qo_ptr + sequence_id + 1)
    extend_length = query_end - query_start
    prefix_start = tl.load(kvp_ptr + sequence_id)
    prefix_end = tl.load(kvp_ptr + sequence_id + 1)
    prefix_length = prefix_end - prefix_start
    total_length = prefix_length + extend_length
    offs_m = query_block * block_m + tl.arange(0, block_m)
    offs_n = key_block * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    mask_m = offs_m < extend_length
    mask_n = offs_n < total_length
    q_ptrs = (
        q_ptr
        + (query_start + offs_m)[:, None] * stride_q_token
        + head_id * stride_q_head
        + offs_k[None, :] * stride_q_dim
    )
    k_ptrs = (
        kt_ptr
        + sequence_id * kt_stride_batch
        + kv_head_id * kt_stride_head
        + offs_k[:, None] * kt_stride_dim
        + offs_n[None, :] * kt_stride_key
    )
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for d_start in range(0, head_dim, block_k):
        mask_d = d_start + offs_k < head_dim
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(
            q.to(tl.float32),
            k.to(tl.float32),
            out_dtype=tl.float32,
            input_precision="ieee",
        )
        q_ptrs += block_k * stride_q_dim
        k_ptrs += block_k * kt_stride_dim
    tl.store(
        score_ptr
        + (query_start + offs_m)[:, None] * stride_score_token
        + head_id * stride_score_head
        + offs_n[None, :] * stride_score_key,
        acc * sm_scale,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _extend_attention_kunlunxin_exp_kernel(
    score_ptr,
    prob_ptr,
    qo_ptr,
    kvp_ptr,
    HEADS_KEYS: tl.constexpr,
    KEYS: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    local_row = tl.program_id(0)
    sequence_id = tl.program_id(1)
    head_start = tl.program_id(2) * HEADS_PER_PROGRAM
    query_start = tl.load(qo_ptr + sequence_id)
    query_end = tl.load(qo_ptr + sequence_id + 1)
    extend_length = query_end - query_start
    prefix_start = tl.load(kvp_ptr + sequence_id)
    prefix_end = tl.load(kvp_ptr + sequence_id + 1)
    prefix_length = prefix_end - prefix_start
    live = tl.minimum(tl.maximum(extend_length - local_row, 0), 1)
    limit = prefix_length + local_row
    base = (query_start + local_row) * HEADS_KEYS + head_start * KEYS
    keys = tl.arange(0, KEYS)
    heads = tl.arange(0, HEADS_PER_PROGRAM)
    offs = heads[:, None] * KEYS + keys[None, :]
    valid = keys[None, :] <= limit
    for active in range(0, live):
        raw = tl.load(score_ptr + base + offs)
        scores = tl.where(valid, raw, float("-inf"))
        row_max = tl.max(scores, axis=1)
        probabilities = tl.exp(scores - row_max[:, None])
        tl.store(
            prob_ptr + base + offs,
            probabilities / tl.sum(probabilities, axis=1)[:, None],
        )


@triton.jit
def _extend_attention_kunlunxin_value_kernel(
    prob_ptr,
    vg_ptr,
    acc_ptr,
    qo_ptr,
    kvp_ptr,
    keys_per_batch,
    head_dim,
    stride_prob_token,
    stride_prob_head,
    stride_prob_key,
    stride_vg_token,
    stride_vg_head,
    stride_vg_dim,
    stride_acc_token,
    stride_acc_head,
    stride_acc_dim,
    group_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    query_block = tl.program_id(0)
    head_id = tl.program_id(1)
    sequence_id = tl.program_id(2)
    kv_head_id = head_id // group_size
    query_start = tl.load(qo_ptr + sequence_id)
    query_end = tl.load(qo_ptr + sequence_id + 1)
    extend_length = query_end - query_start
    prefix_start = tl.load(kvp_ptr + sequence_id)
    prefix_end = tl.load(kvp_ptr + sequence_id + 1)
    prefix_length = prefix_end - prefix_start
    key_end = tl.minimum(
        prefix_length + extend_length,
        prefix_length + query_block * block_m + block_m,
    )
    row_live = tl.minimum(
        tl.maximum(extend_length - query_block * block_m, 0), 1
    )
    offs_m = query_block * block_m + tl.arange(0, block_m)
    mask_m = offs_m < extend_length
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)
    rows = query_start + offs_m
    prob_base = (
        prob_ptr
        + rows[:, None] * stride_prob_token
        + head_id * stride_prob_head
    )
    value_base = (
        vg_ptr
        + sequence_id * keys_per_batch * stride_vg_token
        + kv_head_id * stride_vg_head
        + offs_d[None, :] * stride_vg_dim
    )
    accumulator = tl.zeros((block_m, block_d), dtype=tl.float32)
    for key_start in range(0, key_end * row_live, block_n):
        keys = key_start + offs_n
        v = tl.load(value_base + keys[:, None] * stride_vg_token)
        weights = tl.load(
            prob_base + keys[None, :] * stride_prob_key,
            mask=mask_m[:, None],
            other=0.0,
        ).to(v.dtype)
        accumulator += tl.dot(
            weights.to(tl.float32),
            v.to(tl.float32),
            out_dtype=tl.float32,
            input_precision="ieee",
        )
    tl.store(
        acc_ptr
        + rows[:, None] * stride_acc_token
        + head_id * stride_acc_head
        + offs_d[None, :] * stride_acc_dim,
        accumulator,
        mask=mask_m[:, None] & (offs_d[None, :] < head_dim),
    )


def _prepare_index(tensor, device):
    return tensor.to(device=device, dtype=torch.int32).contiguous()


def extend_attention(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    max_len_extend,
):
    total_extend, num_heads, head_dim = q_extend.shape
    num_kv_heads = k_extend.shape[1]
    group_size = num_heads // num_kv_heads
    batch_size = qo_indptr.numel() - 1
    device = q_extend.device
    if batch_size <= 0 or total_extend == 0:
        return torch.zeros(
            (total_extend, num_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )
    output = torch.empty(
        (total_extend, num_heads, head_dim), dtype=torch.float32, device=device
    )
    qo = _prepare_index(qo_indptr, device)
    kvp = _prepare_index(kv_indptr, device)
    kvi = _prepare_index(kv_indices, device)
    extend_lengths = qo[1:] - qo[:-1]
    prefix_lengths = kvp[1:] - kvp[:-1]
    max_extend = max(int(extend_lengths.max().item()), 1)
    max_prefix = max(int(prefix_lengths.max().item()), 0)
    max_total = max(int((extend_lengths + prefix_lengths).max().item()), 1)
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_k = 32 if head_dim % 32 == 0 else 16
    kdim_pad = triton.cdiv(head_dim, block_k) * block_k
    keys_per_batch = max(_BLOCK_N, triton.next_power_of_2(max_total))
    rows_pad = total_extend + _BLOCK_M
    keys_t = torch.zeros(
        (batch_size, num_kv_heads, kdim_pad, keys_per_batch),
        dtype=torch.float32,
        device=device,
    )
    values = torch.zeros(
        (batch_size * keys_per_batch, num_kv_heads, block_d),
        dtype=torch.float32,
        device=device,
    )
    scores = torch.zeros(
        (rows_pad, num_heads, keys_per_batch),
        dtype=torch.float32,
        device=device,
    )
    probs = torch.zeros(
        (rows_pad, num_heads, keys_per_batch),
        dtype=torch.float32,
        device=device,
    )
    if max_prefix > 0:
        _extend_attention_kunlunxin_pack_prefix_kernel[
            triton.cdiv(max_prefix, _BLOCK_R), num_kv_heads, batch_size
        ](
            k_buffer,
            v_buffer,
            keys_t,
            values,
            kvi,
            kvp,
            head_dim,
            keys_per_batch,
            k_buffer.stride(0),
            k_buffer.stride(1),
            k_buffer.stride(2),
            v_buffer.stride(0),
            v_buffer.stride(1),
            v_buffer.stride(2),
            values.stride(0),
            values.stride(1),
            values.stride(2),
            keys_t.stride(0),
            keys_t.stride(1),
            keys_t.stride(2),
            keys_t.stride(3),
            block_r=_BLOCK_R,
            block_d=block_d,
            num_warps=4,
            num_stages=1,
            buffer_tokens=k_buffer.shape[0],
        )
    _extend_attention_kunlunxin_pack_extend_kernel[
        triton.cdiv(max_extend, _BLOCK_R), num_kv_heads, batch_size
    ](
        k_extend,
        v_extend,
        keys_t,
        values,
        qo,
        kvp,
        head_dim,
        keys_per_batch,
        k_extend.stride(0),
        k_extend.stride(1),
        k_extend.stride(2),
        v_extend.stride(0),
        v_extend.stride(1),
        v_extend.stride(2),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        keys_t.stride(0),
        keys_t.stride(1),
        keys_t.stride(2),
        keys_t.stride(3),
        block_r=_BLOCK_R,
        block_d=block_d,
        num_warps=4,
        num_stages=1,
    )
    num_query_blocks = triton.cdiv(max_extend, _BLOCK_M)
    num_key_blocks = keys_per_batch // _BLOCK_N
    _extend_attention_kunlunxin_score_kernel[
        num_query_blocks * num_key_blocks, batch_size, num_heads
    ](
        q_extend,
        keys_t,
        scores,
        qo,
        kvp,
        1.0 / math.sqrt(head_dim),
        head_dim,
        num_key_blocks,
        q_extend.stride(0),
        q_extend.stride(1),
        q_extend.stride(2),
        keys_t.stride(0),
        keys_t.stride(1),
        keys_t.stride(2),
        keys_t.stride(3),
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        group_size=group_size,
        block_m=_BLOCK_M,
        block_n=_BLOCK_N,
        block_k=block_k,
        num_warps=4,
        num_stages=1,
    )
    heads_keys = num_heads * keys_per_batch
    heads_per_program = (
        4 if num_heads % 4 == 0 else 2 if num_heads % 2 == 0 else 1
    )
    _extend_attention_kunlunxin_exp_kernel[
        max_extend, batch_size, num_heads // heads_per_program
    ](
        scores,
        probs,
        qo,
        kvp,
        HEADS_KEYS=heads_keys,
        KEYS=keys_per_batch,
        HEADS_PER_PROGRAM=heads_per_program,
        num_warps=4,
        num_stages=1,
    )
    _extend_attention_kunlunxin_value_kernel[
        num_query_blocks, num_heads, batch_size
    ](
        probs,
        values,
        output,
        qo,
        kvp,
        keys_per_batch,
        head_dim,
        probs.stride(0),
        probs.stride(1),
        probs.stride(2),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        group_size=group_size,
        block_m=_BLOCK_M,
        block_n=_BLOCK_N,
        block_d=block_d,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["extend_attention"]
