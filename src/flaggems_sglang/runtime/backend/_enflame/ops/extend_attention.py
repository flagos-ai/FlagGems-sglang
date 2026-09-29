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
_BLOCK_N = 64
_MAX_PROGRAMS = 8192
_BLOCK_R = 64


@triton.jit
def _extend_attention_gather_kernel(
    kb_ptr,
    vb_ptr,
    kg_ptr,
    vg_ptr,
    kvi_ptr,
    total_prefix,
    head_dim,
    stride_kb_token,
    stride_kb_head,
    stride_kb_dim,
    stride_vb_token,
    stride_vb_head,
    stride_vb_dim,
    stride_kg_token,
    stride_kg_head,
    stride_kg_dim,
    block_r: tl.constexpr,
    block_d: tl.constexpr,
    buffer_tokens: tl.constexpr,
):
    row_block = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    rows = row_block * block_r + tl.arange(0, block_r)
    row_mask = rows < total_prefix
    token_ids = tl.load(kvi_ptr + rows, mask=row_mask, other=0)
    token_ids = tl.where(token_ids < 0, token_ids + buffer_tokens, token_ids)
    dim_offsets = tl.arange(0, block_d)
    dim_mask = dim_offsets < head_dim
    tile_mask = row_mask[:, None] & dim_mask[None, :]
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
    offsets = (
        rows[:, None] * stride_kg_token
        + kv_head_id * stride_kg_head
        + dim_offsets[None, :] * stride_kg_dim
    )
    tl.store(kg_ptr + offsets, k, mask=tile_mask)
    tl.store(vg_ptr + offsets, v, mask=tile_mask)


@triton.jit
def _extend_attention_kernel(
    q_ptr,
    ke_ptr,
    ve_ptr,
    kb_ptr,
    vb_ptr,
    out_ptr,
    qo_ptr,
    kvp_ptr,
    kvi_ptr,
    sm_scale,
    num_heads,
    group_size,
    head_dim,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_ke_token,
    stride_ke_head,
    stride_ke_dim,
    stride_ve_token,
    stride_ve_head,
    stride_ve_dim,
    stride_kb_token,
    stride_kb_head,
    stride_kb_dim,
    stride_vb_token,
    stride_vb_head,
    stride_vb_dim,
    stride_out_token,
    stride_out_head,
    stride_out_dim,
    pid_base,
    num_query_blocks,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    task = pid_base + tl.program_id(0)
    query_block = task % num_query_blocks
    head_task = task // num_query_blocks
    head_id = head_task % num_heads
    sequence_id = head_task // num_heads
    kv_head_id = head_id // group_size
    query_start = tl.load(qo_ptr + sequence_id)
    query_end = tl.load(qo_ptr + sequence_id + 1)
    extend_length = query_end - query_start
    if query_block * block_m >= extend_length:
        return
    prefix_start = tl.load(kvp_ptr + sequence_id)
    prefix_end = tl.load(kvp_ptr + sequence_id + 1)
    prefix_length = prefix_end - prefix_start
    query_offsets = query_block * block_m + tl.arange(0, block_m)
    query_mask = query_offsets < extend_length
    dim_offsets = tl.arange(0, block_d)
    dim_mask = dim_offsets < head_dim
    q = tl.load(
        q_ptr
        + (query_start + query_offsets)[:, None] * stride_q_token
        + head_id * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=query_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    running_max = tl.full((block_m,), float("-inf"), dtype=tl.float32)
    normalizer = tl.zeros((block_m,), dtype=tl.float32)
    accumulator = tl.zeros((block_m, block_d), dtype=tl.float32)
    for key_start in range(0, prefix_length, block_n):
        key_offsets = key_start + tl.arange(0, block_n)
        key_valid = key_offsets < prefix_length
        attention_mask = key_valid[None, :] & query_mask[:, None]
        gathered = prefix_start + key_offsets
        k = tl.load(
            kb_ptr
            + gathered[None, :] * stride_kb_token
            + kv_head_id * stride_kb_head
            + dim_offsets[:, None] * stride_kb_dim,
            mask=dim_mask[:, None] & key_valid[None, :],
            other=0.0,
        )
        qk = (
            tl.dot(
                q.to(tl.float32),
                k.to(tl.float32),
                out_dtype=tl.float32,
                input_precision="ieee",
            )
            * sm_scale
        )
        qk = tl.where(attention_mask, qk, float("-inf"))
        next_max = tl.maximum(running_max, tl.max(qk, axis=1))
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        old_scale = tl.exp(running_max - safe_max)
        probabilities = tl.exp(qk - safe_max[:, None])
        v = tl.load(
            vb_ptr
            + gathered[:, None] * stride_vb_token
            + kv_head_id * stride_vb_head
            + dim_offsets[None, :] * stride_vb_dim,
            mask=key_valid[:, None] & dim_mask[None, :],
            other=0.0,
        )
        rescaled_normalizer = normalizer * old_scale
        next_normalizer = rescaled_normalizer + tl.sum(probabilities, axis=1)
        safe_normalizer = tl.maximum(next_normalizer, 1.0)
        accumulator = (
            accumulator * (rescaled_normalizer / safe_normalizer)[:, None]
        )
        normalized_probabilities = probabilities / safe_normalizer[:, None]
        accumulator += tl.dot(
            normalized_probabilities.to(tl.float32),
            v.to(tl.float32),
            out_dtype=tl.float32,
            input_precision="ieee",
        )
        normalizer = next_normalizer
        running_max = next_max
    extend_limit = tl.minimum((query_block + 1) * block_m, extend_length)
    for key_start in range(0, extend_limit, block_n):
        key_offsets = key_start + tl.arange(0, block_n)
        key_valid = key_offsets < extend_length
        attention_mask = (
            key_valid[None, :]
            & query_mask[:, None]
            & (key_offsets[None, :] <= query_offsets[:, None])
        )
        k = tl.load(
            ke_ptr
            + (query_start + key_offsets)[None, :] * stride_ke_token
            + kv_head_id * stride_ke_head
            + dim_offsets[:, None] * stride_ke_dim,
            mask=dim_mask[:, None] & key_valid[None, :],
            other=0.0,
        )
        qk = (
            tl.dot(
                q.to(tl.float32),
                k.to(tl.float32),
                out_dtype=tl.float32,
                input_precision="ieee",
            )
            * sm_scale
        )
        qk = tl.where(attention_mask, qk, float("-inf"))
        next_max = tl.maximum(running_max, tl.max(qk, axis=1))
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        old_scale = tl.exp(running_max - safe_max)
        probabilities = tl.exp(qk - safe_max[:, None])
        v = tl.load(
            ve_ptr
            + (query_start + key_offsets)[:, None] * stride_ve_token
            + kv_head_id * stride_ve_head
            + dim_offsets[None, :] * stride_ve_dim,
            mask=key_valid[:, None] & dim_mask[None, :],
            other=0.0,
        )
        rescaled_normalizer = normalizer * old_scale
        next_normalizer = rescaled_normalizer + tl.sum(probabilities, axis=1)
        safe_normalizer = tl.maximum(next_normalizer, 1.0)
        accumulator = (
            accumulator * (rescaled_normalizer / safe_normalizer)[:, None]
        )
        normalized_probabilities = probabilities / safe_normalizer[:, None]
        accumulator += tl.dot(
            normalized_probabilities.to(tl.float32),
            v.to(tl.float32),
            out_dtype=tl.float32,
            input_precision="ieee",
        )
        normalizer = next_normalizer
        running_max = next_max
    output = accumulator
    tl.store(
        out_ptr
        + (query_start + query_offsets)[:, None] * stride_out_token
        + head_id * stride_out_head
        + dim_offsets[None, :] * stride_out_dim,
        output,
        mask=query_mask[:, None] & dim_mask[None, :],
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
    output = torch.zeros_like(q_extend, dtype=torch.float32)
    if batch_size <= 0 or total_extend == 0:
        return output
    device = q_extend.device
    qo = _prepare_index(qo_indptr, device)
    kvp = _prepare_index(kv_indptr, device)
    kvi = _prepare_index(kv_indices, device)
    lengths = qo[1:] - qo[:-1]
    max_length = max(int(lengths.max().item()), int(max_len_extend), 1)
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_m = 16
    block_n = _BLOCK_N
    total_prefix = int(kvi.numel())
    k_gathered = torch.empty(
        (max(total_prefix, 1), num_kv_heads, head_dim),
        dtype=k_buffer.dtype,
        device=device,
    )
    v_gathered = torch.empty(
        k_gathered.shape, dtype=v_buffer.dtype, device=device
    )
    if total_prefix > 0:
        _extend_attention_gather_kernel[
            triton.cdiv(total_prefix, _BLOCK_R), num_kv_heads
        ](
            k_buffer,
            v_buffer,
            k_gathered,
            v_gathered,
            kvi,
            total_prefix,
            head_dim,
            k_buffer.stride(0),
            k_buffer.stride(1),
            k_buffer.stride(2),
            v_buffer.stride(0),
            v_buffer.stride(1),
            v_buffer.stride(2),
            k_gathered.stride(0),
            k_gathered.stride(1),
            k_gathered.stride(2),
            block_r=_BLOCK_R,
            block_d=block_d,
            num_warps=4,
            num_stages=1,
            buffer_tokens=k_buffer.shape[0],
        )
    num_query_blocks = triton.cdiv(max_length, block_m)
    total_programs = num_query_blocks * num_heads * batch_size
    for pid_base in range(0, total_programs, _MAX_PROGRAMS):
        launch_programs = min(_MAX_PROGRAMS, total_programs - pid_base)
        _extend_attention_kernel[launch_programs,](
            q_extend,
            k_extend,
            v_extend,
            k_gathered,
            v_gathered,
            output,
            qo,
            kvp,
            kvi,
            1.0 / math.sqrt(head_dim),
            num_heads,
            group_size,
            head_dim,
            q_extend.stride(0),
            q_extend.stride(1),
            q_extend.stride(2),
            k_extend.stride(0),
            k_extend.stride(1),
            k_extend.stride(2),
            v_extend.stride(0),
            v_extend.stride(1),
            v_extend.stride(2),
            k_gathered.stride(0),
            k_gathered.stride(1),
            k_gathered.stride(2),
            v_gathered.stride(0),
            v_gathered.stride(1),
            v_gathered.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            pid_base,
            num_query_blocks,
            block_m=block_m,
            block_n=block_n,
            block_d=block_d,
            num_warps=4,
            num_stages=1,
        )
    return output


__all__ = ["extend_attention"]
