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


@triton.jit
def _flat_tensor_copy(
    X,
    Y,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token = offsets // (H * D)
    head = offsets // D % H
    dim = offsets % D
    value = tl.load(X + token * S0 + head * S1 + dim * S2, offsets < N, 0)
    tl.store(Y + offsets, value, offsets < N)


def _contiguous_tensor(tensor):
    if tensor.is_contiguous() or tensor.numel() == 0:
        return tensor
    output = torch.empty(
        tensor.shape, device=tensor.device, dtype=tensor.dtype
    )
    _flat_tensor_copy[triton.cdiv(tensor.numel(), 512),](
        tensor,
        output,
        tensor.numel(),
        tensor.shape[1],
        tensor.shape[2],
        *tensor.stride(),
        BLOCK=512,
        num_warps=4,
        num_stages=1,
    )
    return output


@triton.jit
def _flat_prefix_gather(
    K,
    V,
    KG,
    VG,
    I,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    BUFFER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // D
    dim = offsets % D
    head = tl.program_id(1)
    valid = row < ROWS
    token = tl.load(I + row, valid, 0).to(tl.int64)
    token = tl.where(token < 0, token + BUFFER, token)
    k = tl.load(K + token * KS0 + head * KS1 + dim * KS2, valid, 0)
    v = tl.load(V + token * VS0 + head * VS1 + dim * VS2, valid, 0)
    dest = (row * H + head) * D + dim
    tl.store(KG + dest, k, valid)
    tl.store(VG + dest, v, valid)


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
    head_dim: tl.constexpr,
    stride_kb_token: tl.constexpr,
    stride_kb_head: tl.constexpr,
    stride_kb_dim: tl.constexpr,
    stride_vb_token: tl.constexpr,
    stride_vb_head: tl.constexpr,
    stride_vb_dim: tl.constexpr,
    stride_kg_token: tl.constexpr,
    stride_kg_head: tl.constexpr,
    stride_kg_dim: tl.constexpr,
    block_r: tl.constexpr,
    block_d: tl.constexpr,
    buffer_tokens: tl.constexpr,
):
    row_block = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    rows = row_block * block_r + tl.arange(0, block_r)
    row_mask = rows < total_prefix
    token_ids = tl.load(kvi_ptr + rows, mask=row_mask, other=0).to(tl.int64)
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
    num_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_ke_token: tl.constexpr,
    stride_ke_head: tl.constexpr,
    stride_ke_dim: tl.constexpr,
    stride_ve_token: tl.constexpr,
    stride_ve_head: tl.constexpr,
    stride_ve_dim: tl.constexpr,
    stride_kb_token: tl.constexpr,
    stride_kb_head: tl.constexpr,
    stride_kb_dim: tl.constexpr,
    stride_vb_token: tl.constexpr,
    stride_vb_head: tl.constexpr,
    stride_vb_dim: tl.constexpr,
    stride_out_token: tl.constexpr,
    stride_out_head: tl.constexpr,
    stride_out_dim: tl.constexpr,
    pid_base,
    num_query_blocks: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    task = pid_base + tl.program_id(0)
    query_block_start = task % num_query_blocks
    head_task = task // num_query_blocks
    head_id = head_task % num_heads
    sequence_id = head_task // num_heads
    kv_head_id = head_id // group_size
    query_start = tl.load(qo_ptr + sequence_id)
    query_end = tl.load(qo_ptr + sequence_id + 1)
    extend_length = query_end - query_start
    prefix_start = tl.load(kvp_ptr + sequence_id)
    prefix_end = tl.load(kvp_ptr + sequence_id + 1)
    prefix_length = prefix_end - prefix_start
    for query_block in range(
        query_block_start, tl.cdiv(extend_length, block_m), num_query_blocks
    ):
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
        extend_limit = tl.minimum((query_block + 1) * block_m, extend_length)
        for key_start in range(0, prefix_length + extend_limit, block_n):
            key_offsets = key_start + tl.arange(0, block_n)
            prefix_valid = key_offsets < prefix_length
            extend_offsets = key_offsets - prefix_length
            extend_valid = (extend_offsets >= 0) & (
                extend_offsets < extend_length
            )
            key_valid = key_offsets < prefix_length + extend_length
            attention_mask = (
                key_valid[None, :]
                & query_mask[:, None]
                & (
                    key_offsets[None, :]
                    <= prefix_length + query_offsets[:, None]
                )
            )
            gathered = prefix_start + key_offsets
            kp = tl.load(
                kb_ptr
                + gathered[None, :] * stride_kb_token
                + kv_head_id * stride_kb_head
                + dim_offsets[:, None] * stride_kb_dim,
                mask=dim_mask[:, None] & prefix_valid[None, :],
                other=0.0,
            )
            ke = tl.load(
                ke_ptr
                + (query_start + extend_offsets)[None, :] * stride_ke_token
                + kv_head_id * stride_ke_head
                + dim_offsets[:, None] * stride_ke_dim,
                mask=dim_mask[:, None] & extend_valid[None, :],
                other=0.0,
            )
            if kp.dtype == ke.dtype:
                k = tl.where(prefix_valid[None, :], kp, ke)
            else:
                k = tl.where(
                    prefix_valid[None, :], kp.to(tl.float32), ke.to(tl.float32)
                )
            if not (
                q.dtype == tl.float16
                and k.dtype == tl.float16
                or (q.dtype == tl.bfloat16 and k.dtype == tl.bfloat16)
            ):
                qk = (
                    tl.dot(
                        q.to(tl.float32),
                        k.to(tl.float32),
                        out_dtype=tl.float32,
                        input_precision="ieee",
                    )
                    * sm_scale
                )
            else:
                qk = tl.dot(q, k, out_dtype=tl.float32) * sm_scale
            qk = tl.where(attention_mask, qk, float("-inf"))
            next_max = tl.maximum(running_max, tl.max(qk, axis=1))
            safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
            old_scale = tl.exp(running_max - safe_max)
            probabilities = tl.exp(qk - safe_max[:, None])
            rescaled_normalizer = normalizer * old_scale
            next_normalizer = rescaled_normalizer + tl.sum(
                probabilities, axis=1
            )
            normalizer = next_normalizer
            running_max = next_max
        denominator = tl.maximum(normalizer, 1.0)
        accumulator = tl.zeros((block_m, block_d), dtype=tl.float32)
        for key_start in range(0, prefix_length + extend_limit, block_n):
            key_offsets = key_start + tl.arange(0, block_n)
            prefix_valid = key_offsets < prefix_length
            extend_offsets = key_offsets - prefix_length
            extend_valid = (extend_offsets >= 0) & (
                extend_offsets < extend_length
            )
            key_valid = key_offsets < prefix_length + extend_length
            attention_mask = (
                key_valid[None, :]
                & query_mask[:, None]
                & (
                    key_offsets[None, :]
                    <= prefix_length + query_offsets[:, None]
                )
            )
            gathered = prefix_start + key_offsets
            kp = tl.load(
                kb_ptr
                + gathered[None, :] * stride_kb_token
                + kv_head_id * stride_kb_head
                + dim_offsets[:, None] * stride_kb_dim,
                mask=dim_mask[:, None] & prefix_valid[None, :],
                other=0.0,
            )
            ke = tl.load(
                ke_ptr
                + (query_start + extend_offsets)[None, :] * stride_ke_token
                + kv_head_id * stride_ke_head
                + dim_offsets[:, None] * stride_ke_dim,
                mask=dim_mask[:, None] & extend_valid[None, :],
                other=0.0,
            )
            if kp.dtype == ke.dtype:
                k = tl.where(prefix_valid[None, :], kp, ke)
            else:
                k = tl.where(
                    prefix_valid[None, :], kp.to(tl.float32), ke.to(tl.float32)
                )
            if not (
                q.dtype == tl.float16
                and k.dtype == tl.float16
                or (q.dtype == tl.bfloat16 and k.dtype == tl.bfloat16)
            ):
                qk = (
                    tl.dot(
                        q.to(tl.float32),
                        k.to(tl.float32),
                        out_dtype=tl.float32,
                        input_precision="ieee",
                    )
                    * sm_scale
                )
            else:
                qk = tl.dot(q, k, out_dtype=tl.float32) * sm_scale
            qk = tl.where(attention_mask, qk, float("-inf"))
            vp = tl.load(
                vb_ptr
                + gathered[:, None] * stride_vb_token
                + kv_head_id * stride_vb_head
                + dim_offsets[None, :] * stride_vb_dim,
                mask=prefix_valid[:, None] & dim_mask[None, :],
                other=0.0,
            )
            ve = tl.load(
                ve_ptr
                + (query_start + extend_offsets)[:, None] * stride_ve_token
                + kv_head_id * stride_ve_head
                + dim_offsets[None, :] * stride_ve_dim,
                mask=extend_valid[:, None] & dim_mask[None, :],
                other=0.0,
            )
            v = tl.where(
                prefix_valid[:, None], vp.to(tl.float32), ve.to(tl.float32)
            )
            probabilities = tl.exp(qk - running_max[:, None])
            normalized_probabilities = probabilities / denominator[:, None]
            accumulator += tl.dot(
                normalized_probabilities.to(tl.float32),
                v.to(tl.float32),
                out_dtype=tl.float32,
                input_precision="ieee",
            )
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
    return tensor.to(device=device).contiguous()


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
    q_extend = _contiguous_tensor(q_extend)
    k_extend = _contiguous_tensor(k_extend)
    v_extend = _contiguous_tensor(v_extend)
    total_extend, num_heads, head_dim = q_extend.shape
    num_kv_heads = k_extend.shape[1]
    group_size = num_heads // num_kv_heads
    batch_size = qo_indptr.numel() - 1
    output = torch.empty_like(q_extend, dtype=torch.float32)
    if batch_size <= 0 or total_extend == 0:
        return output
    device = q_extend.device
    qo = _prepare_index(qo_indptr, device)
    kvp = _prepare_index(kv_indptr, device)
    kvi = _prepare_index(kv_indices, device)
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_m = (
        64
        if q_extend.dtype == k_extend.dtype == k_buffer.dtype
        and q_extend.dtype in (torch.bfloat16, torch.float16)
        else 16
    )
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
        if not k_buffer.is_contiguous() or not v_buffer.is_contiguous():
            _flat_prefix_gather[
                triton.cdiv(total_prefix * head_dim, 512), num_kv_heads
            ](
                k_buffer,
                v_buffer,
                k_gathered,
                v_gathered,
                kvi,
                total_prefix,
                num_kv_heads,
                head_dim,
                *k_buffer.stride(),
                *v_buffer.stride(),
                k_buffer.shape[0],
                BLOCK=512,
                num_warps=4,
                num_stages=1,
            )
        else:
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
    num_query_blocks = min(
        triton.cdiv(total_extend, block_m),
        max(1, triton.cdiv(4096, batch_size * num_heads)),
    )
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
