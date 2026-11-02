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

"""extend_attention for RDNA3 (gfx1100) — the prefill stage of paged
attention, where new tokens attend to a cached prefix and causally to
themselves (Triton).

For a small query group the packed kernel below reads the prefix straight
through ``kv_indices`` and holds several tokens of every query head of a group
in one tile, so each K/V tile is read once a group instead of once a head, and
no copy of the prefix has to be staged first. Its products run on the matrix
units for 16-bit inputs and, since gfx1100 has no float32 matrix instruction,
in IEEE mode on the FMA units for float32, which also wants a narrower tile.
The other cases keep the gather pass and the kernel after it.

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator (no
``F.scaled_dot_product_attention``, no ``_compiled`` handles, no
``torch.ops.*`` attention) — the whole computation is done inside the Triton
kernel.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_qk(q, k):
    if (
        q.dtype == tl.float16
        and k.dtype == tl.float16
        or (q.dtype == tl.bfloat16 and k.dtype == tl.bfloat16)
    ):
        return tl.dot(q, k, out_dtype=tl.float32)
    else:
        return tl.dot(
            q.to(tl.float32),
            k.to(tl.float32),
            out_dtype=tl.float32,
            input_precision="ieee",
        )


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
            qk = _attention_qk(q, k) * sm_scale
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
            accumulator = accumulator * old_scale[:, None]
            accumulator += tl.dot(
                probabilities.to(vb_ptr.dtype.element_ty),
                v,
                out_dtype=tl.float32,
                input_precision="ieee",
            )
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=1)
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
            qk = _attention_qk(q, k) * sm_scale
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
            accumulator = accumulator * old_scale[:, None]
            accumulator += tl.dot(
                probabilities.to(ve_ptr.dtype.element_ty),
                v,
                out_dtype=tl.float32,
                input_precision="ieee",
            )
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=1)
            running_max = next_max
        denominator = tl.maximum(normalizer, 1.0)
        output = accumulator / denominator[:, None]
        tl.store(
            out_ptr
            + (query_start + query_offsets)[:, None] * stride_out_token
            + head_id * stride_out_head
            + dim_offsets[None, :] * stride_out_dim,
            output,
            mask=query_mask[:, None] & dim_mask[None, :],
        )


_LOG2E = 1.4426950408889634
# 16-bit products run on the matrix units, which a 64-row tile keeps fed;
# float32 products run on the FMA units, where the registers a tile that wide
# needs cost more than the reuse it buys.
_PACKED_BLOCK_M = 64
_PACKED_BLOCK_M_FP32 = 32
_PACKED_BLOCK_N = 32
_PACKED_WARPS = 4
_PACKED_WARPS_FP32 = 8
_PACKED_MAX_GROUP = 8


@triton.jit
def _packed_block(
    acc,
    l_i,
    m_i,
    q,
    k_base,
    v_base,
    rows,
    stride_t,
    dim_offsets,
    dim_mask,
    key_valid,
    allow,
    qk_scale,
    IEEE: tl.constexpr,
):
    kv_mask = key_valid[:, None] & dim_mask[None, :]
    k = tl.load(
        k_base + rows[None, :] * stride_t + dim_offsets[:, None],
        mask=tl.trans(kv_mask),
        other=0.0,
    )
    if IEEE:
        qk = tl.dot(q, k, input_precision="ieee") * qk_scale
    else:
        qk = tl.dot(q, k) * qk_scale
    qk = tl.where(allow, qk, -1.0e30)
    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
    p = tl.math.exp2(qk - m_new[:, None])
    alpha = tl.math.exp2(m_i - m_new)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    v = tl.load(
        v_base + rows[:, None] * stride_t + dim_offsets[None, :],
        mask=kv_mask,
        other=0.0,
    )
    if IEEE:
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
    else:
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return acc, l_i, m_new


@triton.jit
def _extend_attention_packed_kernel(
    q_ptr,
    ke_ptr,
    ve_ptr,
    kb_ptr,
    vb_ptr,
    out_ptr,
    qo_ptr,
    kvp_ptr,
    kvi_ptr,
    qk_scale,
    grid_m,
    buffer_tokens,
    stride_q_token,
    stride_q_head,
    stride_ke_token,
    stride_ke_head,
    stride_kb_token,
    stride_kb_head,
    stride_out_token,
    stride_out_head,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    IEEE: tl.constexpr,
):
    # row r of a tile is token r // group_size and query head r % group_size of the group, so every K/V
    # tile is loaded once for the whole group
    pid_m = tl.program_id(0)
    kv_head = tl.program_id(1)
    seq = tl.program_id(2)
    tokens: tl.constexpr = block_m // group_size
    r = tl.arange(0, block_m)
    r_token = r // group_size
    head = kv_head * group_size + r % group_size
    row_ok = r < tokens * group_size
    q_start = tl.load(qo_ptr + seq)
    extend_length = tl.load(qo_ptr + seq + 1) - q_start
    prefix_start = tl.load(kvp_ptr + seq)
    prefix_length = tl.load(kvp_ptr + seq + 1) - prefix_start
    num_m = tl.cdiv(extend_length, tokens)
    dim_offsets = tl.arange(0, block_d)
    dim_mask = dim_offsets < head_dim
    key_offsets = tl.arange(0, block_n)
    kb_base = kb_ptr + kv_head * stride_kb_head
    vb_base = vb_ptr + kv_head * stride_kb_head
    ke_base = ke_ptr + q_start * stride_ke_token + kv_head * stride_ke_head
    ve_base = ve_ptr + q_start * stride_ke_token + kv_head * stride_ke_head
    # every program walks the query blocks pid_m, pid_m + grid_m, ... from the longest causal range down
    for it in range(pid_m, num_m, grid_m):
        m_block = num_m - 1 - it
        token = m_block * tokens + r_token
        query_mask = row_ok & (token < extend_length)
        q_rows = q_start + token
        q = tl.load(
            q_ptr
            + q_rows[:, None] * stride_q_token
            + head[:, None] * stride_q_head
            + dim_offsets[None, :],
            mask=query_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        m_i = tl.full([block_m], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([block_m], dtype=tl.float32)
        acc = tl.zeros([block_m, block_d], dtype=tl.float32)
        for key_start in range(0, prefix_length, block_n):
            key_valid = key_start + key_offsets < prefix_length
            rows = tl.load(
                kvi_ptr + prefix_start + key_start + key_offsets,
                mask=key_valid,
                other=0,
            )
            rows = tl.where(rows < 0, rows + buffer_tokens, rows)
            acc, l_i, m_i = _packed_block(
                acc,
                l_i,
                m_i,
                q,
                kb_base,
                vb_base,
                rows,
                stride_kb_token,
                dim_offsets,
                dim_mask,
                key_valid,
                key_valid[None, :],
                qk_scale,
                IEEE,
            )
        extend_limit = tl.minimum(extend_length, (m_block + 1) * tokens)
        for key_start in range(0, extend_limit, block_n):
            keys = key_start + key_offsets
            key_valid = keys < extend_length
            allow = key_valid[None, :] & (keys[None, :] <= token[:, None])
            acc, l_i, m_i = _packed_block(
                acc,
                l_i,
                m_i,
                q,
                ke_base,
                ve_base,
                keys,
                stride_ke_token,
                dim_offsets,
                dim_mask,
                key_valid,
                allow,
                qk_scale,
                IEEE,
            )
        tl.store(
            out_ptr
            + q_rows[:, None] * stride_out_token
            + head[:, None] * stride_out_head
            + dim_offsets[None, :],
            acc / l_i[:, None],
            mask=query_mask[:, None] & dim_mask[None, :],
        )


def _packed_supported(tensors, head_dim, group_size, batch_size):
    # the packed kernel uses 32-bit offsets and needs the K and V layouts to match
    q_extend, k_extend, v_extend, k_buffer, v_buffer = tensors
    return (
        q_extend.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and all(t.dtype == q_extend.dtype for t in tensors)
        and all(t.stride(-1) == 1 for t in tensors)
        and all(t.numel() < 2**31 for t in tensors)
        and k_extend.stride() == v_extend.stride()
        and k_buffer.stride() == v_buffer.stride()
        and v_extend.shape[-1] == head_dim
        and head_dim <= 128
        and group_size <= _PACKED_MAX_GROUP
        and batch_size <= 65535
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
    total_extend, num_heads, head_dim = q_extend.shape
    num_kv_heads = k_extend.shape[1]
    group_size = num_heads // num_kv_heads
    batch_size = qo_indptr.numel() - 1
    if batch_size <= 0 or total_extend == 0:
        return torch.zeros_like(q_extend, dtype=torch.float32)
    device = q_extend.device
    qo = _prepare_index(qo_indptr, device)
    kvp = _prepare_index(kv_indptr, device)
    kvi = _prepare_index(kv_indices, device)
    block_d = max(16, triton.next_power_of_2(head_dim))
    # W7900D (gfx1100): the gather pass below copies the prefix at a fraction of the memory bandwidth and
    # every query head reloads its K/V tiles; the packed kernel reads the prefix through kv_indices and
    # loads each K/V tile once for all query heads of a group
    tensors = (q_extend, k_extend, v_extend, k_buffer, v_buffer)
    if _packed_supported(tensors, head_dim, group_size, batch_size):
        # every token of every head is written below, and at this size zeroing
        # the float32 output first costs as much as a pass over the KV cache
        output = torch.empty_like(q_extend, dtype=torch.float32)
        ieee = q_extend.dtype == torch.float32
        block_m = _PACKED_BLOCK_M_FP32 if ieee else _PACKED_BLOCK_M
        tokens = block_m // group_size
        grid_m = max(1, triton.cdiv(int(max_len_extend), tokens))
        _extend_attention_packed_kernel[(grid_m, num_kv_heads, batch_size)](
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            output,
            qo,
            kvp,
            kvi,
            _LOG2E / math.sqrt(head_dim),
            grid_m,
            k_buffer.shape[0],
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            k_extend.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            output.stride(0),
            output.stride(1),
            group_size=group_size,
            head_dim=head_dim,
            block_m=block_m,
            block_n=_PACKED_BLOCK_N,
            block_d=block_d,
            IEEE=ieee,
            num_warps=_PACKED_WARPS_FP32 if ieee else _PACKED_WARPS,
            num_stages=1,
        )
        return output
    output = torch.zeros_like(q_extend, dtype=torch.float32)
    block_m = 32 if q_extend.dtype == torch.float32 else 64
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
