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
def bf16__load_complete_tile(pointer, mask, complete):
    if complete:
        value = tl.load(pointer)
    else:
        value = tl.load(pointer, mask=mask, other=0.0)
    return value


@triton.jit
def bf16__pv_native_value(p, v):
    if v.dtype == tl.bfloat16:
        p = p * 65536.0
        ph = p.to(tl.bfloat16)
        pr = p - ph.to(tl.float32)
        pm = pr.to(tl.bfloat16)
        pl = (pr - pm.to(tl.float32)).to(tl.bfloat16)
        vh = (v.to(tl.float32) * 1.52587890625e-05).to(tl.bfloat16)
        acc = tl.dot(pl, vh, out_dtype=tl.float32)
        acc = tl.dot(pm, vh, acc, out_dtype=tl.float32)
        return tl.dot(ph, vh, acc, out_dtype=tl.float32)
    else:
        return tl.dot(p, v.to(tl.float32), input_precision="ieee")


bf16__BLOCK_M = 64
bf16__BLOCK_N = 64
bf16__MAX_PROGRAMS = 8192
bf16__BLOCK_R = 64


@triton.jit
def bf16__extend_attention_kernel(
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
    buffer_tokens: tl.constexpr,
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
        q = bf16__load_complete_tile(
            q_ptr
            + (query_start + query_offsets)[:, None] * stride_q_token
            + head_id * stride_q_head
            + dim_offsets[None, :] * stride_q_dim,
            query_mask[:, None] & dim_mask[None, :],
            (query_block * block_m + block_m <= extend_length)
            & (head_dim == block_d),
        )
        running_max = tl.full((block_m,), float("-inf"), dtype=tl.float32)
        normalizer = tl.zeros((block_m,), dtype=tl.float32)
        accumulator = tl.zeros((block_m, block_d), dtype=tl.float32)
        for key_start in range(0, prefix_length, block_n):
            key_offsets = key_start + tl.arange(0, block_n)
            key_valid = key_offsets < prefix_length
            attention_mask = key_valid[None, :] & query_mask[:, None]
            gathered = tl.load(
                kvi_ptr + prefix_start + key_offsets, mask=key_valid, other=0
            ).to(tl.int64)
            gathered = tl.where(
                gathered < 0, gathered + buffer_tokens, gathered
            )
            k = bf16__load_complete_tile(
                kb_ptr
                + gathered[None, :] * stride_kb_token
                + kv_head_id * stride_kb_head
                + dim_offsets[:, None] * stride_kb_dim,
                dim_mask[:, None] & key_valid[None, :],
                (key_start + block_n <= prefix_length) & (head_dim == block_d),
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
            v = bf16__load_complete_tile(
                vb_ptr
                + gathered[:, None] * stride_vb_token
                + kv_head_id * stride_vb_head
                + dim_offsets[None, :] * stride_vb_dim,
                key_valid[:, None] & dim_mask[None, :],
                (key_start + block_n <= prefix_length) & (head_dim == block_d),
            )
            rescaled_normalizer = normalizer * old_scale
            next_normalizer = rescaled_normalizer + tl.sum(
                probabilities, axis=1
            )
            safe_normalizer = tl.maximum(next_normalizer, 1.0)
            accumulator = (
                accumulator * (rescaled_normalizer / safe_normalizer)[:, None]
            )
            normalized_probabilities = tl.div_rn(
                probabilities, safe_normalizer[:, None]
            )
            accumulator += bf16__pv_native_value(
                normalized_probabilities.to(tl.float32), v
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
            k = bf16__load_complete_tile(
                ke_ptr
                + (query_start + key_offsets)[None, :] * stride_ke_token
                + kv_head_id * stride_ke_head
                + dim_offsets[:, None] * stride_ke_dim,
                dim_mask[:, None] & key_valid[None, :],
                (key_start + block_n <= extend_length) & (head_dim == block_d),
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
            v = bf16__load_complete_tile(
                ve_ptr
                + (query_start + key_offsets)[:, None] * stride_ve_token
                + kv_head_id * stride_ve_head
                + dim_offsets[None, :] * stride_ve_dim,
                key_valid[:, None] & dim_mask[None, :],
                (key_start + block_n <= extend_length) & (head_dim == block_d),
            )
            rescaled_normalizer = normalizer * old_scale
            next_normalizer = rescaled_normalizer + tl.sum(
                probabilities, axis=1
            )
            safe_normalizer = tl.maximum(next_normalizer, 1.0)
            accumulator = (
                accumulator * (rescaled_normalizer / safe_normalizer)[:, None]
            )
            normalized_probabilities = tl.div_rn(
                probabilities, safe_normalizer[:, None]
            )
            accumulator += bf16__pv_native_value(
                normalized_probabilities.to(tl.float32), v
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


def bf16__prepare_index(tensor, device):
    return tensor.to(device=device).contiguous()


def bf16_extend_attention(
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
    output = torch.empty_like(q_extend, dtype=torch.float32)
    if batch_size <= 0 or total_extend == 0:
        return output
    device = q_extend.device
    qo = bf16__prepare_index(qo_indptr, device)
    kvp = bf16__prepare_index(kv_indptr, device)
    kvi = bf16__prepare_index(kv_indices, device)
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_m = 64
    block_n = bf16__BLOCK_N
    num_query_blocks = min(
        triton.cdiv(total_extend, block_m),
        max(1, triton.cdiv(4096, batch_size * num_heads)),
    )
    total_programs = num_query_blocks * num_heads * batch_size
    for pid_base in range(0, total_programs, bf16__MAX_PROGRAMS):
        launch_programs = min(bf16__MAX_PROGRAMS, total_programs - pid_base)
        bf16__extend_attention_kernel[launch_programs,](
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
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
            k_buffer.stride(0),
            k_buffer.stride(1),
            k_buffer.stride(2),
            v_buffer.stride(0),
            v_buffer.stride(1),
            v_buffer.stride(2),
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
            buffer_tokens=k_buffer.shape[0],
        )
    return output


@triton.jit
def rows__attention_fp32_rows(
    Q,
    KE,
    VE,
    KB,
    VB,
    QO,
    KP,
    KI,
    O,
    HQ: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SQ2: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SV0: tl.constexpr,
    SV1: tl.constexpr,
    SV2: tl.constexpr,
    SBK0: tl.constexpr,
    SBK1: tl.constexpr,
    SBK2: tl.constexpr,
    SBV0: tl.constexpr,
    SBV1: tl.constexpr,
    SBV2: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    ROWS: tl.constexpr,
    buffer_tokens: tl.constexpr,
):
    row_start = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    start = tl.load(QO + batch)
    end = tl.load(QO + batch + 1)
    prefix_start = tl.load(KP + batch)
    prefix_end = tl.load(KP + batch + 1)
    prefix_len = prefix_end - prefix_start
    kv_head = head // GROUP
    ds = tl.arange(0, BD)
    ns = tl.arange(0, BN)
    for row in range(row_start, end - start, ROWS):
        q = tl.load(
            Q + (start + row) * SQ0 + head * SQ1 + ds * SQ2, ds < D, 0
        ).to(tl.float32)
        m = tl.full((), float("-inf"), tl.float32)
        normalizer = tl.full((), 0.0, tl.float32)
        acc = tl.full((BD,), 0.0, tl.float32)
        for base in range(0, prefix_len, BN):
            keys = base + ns
            valid = keys < prefix_len
            ids = tl.load(KI + prefix_start + keys, valid, 0).to(tl.int64)
            ids = tl.where(ids < 0, ids + buffer_tokens, ids)
            k = tl.load(
                KB + ids[:, None] * SBK0 + kv_head * SBK1 + ds[None, :] * SBK2,
                valid[:, None] & (ds[None, :] < D),
                0,
            ).to(tl.float32)
            s = tl.sum(k * q[None, :], 1) * D ** (-0.5)
            s = tl.where(valid, s, float("-inf"))
            next_m = tl.maximum(m, tl.max(s, 0))
            alpha = tl.exp(m - next_m)
            p = tl.exp(s - next_m)
            old_l = normalizer * alpha
            next_l = old_l + tl.sum(p, 0)
            acc = acc * (old_l / next_l)
            weights = p / next_l
            for j in range(BN):
                probability_j = tl.sum(tl.where(ns == j, weights, 0.0), 0)
                valid_j = base + j < prefix_len
                id_j = tl.load(KI + prefix_start + base + j, valid_j, 0).to(
                    tl.int64
                )
                id_j = tl.where(id_j < 0, id_j + buffer_tokens, id_j)
                value_j = tl.load(
                    VB + id_j * SBV0 + kv_head * SBV1 + ds * SBV2,
                    valid_j & (ds < D),
                    0,
                ).to(tl.float32)
                acc = tl.fma(probability_j, value_j, acc)
            normalizer = next_l
            m = next_m
        for base in range(0, row + 1, BN):
            keys = base + ns
            valid = keys <= row
            k = tl.load(
                KE
                + (start + keys)[:, None] * SK0
                + kv_head * SK1
                + ds[None, :] * SK2,
                valid[:, None] & (ds[None, :] < D),
                0,
            ).to(tl.float32)
            s = tl.sum(k * q[None, :], 1) * D ** (-0.5)
            s = tl.where(valid, s, float("-inf"))
            next_m = tl.maximum(m, tl.max(s, 0))
            alpha = tl.exp(m - next_m)
            p = tl.exp(s - next_m)
            old_l = normalizer * alpha
            next_l = old_l + tl.sum(p, 0)
            acc = acc * (old_l / next_l)
            weights = p / next_l
            for j in range(BN):
                probability_j = tl.sum(tl.where(ns == j, weights, 0.0), 0)
                key_j = base + j
                value_j = tl.load(
                    VE + (start + key_j) * SV0 + kv_head * SV1 + ds * SV2,
                    (key_j <= row) & (ds < D),
                    0,
                ).to(tl.float32)
                acc = tl.fma(probability_j, value_j, acc)
            normalizer = next_l
            m = next_m
        tl.store(O + ((start + row) * HQ + head) * D + ds, acc, ds < D)


def rows_extend_attention(
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
    e, hq, d = q_extend.shape
    hkv = k_extend.shape[1]
    batch = qo_indptr.numel() - 1
    device = q_extend.device
    out = torch.empty((e, hq, d), dtype=torch.float32, device=device)
    if e == 0 or batch <= 0:
        return out
    qo = qo_indptr.to(device=device).contiguous()
    kp = kv_indptr.to(device=device).contiguous()
    ki = kv_indices.to(device=device).contiguous()
    max_extend = min(e, max(1, triton.cdiv(4096, batch * hq)))
    rows__attention_fp32_rows[max_extend, hq, batch](
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo,
        kp,
        ki,
        out,
        hq,
        hq // hkv,
        d,
        *q_extend.stride(),
        *k_extend.stride(),
        *v_extend.stride(),
        *k_buffer.stride(),
        *v_buffer.stride(),
        BD=triton.next_power_of_2(d),
        BN=32,
        num_warps=4,
        num_stages=1,
        ROWS=max_extend,
        buffer_tokens=k_buffer.shape[0],
    )
    return out


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
    if all(
        (
            t.dtype == torch.bfloat16
            for t in (q_extend, k_extend, v_extend, k_buffer, v_buffer)
        )
    ):
        return bf16_extend_attention(
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            max_len_extend,
        )
    return rows_extend_attention(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        max_len_extend,
    )


__all__ = ["extend_attention"]
