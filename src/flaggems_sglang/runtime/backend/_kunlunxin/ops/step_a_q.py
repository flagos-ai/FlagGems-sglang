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

"""step_a_q implementation for Kunlunxin."""

import torch
import triton
import triton.language as tl


def _step_a_q_metadata(batch_info):
    return tuple(
        (
            x if x.stride(0) == 1 else x.contiguous()
            for x in (
                batch_info.seg_indptr,
                batch_info.weight_indices,
                batch_info.lora_ranks,
            )
        )
    )


@triton.jit
def _step_a_q_direct_kernel(
    input_ptr,
    weight_ptr,
    out_ptr,
    seg_indptr,
    weight_indices,
    lora_ranks,
    NUM_HEADS: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    INPUT_DIM: tl.constexpr,
    FULL_K_PER_HEAD: tl.constexpr,
    TILES: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    W_STRIDE_0: tl.constexpr,
    W_STRIDE_1: tl.constexpr,
    W_STRIDE_2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid0 = tl.program_id(0)
    h = tl.program_id(1)
    segment = pid0 // TILES
    pid_s = pid0 % TILES
    adapter = tl.load(weight_indices + segment)
    if tl.load(lora_ranks + adapter) == 0:
        return
    segment_start = tl.load(seg_indptr + segment)
    segment_end = tl.load(seg_indptr + segment + 1)
    i0 = segment_start + pid_s * BLOCK_M
    if i0 >= segment_end:
        return
    idx = i0 + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    row_mask = idx < segment_end
    col_mask = offs_n < OUTPUT_DIM
    x = tl.load(
        input_ptr
        + idx[:, None] * X_STRIDE_0
        + h * X_STRIDE_1
        + offs_k[None, :] * X_STRIDE_2,
        mask=row_mask[:, None] & (offs_k < INPUT_DIM)[None, :],
        other=0.0,
    )
    wb = tl.load(
        weight_ptr
        + adapter * W_STRIDE_0
        + (h * FULL_K_PER_HEAD + offs_k[:, None]) * W_STRIDE_1
        + offs_n[None, :] * W_STRIDE_2,
        mask=(offs_k < INPUT_DIM)[:, None] & col_mask[None, :],
        other=0.0,
    )
    acc = tl.dot(x, wb, input_precision="ieee")
    tl.store(
        out_ptr
        + idx[:, None] * (NUM_HEADS * OUTPUT_DIM)
        + h * OUTPUT_DIM
        + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _step_a_q_pack_heads_kernel(
    input_ptr,
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    packed_input_ptr,
    NUM_HEADS: tl.constexpr,
    INPUT_DIM: tl.constexpr,
    PADDED_INPUT_DIM: tl.constexpr,
    PADDED_ROWS: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
    PERM_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    segment = tl.program_id(2)
    segment_start = tl.load(seg_indptr + segment)
    segment_end = tl.load(seg_indptr + segment + 1)
    adapter = tl.load(weight_indices + segment)
    p = tl.minimum(segment_start + m, PERM_SIZE - 1)
    row = tl.load(permutation + p * PERM_STRIDE).to(tl.int32)
    k = tl.arange(0, BLOCK_SIZE)
    x = tl.load(
        input_ptr
        + row * X_STRIDE_0
        + h * X_STRIDE_1
        + tl.minimum(k, INPUT_DIM - 1) * X_STRIDE_2
    )
    x = tl.where(
        (k < INPUT_DIM)
        & (m < segment_end - segment_start)
        & (tl.load(lora_ranks + adapter) > 0),
        x,
        0.0,
    )
    tl.store(
        packed_input_ptr
        + (segment * NUM_HEADS + h) * PADDED_ROWS * PADDED_INPUT_DIM
        + m * PADDED_INPUT_DIM
        + k,
        x,
        k < PADDED_INPUT_DIM,
    )


@triton.jit
def _step_a_q_matmul_heads_kernel(
    packed_input_ptr,
    weight_ptr,
    weight_indices,
    matmul_output_ptr,
    NUM_HEADS: tl.constexpr,
    FULL_K_PER_HEAD: tl.constexpr,
    PADDED_ROWS: tl.constexpr,
    PADDED_INPUT_DIM: tl.constexpr,
    PADDED_OUTPUT_DIM: tl.constexpr,
    W_STRIDE_0: tl.constexpr,
    W_STRIDE_1: tl.constexpr,
    W_STRIDE_2: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    hb = pid // (NUM_M_TILES * NUM_N_TILES)
    h = hb % NUM_HEADS
    segment = hb // NUM_HEADS
    adapter = tl.load(weight_indices + segment)
    m = pid // NUM_N_TILES % NUM_M_TILES * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid % NUM_N_TILES * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    input_block_ptr = (
        packed_input_ptr
        + hb * PADDED_ROWS * PADDED_INPUT_DIM
        + m[:, None] * PADDED_INPUT_DIM
        + k[None, :]
    )
    weight_block_ptr = (
        weight_ptr
        + adapter * W_STRIDE_0
        + (h * FULL_K_PER_HEAD + k[:, None]) * W_STRIDE_1
        + n[None, :] * W_STRIDE_2
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_block in range(PADDED_INPUT_DIM // BLOCK_K):
        a = tl.load(input_block_ptr)
        weight_block = tl.load(weight_block_ptr)
        acc = tl.dot(a, weight_block, acc, input_precision="ieee")
        input_block_ptr += BLOCK_K
        weight_block_ptr += BLOCK_K * W_STRIDE_1
    tl.store(
        matmul_output_ptr
        + hb * PADDED_ROWS * PADDED_OUTPUT_DIM
        + m[:, None] * PADDED_OUTPUT_DIM
        + n[None, :],
        acc,
    )


@triton.jit
def _step_a_q_scatter_heads_kernel(
    matmul_output_ptr,
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    out_ptr,
    NUM_HEADS: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    PADDED_OUTPUT_DIM: tl.constexpr,
    PADDED_ROWS: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
    PERM_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    segment = tl.program_id(2)
    segment_start = tl.load(seg_indptr + segment)
    segment_end = tl.load(seg_indptr + segment + 1)
    adapter = tl.load(weight_indices + segment)
    p = tl.minimum(segment_start + m, PERM_SIZE - 1)
    row = tl.load(permutation + p * PERM_STRIDE).to(tl.int32)
    row = tl.where(m < segment_end - segment_start, row, NUM_TOKENS)
    n = tl.arange(0, BLOCK_SIZE)
    c = tl.load(
        matmul_output_ptr
        + (segment * NUM_HEADS + h) * PADDED_ROWS * PADDED_OUTPUT_DIM
        + m * PADDED_OUTPUT_DIM
        + n
    )
    c = tl.where(tl.load(lora_ranks + adapter) > 0, c, 0.0)
    tl.store(
        out_ptr + (row * NUM_HEADS + h) * OUTPUT_DIM + n, c, n < OUTPUT_DIM
    )


@triton.jit
def _step_a_q_gemv_kernel(
    input_ptr,
    weight_ptr,
    out_ptr,
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    NUM_HEADS: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    INPUT_DIM: tl.constexpr,
    FULL_K_PER_HEAD: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    W_STRIDE_0: tl.constexpr,
    W_STRIDE_1: tl.constexpr,
    W_STRIDE_2: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_PERM: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    segment = tl.program_id(2) // OUTPUT_DIM
    n = tl.program_id(2) % OUTPUT_DIM
    segment_start = tl.load(seg_indptr + segment)
    segment_end = tl.load(seg_indptr + segment + 1)
    if segment_start + m >= segment_end:
        return
    adapter = tl.load(weight_indices + segment)
    if tl.load(lora_ranks + adapter) == 0:
        return
    if HAS_PERM:
        row = tl.load(permutation + (segment_start + m) * PERM_STRIDE).to(
            tl.int32
        )
    else:
        row = segment_start + m
    kk = tl.arange(0, BLOCK_K)
    x = tl.load(
        input_ptr + row * X_STRIDE_0 + h * X_STRIDE_1 + kk * X_STRIDE_2,
        kk < INPUT_DIM,
        0,
    ).to(tl.float32)
    wt = tl.load(
        weight_ptr
        + adapter * W_STRIDE_0
        + (h * FULL_K_PER_HEAD + kk) * W_STRIDE_1
        + n * W_STRIDE_2,
        kk < INPUT_DIM,
        0,
    ).to(tl.float32)
    value = tl.sum(x * wt, 0)
    tl.store(out_ptr + (row * NUM_HEADS + h) * OUTPUT_DIM + n, value)


@triton.jit
def _step_a_q_pack_rows_kernel(
    input_ptr,
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    packed_input_ptr,
    INPUT_DIM: tl.constexpr,
    PADDED_ROWS: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
    PERM_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    m = tl.program_id(0)
    segment = tl.program_id(1)
    segment_start = tl.load(seg_indptr + segment)
    segment_end = tl.load(seg_indptr + segment + 1)
    adapter = tl.load(weight_indices + segment)
    p = tl.minimum(segment_start + m, PERM_SIZE - 1)
    row = tl.load(permutation + p * PERM_STRIDE).to(tl.int32)
    offset = tl.arange(0, BLOCK_SIZE)
    x = tl.load(
        input_ptr
        + row * X_STRIDE_0
        + offset // INPUT_DIM * X_STRIDE_1
        + offset % INPUT_DIM * X_STRIDE_2
    )
    x = tl.where(
        (m < segment_end - segment_start)
        & (tl.load(lora_ranks + adapter) > 0),
        x,
        0.0,
    )
    tl.store(
        packed_input_ptr + (segment * PADDED_ROWS + m) * BLOCK_SIZE + offset, x
    )


@triton.jit
def _step_a_q_matmul_rows_kernel(
    packed_input_ptr,
    weight_ptr,
    weight_indices,
    matmul_output_ptr,
    NUM_HEADS: tl.constexpr,
    FULL_K_PER_HEAD: tl.constexpr,
    PADDED_ROWS: tl.constexpr,
    PADDED_INPUT_DIM: tl.constexpr,
    PADDED_OUTPUT_DIM: tl.constexpr,
    W_STRIDE_0: tl.constexpr,
    W_STRIDE_1: tl.constexpr,
    W_STRIDE_2: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    hb = pid // (NUM_M_TILES * NUM_N_TILES)
    h = hb % NUM_HEADS
    segment = hb // NUM_HEADS
    adapter = tl.load(weight_indices + segment)
    mt = pid // NUM_N_TILES % NUM_M_TILES
    n = pid % NUM_N_TILES * BLOCK_N + tl.arange(0, BLOCK_N)
    m = tl.arange(0, BLOCK_M)
    k = tl.arange(0, BLOCK_K)
    input_block_ptr = (
        packed_input_ptr
        + (segment * PADDED_ROWS + mt * BLOCK_M)
        * (NUM_HEADS * PADDED_INPUT_DIM)
        + h * PADDED_INPUT_DIM
        + m[:, None] * (NUM_HEADS * PADDED_INPUT_DIM)
        + k[None, :]
    )
    weight_block_ptr = (
        weight_ptr
        + adapter * W_STRIDE_0
        + (h * FULL_K_PER_HEAD + k[:, None]) * W_STRIDE_1
        + n[None, :] * W_STRIDE_2
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_block in range(PADDED_INPUT_DIM // BLOCK_K):
        a = tl.load(input_block_ptr)
        weight_block = tl.load(weight_block_ptr)
        acc = tl.dot(a, weight_block, acc, input_precision="ieee")
        input_block_ptr += BLOCK_K
        weight_block_ptr += BLOCK_K * W_STRIDE_1
    output_block_ptr = (
        matmul_output_ptr
        + (segment * PADDED_ROWS + mt * BLOCK_M)
        * (NUM_HEADS * PADDED_OUTPUT_DIM)
        + h * PADDED_OUTPUT_DIM
        + m[:, None] * (NUM_HEADS * PADDED_OUTPUT_DIM)
        + n[None, :]
    )
    tl.store(output_block_ptr, acc)


@triton.jit
def _step_a_q_scatter_rows_kernel(
    matmul_output_ptr,
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    out_ptr,
    PADDED_ROWS: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
    PERM_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    FULL: tl.constexpr,
):
    m = tl.program_id(0)
    segment = tl.program_id(1)
    if FULL:
        if segment == NUM_SEGMENTS:
            first = tl.load(seg_indptr)
            last = tl.load(seg_indptr + NUM_SEGMENTS)
            if (m < first) | (m >= last):
                offset = tl.arange(0, BLOCK_SIZE)
                row = tl.load(permutation + m * PERM_STRIDE).to(tl.int32)
                tl.store(out_ptr + row * BLOCK_SIZE + offset, 0.0)
            return
    segment_start = tl.load(seg_indptr + segment)
    segment_end = tl.load(seg_indptr + segment + 1)
    if m >= segment_end - segment_start:
        return
    adapter = tl.load(weight_indices + segment)
    p = tl.minimum(segment_start + m, PERM_SIZE - 1)
    row = tl.load(permutation + p * PERM_STRIDE).to(tl.int32)
    offset = tl.arange(0, BLOCK_SIZE)
    if tl.load(lora_ranks + adapter) == 0:
        tl.store(out_ptr + row * BLOCK_SIZE + offset, 0.0)
        return
    c = tl.load(
        matmul_output_ptr + (segment * PADDED_ROWS + m) * BLOCK_SIZE + offset
    )
    tl.store(out_ptr + row * BLOCK_SIZE + offset, c)


def _step_a_q_segmented(
    q_nope,
    weights,
    batch_info,
    full_K_per_head,
    seg_indptr,
    weight_indices,
    ranks,
    max_len,
):
    (num_tokens, num_heads, input_dim) = q_nope.shape
    output_dim = weights.shape[-1]
    permutation = batch_info.permutation
    if num_tokens == 0 or batch_info.bs == 0 or max_len == 0:
        return torch.zeros(
            (num_tokens, num_heads, output_dim),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
    fast_shape = (
        input_dim >= 64
        and input_dim & input_dim - 1 == 0
        and (output_dim >= 16)
        and (output_dim & output_dim - 1 == 0)
    )
    padded_n = triton.cdiv(output_dim, 64) * 64
    largest_read = (
        (num_heads - 1) * full_K_per_head + input_dim - 1
    ) * weights.stride(1)
    largest_read += (padded_n - 1) * weights.stride(2)
    adapter_span = (weights.shape[1] - 1) * weights.stride(1)
    adapter_span += (output_dim - 1) * weights.stride(2)
    if not fast_shape or (
        permutation is not None and largest_read > adapter_span
    ):
        out = torch.zeros(
            (num_tokens, num_heads, output_dim),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        has_perm = permutation is not None
        permutation_ptr = permutation if has_perm else seg_indptr
        _step_a_q_gemv_kernel[max_len, num_heads, batch_info.bs * output_dim](
            q_nope,
            weights,
            out,
            seg_indptr,
            weight_indices,
            ranks,
            permutation_ptr,
            num_heads,
            output_dim,
            input_dim,
            full_K_per_head,
            q_nope.stride(0),
            q_nope.stride(1),
            q_nope.stride(2),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            permutation_ptr.stride(0),
            triton.next_power_of_2(input_dim),
            has_perm,
            num_warps=4,
        )
        return out
    block_n = max(16, triton.next_power_of_2(output_dim))
    block_k = max(16, triton.next_power_of_2(input_dim))
    if permutation is None:
        out = torch.zeros(
            (num_tokens, num_heads, output_dim),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        block_m = min(256, max(16, triton.next_power_of_2(max_len)))
        tiles = max(1, triton.cdiv(max_len, block_m))
        _step_a_q_direct_kernel[tiles * batch_info.bs, num_heads](
            q_nope,
            weights,
            out,
            seg_indptr,
            weight_indices,
            ranks,
            num_heads,
            output_dim,
            input_dim,
            full_K_per_head,
            tiles,
            q_nope.stride(0),
            q_nope.stride(1),
            q_nope.stride(2),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            block_m,
            block_n,
            block_k,
            num_stages=4,
        )
        return out
    head_input_size = num_heads * input_dim
    head_output_size = num_heads * output_dim
    row_major = (
        head_input_size & head_input_size - 1 == 0
        and head_output_size & head_output_size - 1 == 0
    )
    if row_major:
        block_m = 16
        padded_rows = max(block_m, triton.cdiv(max_len, block_m) * block_m)
        packed_input = torch.empty(
            (batch_info.bs, padded_rows, head_input_size),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        matmul_output = torch.empty(
            (batch_info.bs, padded_rows, head_output_size),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        full_permutation = permutation.numel() == num_tokens
        if full_permutation:
            out = torch.empty(
                (num_tokens + 1, head_output_size),
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
        else:
            out = torch.zeros(
                (num_tokens + 1, head_output_size),
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
        _step_a_q_pack_rows_kernel[padded_rows, batch_info.bs](
            q_nope,
            seg_indptr,
            weight_indices,
            ranks,
            permutation,
            packed_input,
            input_dim,
            padded_rows,
            q_nope.stride(0),
            q_nope.stride(1),
            q_nope.stride(2),
            permutation.stride(0),
            permutation.numel(),
            head_input_size,
        )
        _step_a_q_matmul_rows_kernel[
            batch_info.bs * num_heads * (padded_rows // block_m),
        ](
            packed_input,
            weights,
            weight_indices,
            matmul_output,
            num_heads,
            full_K_per_head,
            padded_rows,
            input_dim,
            output_dim,
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            padded_rows // block_m,
            1,
            block_m,
            output_dim,
            input_dim,
            num_warps=4,
            num_stages=4,
        )
        _step_a_q_scatter_rows_kernel[
            num_tokens if full_permutation else max_len,
            batch_info.bs + int(full_permutation),
        ](
            matmul_output,
            seg_indptr,
            weight_indices,
            ranks,
            permutation,
            out,
            padded_rows,
            permutation.stride(0),
            permutation.numel(),
            head_output_size,
            batch_info.bs,
            full_permutation,
        )
        return out[:num_tokens].view(num_tokens, num_heads, output_dim)
    permutation_size = permutation.numel()
    padded_rows = max(64, triton.cdiv(max_len, 64) * 64)
    padded_input_dim = input_dim
    padded_output_dim = triton.cdiv(output_dim, 64) * 64
    num_head_batches = batch_info.bs * num_heads
    packed_input = torch.zeros(
        (num_head_batches, padded_rows, padded_input_dim),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )
    matmul_output = torch.empty(
        (num_head_batches, padded_rows, padded_output_dim),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )
    out = torch.zeros(
        (num_tokens + 1, num_heads, output_dim),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )
    _step_a_q_pack_heads_kernel[max_len, num_heads, batch_info.bs](
        q_nope,
        seg_indptr,
        weight_indices,
        ranks,
        permutation,
        packed_input,
        num_heads,
        input_dim,
        padded_input_dim,
        padded_rows,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        permutation.stride(0),
        permutation_size,
        triton.next_power_of_2(padded_input_dim),
    )
    _step_a_q_matmul_heads_kernel[
        num_head_batches * (padded_rows // 64) * (padded_output_dim // 64),
    ](
        packed_input,
        weights,
        weight_indices,
        matmul_output,
        num_heads,
        full_K_per_head,
        padded_rows,
        padded_input_dim,
        padded_output_dim,
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        padded_rows // 64,
        padded_output_dim // 64,
        64,
        64,
        min(128, padded_input_dim),
        num_warps=4,
        num_stages=4,
    )
    _step_a_q_scatter_heads_kernel[max_len, num_heads, batch_info.bs](
        matmul_output,
        seg_indptr,
        weight_indices,
        ranks,
        permutation,
        out,
        num_heads,
        output_dim,
        padded_output_dim,
        padded_rows,
        num_tokens,
        permutation.stride(0),
        permutation_size,
        max(32, output_dim),
    )
    return out[:num_tokens]


@triton.jit
def _step_a_q_adapter_kernel(
    input_ptr,
    weight_ptr,
    out_ptr,
    seg_indptr,
    weight_indices,
    lora_ranks,
    NUM_TOKENS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    INPUT_DIM: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    FULL_K_PER_HEAD: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    W_STRIDE_0: tl.constexpr,
    W_STRIDE_1: tl.constexpr,
    W_STRIDE_2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILES: tl.constexpr,
):
    tile = tl.program_id(0) % TILES
    h = tl.program_id(1)
    adapter = tl.program_id(0) // TILES
    if (tile == 0) & (h == 0) & (adapter == 0):
        offset = tl.arange(0, 1024)
        first = tl.load(seg_indptr) * NUM_HEADS * OUTPUT_DIM
        last = tl.load(seg_indptr + NUM_SEGMENTS) * NUM_HEADS * OUTPUT_DIM
        for start in range(0, first, 1024):
            pos = start + offset
            tl.store(out_ptr + pos, 0.0, pos < first)
        for start in range(last, NUM_TOKENS * NUM_HEADS * OUTPUT_DIM, 1024):
            pos = start + offset
            tl.store(
                out_ptr + pos, 0.0, pos < NUM_TOKENS * NUM_HEADS * OUTPUT_DIM
            )
    active = tl.load(lora_ranks + adapter) > 0
    k = tl.arange(0, BLOCK_K)
    n = tl.arange(0, BLOCK_N)
    wb = tl.load(
        weight_ptr
        + adapter * W_STRIDE_0
        + (h * FULL_K_PER_HEAD + k[:, None]) * W_STRIDE_1
        + n[None, :] * W_STRIDE_2,
        (k < INPUT_DIM)[:, None] & (n < OUTPUT_DIM)[None, :],
        0,
    )
    for segment in range(NUM_SEGMENTS):
        if tl.load(weight_indices + segment) == adapter:
            segment_start = tl.load(seg_indptr + segment)
            segment_end = tl.load(seg_indptr + segment + 1)
            if tile * BLOCK_M < segment_end - segment_start:
                m = segment_start + tile * BLOCK_M + tl.arange(0, BLOCK_M)
                op = (
                    out_ptr
                    + (m[:, None] * NUM_HEADS + h) * OUTPUT_DIM
                    + n[None, :]
                )
                mask = (m < segment_end)[:, None] & (n < OUTPUT_DIM)[None, :]
                if active:
                    x = tl.load(
                        input_ptr
                        + m[:, None] * X_STRIDE_0
                        + h * X_STRIDE_1
                        + k[None, :] * X_STRIDE_2,
                        (m < segment_end)[:, None] & (k < INPUT_DIM)[None, :],
                        0,
                    )
                    value = tl.dot(x, wb, input_precision="ieee")
                    tl.store(op, value, mask)
                else:
                    tl.store(
                        op, tl.full((BLOCK_M, BLOCK_N), 0, tl.float32), mask
                    )


def step_a_q(q_nope, B_buf, batch_info, full_K_per_head):
    (num_tokens, num_heads, input_dim) = q_nope.shape
    output_dim = B_buf.shape[-1]
    max_len = getattr(batch_info, "max_len", None)
    if max_len is None:
        max_len = num_tokens
    if q_nope.stride(-1) != 1:
        q_nope = q_nope.contiguous()
    if B_buf.stride(-1) != 1:
        B_buf = B_buf.contiguous()
    (seg_indptr, weight_indices, ranks) = _step_a_q_metadata(batch_info)
    if (
        batch_info.permutation is not None
        or num_tokens < 64
        or max_len <= 16
        or (B_buf.shape[0] >= batch_info.bs)
        or (input_dim < 64)
        or (input_dim > 256)
        or (output_dim < 16)
        or (output_dim > 64)
        or (q_nope.dtype == torch.float32)
        or (input_dim & input_dim - 1)
        or (output_dim & output_dim - 1)
    ):
        return _step_a_q_segmented(
            q_nope,
            B_buf,
            batch_info,
            full_K_per_head,
            seg_indptr,
            weight_indices,
            ranks,
            max_len,
        )
    out = torch.empty(
        (num_tokens, num_heads, output_dim),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )
    block_m = min(256, max(16, triton.next_power_of_2(max_len)))
    tiles = triton.cdiv(max_len, block_m)
    _step_a_q_adapter_kernel[tiles * B_buf.shape[0], num_heads](
        q_nope,
        B_buf,
        out,
        seg_indptr,
        weight_indices,
        ranks,
        num_tokens,
        num_heads,
        input_dim,
        output_dim,
        full_K_per_head,
        batch_info.bs,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        B_buf.stride(0),
        B_buf.stride(1),
        B_buf.stride(2),
        block_m,
        max(16, triton.next_power_of_2(output_dim)),
        max(16, triton.next_power_of_2(input_dim)),
        tiles,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["step_a_q"]
