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

import torch
import triton
import triton.language as tl

_LARGE_VOCAB_LIMIT = 32768
_STREAMING_W_SMALL_BATCH = 4096
_STREAMING_W_LARGE_BATCH = 2048
_STREAMING_TREE_K = 8
_SMALL_BATCH_LIMIT = 48

# int64 tensors are int32-backed (Long->Int storage) unless true 64-bit
# support is enabled (TORCH_GCU_ENABLE_INT64_AND_UINT64). Probe the real
# layout once per device instead of assuming: with true 64-bit storage an
# int32 view of an int64 tensor interleaves low/high words, which scrambles
# every value the kernel writes through the view.
_I64_WORD_STRIDE = None  # 1 = packed int32 words, 2 = true int64 words


def _int64_word_stride(device):
    global _I64_WORD_STRIDE
    if _I64_WORD_STRIDE is None:
        probe = torch.zeros(4, dtype=torch.int64, device=device)
        probe[2] = 7
        words = probe.view(torch.int32).tolist()
        if len(words) == 8 and words[4] == 7 and words[5] == 0:
            _I64_WORD_STRIDE = 2  # true 64-bit storage: low word at 2*i
        elif len(words) >= 3 and words[2] == 7:
            _I64_WORD_STRIDE = 1  # packed int32 storage: consecutive words
        else:
            _I64_WORD_STRIDE = 2
    return _I64_WORD_STRIDE


@triton.jit
def _write_outputs(
    row,
    selected_index,
    positions_ptr,
    draft_tokens_ptr,
    topk_p_ptr,
    topk_index_ptr,
    out_positions_ptr,
    out_draft_tokens_ptr,
    stride_positions,
    stride_draft_batch,
    stride_draft_column,
    stride_out_draft_batch,
    stride_out_draft_column,
    DRAFT_SIZE: tl.constexpr,
    BLOCK_DRAFT: tl.constexpr,
    DRAFT_TOKEN_COLUMN: tl.constexpr,
    HAS_DRAFT_TOKENS: tl.constexpr,
):
    tl.store(topk_p_ptr + row, 1.0)
    tl.store(topk_index_ptr + row, selected_index)
    position = tl.load(positions_ptr + row * stride_positions)
    tl.store(out_positions_ptr + row, position + 1)

    if HAS_DRAFT_TOKENS:
        columns = tl.arange(0, BLOCK_DRAFT)
        mask = columns < DRAFT_SIZE
        values = tl.load(
            draft_tokens_ptr
            + row * stride_draft_batch
            + columns * stride_draft_column,
            mask=mask,
            other=0,
        )
        values = tl.where(
            columns == DRAFT_TOKEN_COLUMN,
            selected_index,
            values,
        )
        tl.store(
            out_draft_tokens_ptr
            + row * stride_out_draft_batch
            + columns * stride_out_draft_column,
            values,
            mask=mask,
        )


@triton.jit
def _draft_topk1_streaming_kernel(
    logits_ptr,
    positions_ptr,
    draft_tokens_ptr,
    topk_p_ptr,
    topk_index_ptr,
    out_positions_ptr,
    out_draft_tokens_ptr,
    stride_positions,
    stride_draft_batch,
    stride_draft_column,
    stride_out_draft_batch,
    stride_out_draft_column,
    stride_logits_batch,
    stride_logits_vocab,
    VOCAB_SIZE: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    DRAFT_SIZE: tl.constexpr,
    BLOCK_DRAFT: tl.constexpr,
    DRAFT_TOKEN_COLUMN: tl.constexpr,
    HAS_DRAFT_TOKENS: tl.constexpr,
):
    # One program per row; per loop iteration K independent W-wide loads are
    # issued back-to-back (GCU400 exposes per-iteration load latency unless
    # several loads are in flight), then merged with a leftmost-preferring
    # combine tree into a single per-lane running (m, k) state. Strict '>'
    # against the running state keeps earlier iterations on ties, and '>='
    # inside the tree keeps the leftmost slot, so the composite is the exact
    # leftmost argmax.
    row = tl.program_id(0)
    lane = tl.arange(0, W)
    base = logits_ptr + row * stride_logits_batch
    running_max = tl.full([W], -float("inf"), tl.float32)
    running_index = tl.zeros([W], tl.int32)
    CHUNK: tl.constexpr = K * W
    N_ITER: tl.constexpr = VOCAB_SIZE // CHUNK
    tail = VOCAB_SIZE - N_ITER * CHUNK
    for it in range(0, N_ITER):
        c0 = it * CHUNK + lane
        v0 = tl.load(base + c0 * stride_logits_vocab)
        v1 = tl.load(base + (c0 + W) * stride_logits_vocab)
        if K >= 4:
            v2 = tl.load(base + (c0 + 2 * W) * stride_logits_vocab)
            v3 = tl.load(base + (c0 + 3 * W) * stride_logits_vocab)
        if K >= 8:
            v4 = tl.load(base + (c0 + 4 * W) * stride_logits_vocab)
            v5 = tl.load(base + (c0 + 5 * W) * stride_logits_vocab)
            v6 = tl.load(base + (c0 + 6 * W) * stride_logits_vocab)
            v7 = tl.load(base + (c0 + 7 * W) * stride_logits_vocab)
        if K == 2:
            vM = tl.where(v0 >= v1, v0, v1)
            cM = tl.where(v0 >= v1, c0, c0 + W)
        if K == 4:
            vA = tl.where(v0 >= v1, v0, v1)
            cA = tl.where(v0 >= v1, c0, c0 + W)
            vB = tl.where(v2 >= v3, v2, v3)
            cB = tl.where(v2 >= v3, c0 + 2 * W, c0 + 3 * W)
            vM = tl.where(vA >= vB, vA, vB)
            cM = tl.where(vA >= vB, cA, cB)
        if K == 8:
            vA = tl.where(v0 >= v1, v0, v1)
            cA = tl.where(v0 >= v1, c0, c0 + W)
            vB = tl.where(v2 >= v3, v2, v3)
            cB = tl.where(v2 >= v3, c0 + 2 * W, c0 + 3 * W)
            vC = tl.where(v4 >= v5, v4, v5)
            cC = tl.where(v4 >= v5, c0 + 4 * W, c0 + 5 * W)
            vD = tl.where(v6 >= v7, v6, v7)
            cD = tl.where(v6 >= v7, c0 + 6 * W, c0 + 7 * W)
            vE = tl.where(vA >= vB, vA, vB)
            cE = tl.where(vA >= vB, cA, cB)
            vF = tl.where(vC >= vD, vC, vD)
            cF = tl.where(vC >= vD, cC, cD)
            vM = tl.where(vE >= vF, vE, vF)
            cM = tl.where(vE >= vF, cE, cF)
        hit = vM > running_max
        running_max = tl.where(hit, vM, running_max)
        running_index = tl.where(hit, cM, running_index)
    if tail > 0:
        t_base = N_ITER * CHUNK
        for s in tl.static_range(K):
            sbase = t_base + s * W
            rem = tail - s * W
            vs = tl.load(
                base + (sbase + lane) * stride_logits_vocab,
                mask=lane < rem,
                other=-float("inf"),
            )
            hit = vs > running_max
            running_max = tl.where(hit, vs, running_max)
            running_index = tl.where(hit, sbase + lane, running_index)
    global_max = tl.max(running_max, axis=0)
    selected_index = tl.min(
        tl.where(running_max == global_max, running_index, 2147483647),
        axis=0,
    )
    _write_outputs(
        row,
        selected_index,
        positions_ptr,
        draft_tokens_ptr,
        topk_p_ptr,
        topk_index_ptr,
        out_positions_ptr,
        out_draft_tokens_ptr,
        stride_positions,
        stride_draft_batch,
        stride_draft_column,
        stride_out_draft_batch,
        stride_out_draft_column,
        DRAFT_SIZE,
        BLOCK_DRAFT,
        DRAFT_TOKEN_COLUMN,
        HAS_DRAFT_TOKENS,
    )


def chunk_topk1_outputs(
    next_token_logits, positions, draft_tokens, draft_token_column
):
    # Host-side prep is on the measured op-call critical path; keep the
    # per-call tensor-API op count minimal (allocs + int32 views only).
    batch_size, vocab_size = next_token_logits.shape
    device = next_token_logits.device
    topk_p = torch.empty(
        (batch_size, 1),
        dtype=torch.float32,
        device=device,
    )
    topk_index = torch.empty(
        (batch_size, 1),
        dtype=torch.int64,
        device=device,
    )
    out_positions = torch.empty(
        (batch_size,),
        dtype=torch.int64,
        device=device,
    )
    stride_positions = positions.stride(0)
    positions_ptr = positions
    topk_index_ptr = topk_index
    out_positions_ptr = out_positions
    if positions.dtype == torch.int64 and _int64_word_stride(device) == 1:
        # int64 tensors are int32-backed (Long->Int storage); one int32 word
        # per logical element, so the kernel addresses them through plain
        # int32 views without flattening (explicit strides drive addressing).
        positions_ptr = positions.view(torch.int32)
        topk_index_ptr = topk_index.view(torch.int32)
        out_positions_ptr = out_positions.view(torch.int32)
        stride_positions = 1

    has_draft_tokens = draft_tokens is not None
    if has_draft_tokens:
        draft_size = draft_tokens.shape[1]
        normalized_column = draft_token_column
        if normalized_column < 0:
            normalized_column += draft_size
        out_draft_tokens = torch.empty_like(draft_tokens)
        if (
            draft_tokens.dtype == torch.int64
            and _int64_word_stride(device) == 1
        ):
            draft_tokens_ptr = draft_tokens.view(torch.int32)
            out_draft_tokens_ptr = out_draft_tokens.view(torch.int32)
            stride_draft_batch = draft_size
            stride_draft_column = 1
            stride_out_draft_batch = draft_size
            stride_out_draft_column = 1
        else:
            draft_tokens_ptr = draft_tokens
            out_draft_tokens_ptr = out_draft_tokens
            stride_draft_batch = draft_tokens.stride(0)
            stride_draft_column = draft_tokens.stride(1)
            stride_out_draft_batch = out_draft_tokens.stride(0)
            stride_out_draft_column = out_draft_tokens.stride(1)
        block_draft = triton.next_power_of_2(draft_size)
    else:
        draft_size = 0
        normalized_column = 0
        out_draft_tokens = None
        draft_tokens_ptr = positions_ptr
        out_draft_tokens_ptr = out_positions_ptr
        stride_draft_batch = 0
        stride_draft_column = 0
        stride_out_draft_batch = 0
        stride_out_draft_column = 0
        block_draft = 1

    output_args = (
        positions_ptr,
        draft_tokens_ptr,
        topk_p,
        topk_index_ptr,
        out_positions_ptr,
        out_draft_tokens_ptr,
        stride_positions,
        stride_draft_batch,
        stride_draft_column,
        stride_out_draft_batch,
        stride_out_draft_column,
    )
    output_meta = {
        "DRAFT_SIZE": draft_size,
        "BLOCK_DRAFT": block_draft,
        "DRAFT_TOKEN_COLUMN": normalized_column,
        "HAS_DRAFT_TOKENS": has_draft_tokens,
    }
    return (
        topk_p,
        topk_index,
        out_positions,
        out_draft_tokens,
        output_args,
        output_meta,
        batch_size,
        vocab_size,
    )


def draft_topk1(
    next_token_logits,
    positions,
    draft_tokens=None,
    draft_token_column=0,
):
    (
        topk_p,
        topk_index,
        out_positions,
        out_draft_tokens,
        output_args,
        output_meta,
        batch_size,
        vocab_size,
    ) = chunk_topk1_outputs(
        next_token_logits,
        positions,
        draft_tokens,
        draft_token_column,
    )

    if vocab_size > _LARGE_VOCAB_LIMIT:
        # Latency regime (grid below the 48 SPs) prefers wider per-load
        # vectors; the throughput regime prefers narrower ones (dual-program
        # DSM occupancy). K-way issue depth is shared by both arms.
        if batch_size < _SMALL_BATCH_LIMIT:
            streaming_width = _STREAMING_W_SMALL_BATCH
        else:
            streaming_width = _STREAMING_W_LARGE_BATCH
        streaming_k = _STREAMING_TREE_K
    else:
        # Small vocab: keep every load inside the toolchain-safe width
        # (masked fp32 loads >= 8192 lanes fail to fresh-compile, see
        # compiler_issues/vllm_silu_flat_slice_start.md) and pick the
        # largest K whose chunk still fits the row.
        streaming_width = min(
            _STREAMING_W_SMALL_BATCH,
            triton.next_power_of_2(vocab_size),
        )
        streaming_k = 2
        if streaming_width * 4 <= vocab_size:
            streaming_k = 4
        if streaming_width * 8 <= vocab_size:
            streaming_k = 8
    _draft_topk1_streaming_kernel[(batch_size,)](
        next_token_logits,
        *output_args,
        next_token_logits.stride(0),
        next_token_logits.stride(1),
        VOCAB_SIZE=vocab_size,
        W=streaming_width,
        K=streaming_k,
        **output_meta,
        num_warps=4,
        num_stages=1,
    )

    return topk_p, topk_index, out_positions, out_draft_tokens


__all__ = ["draft_topk1"]
