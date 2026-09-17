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

_BLOCK_M = 64
_BLOCK_N = 128
_BLOCK_K = 32
_GROUP_M = 8


@triton.jit
def _qkv_lora_b_xpu_kernel(
    x_ptr,
    w_ptr,
    base_ptr,
    out_ptr,
    scalings_ptr,
    W_INDEX: tl.constexpr,
    STRIDE_W_LORA: tl.constexpr,
    X_COL_BASE: tl.constexpr,
    O_COL_BASE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_X_M: tl.constexpr,
    STRIDE_X_K: tl.constexpr,
    STRIDE_W_N: tl.constexpr,
    STRIDE_W_K: tl.constexpr,
    STRIDE_O_M: tl.constexpr,
    STRIDE_O_N: tl.constexpr,
    DOT_DTYPE: tl.constexpr,
    IEEE: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    scaling = tl.load(scalings_ptr + W_INDEX)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    tile_id = tl.program_id(0)
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    safe_columns = columns if EVEN_N else tl.minimum(columns, N - 1)

    x_ptrs = (
        x_ptr
        + rows[:, None] * STRIDE_X_M
        + (X_COL_BASE + k_offsets[None, :]) * STRIDE_X_K
    )
    w_ptrs = (
        w_ptr
        + W_INDEX * STRIDE_W_LORA
        + k_offsets[:, None] * STRIDE_W_K
        + (O_COL_BASE + safe_columns[None, :]) * STRIDE_W_N
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(x_ptrs)
        b = tl.load(w_ptrs)
        if IEEE:
            accumulator = tl.dot(
                a.to(DOT_DTYPE),
                b.to(DOT_DTYPE),
                accumulator,
                input_precision="ieee",
                out_dtype=tl.float32,
            )
        else:
            accumulator = tl.dot(
                a.to(DOT_DTYPE),
                b.to(DOT_DTYPE),
                accumulator,
                out_dtype=tl.float32,
            )
        x_ptrs += BLOCK_K * STRIDE_X_K
        w_ptrs += BLOCK_K * STRIDE_W_K

    offsets = (
        rows[:, None] * STRIDE_O_M
        + (O_COL_BASE + safe_columns[None, :]) * STRIDE_O_N
    )
    base = tl.load(base_ptr + offsets).to(tl.float32)
    result = (base + accumulator * scaling).to(out_ptr.dtype.element_ty)
    if EVEN_N:
        tl.store(out_ptr + offsets, result)
    else:
        tl.store(out_ptr + offsets, result, mask=(columns < N)[None, :])


def _dot_plan(dtype):
    if dtype == torch.float32:
        return tl.float32, True
    return {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[
        dtype
    ], False


def _pad_rows(tensor, padded_rows):
    rows = tensor.shape[0]
    if rows == padded_rows:
        return tensor
    out = torch.zeros(
        (padded_rows, tensor.shape[1]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    out[:rows] = tensor
    return out


def qkv_lora_b(
    x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim, base_output
):
    x = x.contiguous()
    weights = qkv_lora_b.contiguous()
    out = base_output.clone()

    rank_dim = weights.shape[-1]
    offsets_list = output_offset.tolist()
    n_slices = len(offsets_list) - 1
    bs = int(batch_info.bs)

    seg = batch_info.seg_indptr.tolist()
    widx = batch_info.weight_indices.tolist()
    ranks = batch_info.lora_ranks.tolist()
    permutation = batch_info.permutation
    has_perm = permutation is not None

    dot_dtype, ieee = _dot_plan(x.dtype)
    block_k = _BLOCK_K if rank_dim % _BLOCK_K == 0 else 16
    sw = weights.stride()

    for b in range(bs):
        start = seg[b]
        end = seg[b + 1]
        if start == end:
            continue
        w_idx = widx[b]
        if ranks[w_idx] == 0:
            continue

        seg_len = end - start
        block_m = (
            _BLOCK_M if seg_len >= _BLOCK_M else (32 if seg_len >= 32 else 16)
        )
        padded = -(-seg_len // block_m) * block_m

        if has_perm:
            rows = permutation[start:end]
            x_seg = x[rows].contiguous()
            base_seg = out[rows].contiguous()
        else:
            rows = None
            x_seg = x[start:end].contiguous()
            base_seg = out[start:end].contiguous()

        x_pad = _pad_rows(x_seg, padded)
        base_pad = _pad_rows(base_seg, padded)
        out_pad = base_pad.clone()

        sx = x_pad.stride()
        sb = base_pad.stride()
        for i in range(n_slices):
            o_start = offsets_list[i]
            o_end = offsets_list[i + 1]
            width = o_end - o_start
            block_n = _BLOCK_N if width >= _BLOCK_N else 16
            grid = ((-(-padded // block_m)) * (-(-width // block_n)),)
            _qkv_lora_b_xpu_kernel[grid](
                x_pad,
                weights,
                base_pad,
                out_pad,
                batch_info.scalings,
                w_idx,
                sw[0],
                i * rank_dim,
                o_start,
                padded,
                width,
                rank_dim,
                sx[0],
                sx[1],
                sw[1],
                sw[2],
                sb[0],
                sb[1],
                dot_dtype,
                ieee,
                width % block_n == 0,
                block_m,
                block_n,
                block_k,
                _GROUP_M,
                num_warps=4,
                num_stages=4,
            )
        if has_perm:
            out[rows] = out_pad[:seg_len]
        else:
            out[start:end] = out_pad[:seg_len]
    return out


__all__ = ["qkv_lora_b"]
