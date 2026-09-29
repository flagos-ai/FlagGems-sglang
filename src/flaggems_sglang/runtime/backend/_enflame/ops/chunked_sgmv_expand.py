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

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _sgmv_gemm_kernel_enflame(
    a_ptr,
    b_ptr,
    c_ptr,
    scaling,
    M,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Kunlunxin recipe (T28 E11 / T37 E4): a completely regular GEMM
    # with no segment metadata, no indirect rows and no runtime
    # branches - operands are upcast to fp32 and the dot is ieee,
    # the only configuration kunlunxin is known to compute correctly.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_k = offs_k < K - k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        accumulator = tl.dot(a, b, acc=accumulator, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base = tl.load(c_ptrs, mask=mask, other=0.0)
    tl.store(c_ptrs, base + accumulator * scaling, mask=mask)


_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 32
_GROUP_M = 8


def _launch_gemm(a, b, c, scaling, output_width, rank):
    m = a.shape[0]
    if m == 0:
        return
    # Advance B along rank in the K loop. Bound shared memory for large
    # ranks instead of forcing the entire rank into one program tile.
    block_k = min(triton.next_power_of_2(max(rank, 16)), 128)
    grid = (triton.cdiv(m, _BLOCK_M) * triton.cdiv(output_width, _BLOCK_N),)
    _sgmv_gemm_kernel_enflame[grid](
        a,
        b,
        c,
        scaling,
        m,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        c.stride(0),
        c.stride(1),
        N=output_width,
        K=rank,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=block_k,
        GROUP_M=_GROUP_M,
        num_warps=4,
        num_stages=2,
    )


def chunked_sgmv_expand(
    x, weights, batch_info, slice_offsets, max_slice_size, base_output
):
    output = base_output.clone()
    n_slices = slice_offsets.numel() - 1
    rank = weights.shape[-1]
    if x.shape[1] != n_slices * rank:
        raise ValueError("x width must equal n_slices * rank")
    if (
        output.numel() == 0
        or n_slices <= 0
        or batch_info.bs == 0
        or x.shape[0] == 0
    ):
        return output

    # Coalesce repeated adapters before batching their gather/scatter.
    # Dot operands remain regular matrices; kernels carry no route metadata.
    seg_indptr = batch_info.seg_indptr.detach().cpu().tolist()
    weight_indices = batch_info.weight_indices.detach().cpu().tolist()
    lora_ranks = batch_info.lora_ranks.detach().cpu().tolist()
    scalings = batch_info.scalings.detach().cpu().tolist()
    slice_list = slice_offsets.detach().cpu().tolist()
    permutation = batch_info.permutation

    adapter_segments = {}
    for b in range(batch_info.bs):
        start, end = seg_indptr[b], seg_indptr[b + 1]
        if start == end:
            continue
        w_idx = weight_indices[b]
        if lora_ranks[w_idx] == 0:
            continue
        adapter_segments.setdefault(w_idx, []).append((start, end))

    if not adapter_segments:
        return output

    # Materialize all active rows once; each adapter still sees a regular GEMM.
    row_parts = []
    groups = []
    cursor = 0
    for w_idx, segments in adapter_segments.items():
        group_start = cursor
        for start, end in segments:
            row_parts.append(permutation[start:end])
            cursor += end - start
        groups.append((w_idx, group_start, cursor))
    rows = (
        row_parts[0] if len(row_parts) == 1 else torch.cat(row_parts)
    ).long()
    packed_x = x.index_select(0, rows).float()
    packed_out = output.index_select(0, rows).float()

    for w_idx, start, end in groups:
        scaling = float(scalings[w_idx])
        x_seg = packed_x[start:end]
        seg_out = packed_out[start:end]
        for i in range(n_slices):
            o_start, o_end = int(slice_list[i]), int(slice_list[i + 1])
            if o_start == o_end:
                continue
            _launch_gemm(
                x_seg[:, i * rank : (i + 1) * rank],
                weights[w_idx, o_start:o_end, :],
                seg_out[:, o_start:o_end],
                scaling,
                o_end - o_start,
                rank,
            )
    output.index_copy_(0, rows, packed_out.to(base_output.dtype))
    return output


__all__ = ["chunked_sgmv_expand"]
