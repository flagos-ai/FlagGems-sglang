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

_MAX_GROUPED_N = 8192
_MAX_TILE_Q = 256


@triton.jit
def _seqlens_expand_grouped_kernel(
    qo_ptr,
    kv_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    Q: tl.constexpr,
    SINGLE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * GROUP_N

    rows = start + tl.arange(0, GROUP_N)
    row_ok = rows < N
    qo = tl.load(qo_ptr + rows, mask=row_ok, other=0).to(tl.int32)
    kv = tl.load(kv_ptr + rows, mask=row_ok, other=0).to(tl.int32)

    if SINGLE:

        group_base = 0
    else:

        offs = tl.arange(0, BLOCK_N)
        qo_all = tl.load(qo_ptr + offs, mask=offs < N, other=0).to(tl.int32)
        csum = tl.cumsum(qo_all, axis=0)

        group_base = tl.sum(tl.where(offs == start - 1, csum, 0), axis=0)

    rel = tl.cumsum(qo, axis=0) - qo
    base_val = kv - qo + 1
    cols = tl.arange(0, Q)
    ptrs = out_ptr + (group_base + rel)[:, None] + cols[None, :]
    vals = tl.maximum(base_val[:, None] + cols[None, :], 0)
    mask = row_ok[:, None] & (cols[None, :] < qo[:, None])
    tl.store(ptrs, vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_Q": 16}, num_warps=1),
        triton.Config({"BLOCK_N": 256, "BLOCK_Q": 32}, num_warps=2),
        triton.Config({"BLOCK_N": 512, "BLOCK_Q": 64}, num_warps=4),
        triton.Config({"BLOCK_N": 1024, "BLOCK_Q": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 4096, "BLOCK_Q": 256}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _seqlens_expand_fused_kernel(
    qo_ptr,
    kv_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    i = tl.program_id(0)

    carry = 0
    for start in range(0, i, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        qo_chunk = tl.load(qo_ptr + offs, mask=offs < i, other=0).to(tl.int32)
        carry += tl.sum(qo_chunk, axis=0)

    qo = tl.load(qo_ptr + i).to(tl.int32)
    kv = tl.load(kv_ptr + i).to(tl.int32)
    base_val = kv - qo + 1
    for start in range(0, qo, BLOCK_Q):
        offs = start + tl.arange(0, BLOCK_Q)
        mask = offs < qo
        vals = tl.maximum(base_val + offs, 0)
        tl.store(out_ptr + carry + offs, vals, mask=mask)


def seqlens_expand(extend_seq_lens, seq_lens, total_len, max_q_len):
    out = torch.empty(
        total_len, dtype=torch.int32, device=extend_seq_lens.device
    )
    n = extend_seq_lens.numel()
    if n == 0 or total_len == 0:
        return out
    if not extend_seq_lens.is_contiguous():
        extend_seq_lens = extend_seq_lens.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()

    if n <= _MAX_GROUPED_N and max_q_len <= _MAX_TILE_Q:
        block_n = triton.next_power_of_2(n)
        tile_q = triton.next_power_of_2(max(1, max_q_len))

        if n <= 16:
            grid = (1,)
            _seqlens_expand_grouped_kernel[grid](
                extend_seq_lens,
                seq_lens,
                out,
                n,
                BLOCK_N=block_n,
                GROUP_N=block_n,
                Q=tile_q,
                SINGLE=1,
                num_warps=2,
            )
            return out

        if n <= 1024:
            group_n, num_warps = 16, 2
        else:

            group_n, num_warps = 64, 8
        grid = (triton.cdiv(n, group_n),)
        _seqlens_expand_grouped_kernel[grid](
            extend_seq_lens,
            seq_lens,
            out,
            n,
            BLOCK_N=block_n,
            GROUP_N=group_n,
            Q=tile_q,
            SINGLE=0,
            num_warps=num_warps,
        )
        return out

    _seqlens_expand_fused_kernel[(n,)](
        extend_seq_lens,
        seq_lens,
        out,
        n,
    )
    return out


__all__ = ["seqlens_expand"]
