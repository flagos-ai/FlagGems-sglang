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


# A single program handles one request: it recovers that request's write
# offset by summing the preceding extend lengths, then stores its row with a
# plain 1D pointer. The former `_seqlens_expand_grouped_kernel` broadcast a
# cumsum-derived row offset into a 2D store pointer, which tripped a FlagTree
# layout-propagation crash (TritonGPURemoveLayoutConversions cloneElementwise,
# `operandEnc` assertion) for every batch size, so it could never compile.
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

    _seqlens_expand_fused_kernel[(n,)](
        extend_seq_lens,
        seq_lens,
        out,
        n,
    )
    return out


__all__ = ["seqlens_expand"]
