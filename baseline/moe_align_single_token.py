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

"""moe_align_single_token baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.moe_align_single_token.moe_align_single_token。
复刻 problems/moe/moe_align_single_token/baseline.py：op(topk_ids, block_size)，
返回新张量，不原地 -> IS_INPLACE=False。

输入复刻 cases.py BENCH_CASES（k∈{1,4,8}, block∈{16,64,128}, num_experts=256）：
    topk_ids = randperm(num_experts)[:top_k].reshape(1,top_k).int32（单 token）。
"""

import torch

OP_NAME = "moe_align_single_token"
DTYPES = [torch.int32]
IS_INPLACE = False

_NUM_EXPERTS = 256
_SHAPES = [(k, block) for k in (1, 4, 8) for block in (16, 64, 128)]


def native():
    try:
        from sglang.kernels.ops.moe.moe_align_single_token import (
            moe_align_single_token as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"top_k": k, "num_experts": _NUM_EXPERTS, "block_size": b} for (k, b) in _SHAPES]


def build_inputs(binding, dtype, device):
    top_k, num_experts, block_size = (
        binding["top_k"], binding["num_experts"], binding["block_size"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    experts = torch.randperm(num_experts, device=device, generator=g)[:top_k]
    topk_ids = experts.reshape(1, top_k).to(torch.int32)
    return (topk_ids, block_size), {}


def key_shape(binding):
    return f"tk{binding['top_k']}_E{binding['num_experts']}_blk{binding['block_size']}"


def config(binding, dtype):
    top_k, num_experts, block_size = (
        binding["top_k"], binding["num_experts"], binding["block_size"],
    )
    return {
        "inputs": {
            "topk_ids": {"shape": [1, top_k], "dtype": "torch.int32"},
            "block_size": {"scalar": block_size},
        },
        "outputs": {
            "note": "sorted_token_ids / expert_ids / num_tokens_post_pad（单 token 对齐结果）",
        },
        "dims": {"top_k": top_k, "num_experts": num_experts, "block_size": block_size},
    }
