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

"""moe_align_block_size baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.moe_align.moe_align_block_size（JIT C++ 算子，
本体即被采样对象，不可替换）。复刻 problems/moe/moe_align_block_size/baseline.py：
clone sorted_token_ids/expert_ids/num_tokens_post_pad/cumsum_buffer 后调
moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids, eids, npost,
cumsum_buffer, pad_sorted_token_ids)。多个缓冲被原地写回 -> IS_INPLACE=True。

输入复刻 cases.py BENCH_CASES（t∈{1,8,64,512,4096,16384}, top_k=8,
num_experts=256, block_size=128）。注意 kernel 以 num_experts+1 调用（EP filtered
expert 桶）；各缓冲尺寸按 cases.py 的 max_padded/max_blocks 公式分配。
"""

import torch
import triton

OP_NAME = "moe_align_block_size"
DTYPES = [torch.int32]
IS_INPLACE = True

_TOP_K = 8
_NUM_EXPERTS = 256
_BLOCK_SIZE = 128
_TS = [1, 8, 64, 512, 4096, 16384]


def native():
    try:
        from sglang.kernels.ops.moe.moe_align import moe_align_block_size as _f
    except Exception:
        return None

    def _op(topk_ids, num_experts, block_size, sorted_ids, eids, npost,
            cumsum_buffer, pad_sorted_token_ids):
        _f(topk_ids, num_experts, block_size, sorted_ids, eids, npost,
           cumsum_buffer, pad_sorted_token_ids)
        return sorted_ids, eids, npost

    return _op


def grid():
    return [
        {"num_tokens": t, "top_k": _TOP_K, "num_experts": _NUM_EXPERTS,
         "block_size": _BLOCK_SIZE}
        for t in _TS
    ]


def build_inputs(binding, dtype, device):
    num_tokens, top_k, num_experts, block_size = (
        binding["num_tokens"], binding["top_k"],
        binding["num_experts"], binding["block_size"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device, generator=g
    )
    numel = topk_ids.numel()
    arg_experts = num_experts + 1
    if numel < arg_experts + 1:
        max_padded = numel * block_size
    else:
        max_padded = numel + (arg_experts + 1) * (block_size - 1)
    max_blocks = triton.cdiv(max_padded, block_size)

    sorted_token_ids = torch.full((max_padded,), numel, dtype=torch.int32, device=device)
    expert_ids = torch.zeros(max_blocks, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.zeros(1, dtype=torch.int32, device=device)
    cumsum_buffer = torch.zeros(arg_experts + 1, dtype=torch.int32, device=device)
    # baseline 对前三者各 clone 一份做原地目标；cumsum_buffer 也以 clone 传入。
    return (
        topk_ids, arg_experts, block_size,
        sorted_token_ids.clone(), expert_ids.clone(), num_tokens_post_pad.clone(),
        cumsum_buffer.clone(), True,
    ), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_tk{binding['top_k']}_E{binding['num_experts']}_blk{binding['block_size']}"


def config(binding, dtype):
    num_tokens, top_k, num_experts, block_size = (
        binding["num_tokens"], binding["top_k"],
        binding["num_experts"], binding["block_size"],
    )
    numel = num_tokens * top_k
    arg_experts = num_experts + 1
    if numel < arg_experts + 1:
        max_padded = numel * block_size
    else:
        max_padded = numel + (arg_experts + 1) * (block_size - 1)
    max_blocks = triton.cdiv(max_padded, block_size)
    return {
        "inputs": {
            "topk_ids": {"shape": [num_tokens, top_k], "dtype": "torch.int32"},
            "num_experts": {"scalar": arg_experts},
            "block_size": {"scalar": block_size},
            "sorted_token_ids": {"shape": [max_padded], "dtype": "torch.int32", "note": "原地写回"},
            "expert_ids": {"shape": [max_blocks], "dtype": "torch.int32", "note": "原地写回"},
            "num_tokens_post_pad": {"shape": [1], "dtype": "torch.int32", "note": "原地写回"},
            "cumsum_buffer": {"shape": [arg_experts + 1], "dtype": "torch.int32"},
            "pad_sorted_token_ids": {"scalar": True},
        },
        "outputs": {
            "sorted_token_ids": {"shape": [max_padded], "dtype": "torch.int32"},
            "expert_ids": {"shape": [max_blocks], "dtype": "torch.int32"},
            "num_tokens_post_pad": {"shape": [1], "dtype": "torch.int32"},
        },
        "dims": {
            "num_tokens": num_tokens, "top_k": top_k, "num_experts": num_experts,
            "arg_experts": arg_experts, "block_size": block_size,
            "max_padded": max_padded, "max_blocks": max_blocks,
        },
    }
