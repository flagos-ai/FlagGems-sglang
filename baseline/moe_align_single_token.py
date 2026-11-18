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

"""Native SGLang collection workload for moe_align_single_token."""

import torch

OP_NAME = "moe_align_single_token"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(k, 256, block) for k in (1, 4, 8) for block in (16, 64)]
MORE_SHAPES = [(k, 256, 128) for k in (1, 4, 8)]


def native():
    try:
        from sglang.kernels.ops.moe.moe_align_single_token import (
            moe_align_single_token as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"top_k": top_k, "num_experts": num_experts, "block_size": block_size}
        for (top_k, num_experts, block_size) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    top_k, num_experts, block_size = shape
    # The experts of the single token are distinct, which randperm guarantees.
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    g = torch.Generator(device="cpu").manual_seed(0)
    experts = torch.randperm(num_experts, generator=g)[:top_k].to(device)
    return experts.reshape(1, top_k).to(torch.int32), block_size


def build_inputs(binding, dtype, device):
    shape = (binding["top_k"], binding["num_experts"], binding["block_size"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return f"tk{binding['top_k']}_E{binding['num_experts']}_blk{binding['block_size']}"


def config(binding, dtype):
    top_k, num_experts, block_size = (
        binding["top_k"],
        binding["num_experts"],
        binding["block_size"],
    )
    return {
        "inputs": {
            "topk_ids": {"shape": [1, top_k], "dtype": "torch.int32"},
            "block_size": {"scalar": block_size},
        },
        "outputs": {
            "note": "sorted_token_ids / expert_ids / num_tokens_post_pad（单 token 对齐结果）",
        },
        "dims": {
            "top_k": top_k,
            "num_experts": num_experts,
            "block_size": block_size,
        },
    }
