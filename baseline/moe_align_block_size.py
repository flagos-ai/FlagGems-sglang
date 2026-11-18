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

"""Native SGLang collection workload for moe_align_block_size."""

import torch

OP_NAME = "moe_align_block_size"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

MUTATED_INPUTS = (3, 4, 5, 6)

SHAPES = [(t, 8, 256, 128) for t in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(t, 8, 256, 128) for t in (16384,)]


def native():
    try:
        from sglang.kernels.ops.moe.moe_align import moe_align_block_size as _f
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def _op(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        eids,
        npost,
        cumsum_buffer,
        pad_sorted_token_ids,
    ):
        _f(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            eids,
            npost,
            cumsum_buffer,
            pad_sorted_token_ids,
        )
        return sorted_ids, eids, npost

    return _op


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "top_k": top_k,
            "num_experts": num_experts,
            "block_size": block_size,
        }
        for (num_tokens, top_k, num_experts, block_size) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, top_k, num_experts, block_size = shape
    g = torch.Generator(device=device).manual_seed(9)
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    numel = topk_ids.numel()
    # The trailing "filtered expert" slot never receives tokens, matching how
    # moe_runner.triton_utils calls the kernel.
    arg_experts = num_experts + 1
    if numel < arg_experts + 1:
        max_padded = numel * block_size
    else:
        max_padded = numel + (arg_experts + 1) * (block_size - 1)
    max_blocks = (max_padded + block_size - 1) // block_size

    sorted_token_ids = torch.full(
        (max_padded,), numel, dtype=torch.int32, device=device
    )
    expert_ids = torch.zeros(max_blocks, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.zeros(1, dtype=torch.int32, device=device)
    cumsum_buffer = torch.zeros(
        arg_experts + 1, dtype=torch.int32, device=device
    )
    return (
        topk_ids,
        arg_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        {"pad_sorted_token_ids": True},
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_tokens"],
        binding["top_k"],
        binding["num_experts"],
        binding["block_size"],
    )
    *args, kwargs = build_case(shape, dtype, device)
    return (*args, kwargs["pad_sorted_token_ids"]), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_tk{binding['top_k']}_E{binding['num_experts']}_blk{binding['block_size']}"


def config(binding, dtype):
    num_tokens, top_k, num_experts, block_size = (
        binding["num_tokens"],
        binding["top_k"],
        binding["num_experts"],
        binding["block_size"],
    )
    numel = num_tokens * top_k
    arg_experts = num_experts + 1
    if numel < arg_experts + 1:
        max_padded = numel * block_size
    else:
        max_padded = numel + (arg_experts + 1) * (block_size - 1)
    max_blocks = (max_padded + block_size - 1) // block_size
    return {
        "inputs": {
            "topk_ids": {"shape": [num_tokens, top_k], "dtype": "torch.int32"},
            "num_experts": {"scalar": arg_experts},
            "block_size": {"scalar": block_size},
            "sorted_token_ids": {
                "shape": [max_padded],
                "dtype": "torch.int32",
                "note": "原地写回",
            },
            "expert_ids": {
                "shape": [max_blocks],
                "dtype": "torch.int32",
                "note": "原地写回",
            },
            "num_tokens_post_pad": {
                "shape": [1],
                "dtype": "torch.int32",
                "note": "原地写回",
            },
            "cumsum_buffer": {
                "shape": [arg_experts + 1],
                "dtype": "torch.int32",
            },
            "pad_sorted_token_ids": {"scalar": True},
        },
        "outputs": {
            "sorted_token_ids": {
                "shape": [max_padded],
                "dtype": "torch.int32",
            },
            "expert_ids": {"shape": [max_blocks], "dtype": "torch.int32"},
            "num_tokens_post_pad": {"shape": [1], "dtype": "torch.int32"},
        },
        "dims": {
            "num_tokens": num_tokens,
            "top_k": top_k,
            "num_experts": num_experts,
            "arg_experts": arg_experts,
            "block_size": block_size,
            "max_padded": max_padded,
            "max_blocks": max_blocks,
        },
    }
