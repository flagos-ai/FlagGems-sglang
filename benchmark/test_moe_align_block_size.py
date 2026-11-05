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

"""Benchmark for moe/moe_align_block_size."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/moe_align_block_size.
SHAPES = [(t, 8, 256, 128) for t in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(t, 8, 256, 128) for t in (16384,)]


def _input_fn(shape, cur_dtype, device):
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
    yield (
        topk_ids,
        arg_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        {"pad_sorted_token_ids": True},
    )


def _case_fn(shape, cur_dtype):
    (num_tokens, top_k, num_experts, block_size) = shape
    numel = num_tokens * top_k
    max_padded = (
        numel * block_size
        if numel < num_experts + 2
        else numel + (num_experts + 2) * (block_size - 1)
    )
    yield BenchmarkCasePlan(
        shape={
            "topk_ids": (num_tokens, top_k),
            "sorted_token_ids": (max_padded,),
            "expert_ids": ((max_padded + block_size - 1) // block_size,),
            "num_tokens_post_pad": (1,),
            "cumsum_buffer": (num_experts + 2,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "top_k": top_k,
            "num_experts": num_experts,
            "block_size": block_size,
            "numel": numel,
            "max_padded": max_padded,
        },
        builder_args=(shape,),
    )


@pytest.mark.moe_align_block_size
def test_perf_moe_align_block_size():
    bench = OpBenchmark(
        op_name="moe_align_block_size",
        torch_op=get_reference("moe_align_block_size"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, top_k, num_experts, block_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
