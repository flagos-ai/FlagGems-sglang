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

import flaggems_sglang
from baseline import moe_align_block_size as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/moe_align_block_size.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, top_k, num_experts, block_size = shape
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
        baseline_key=workload.key_shape(
            {
                "num_tokens": num_tokens,
                "top_k": top_k,
                "num_experts": num_experts,
                "block_size": block_size,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.moe_align_block_size
def test_perf_moe_align_block_size():
    bench = OpBenchmark(
        op_name="moe_align_block_size",
        torch_op=get_reference("moe_align_block_size"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, top_k, num_experts, block_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
