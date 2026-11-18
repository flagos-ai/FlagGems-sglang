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

"""Benchmark for moe/moe_align_single_token."""

import pytest

import flaggems_sglang
from baseline import moe_align_single_token as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/moe_align_single_token.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    top_k, num_experts, block_size = shape
    yield BenchmarkCasePlan(
        shape={"experts": (1, top_k)},
        params={
            "configuration": shape,
            "seed": 0,
            "top_k": top_k,
            "num_experts": num_experts,
            "block_size": block_size,
        },
        baseline_key=workload.key_shape(
            {
                "top_k": top_k,
                "num_experts": num_experts,
                "block_size": block_size,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.moe_align_single_token
def test_perf_moe_align_single_token():
    bench = OpBenchmark(
        op_name="moe_align_single_token",
        torch_op=get_reference("moe_align_single_token"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="top_k, num_experts, block_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
