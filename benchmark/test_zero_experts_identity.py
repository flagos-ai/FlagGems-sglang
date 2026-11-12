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

"""Benchmark for moe/zero_experts_identity."""

import pytest

import flaggems_sglang
from baseline import zero_experts_identity as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/zero_experts_identity.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, top_k, hidden, num_experts = shape
    yield BenchmarkCasePlan(
        shape={
            "idx": (num_tokens, top_k),
            "scales": (num_tokens, top_k),
            "hidden_states": (num_tokens, hidden),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_NUM_ZERO": workload._NUM_ZERO,
            "num_tokens": num_tokens,
            "top_k": top_k,
            "hidden": hidden,
            "num_experts": num_experts,
        },
        baseline_key=workload.key_shape(
            {
                "num_tokens": num_tokens,
                "top_k": top_k,
                "hidden": hidden,
                "num_experts": num_experts,
                "num_zero": workload._NUM_ZERO,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.zero_experts_identity
def test_perf_zero_experts_identity():
    bench = OpBenchmark(
        op_name="zero_experts_identity",
        torch_op=get_reference("zero_experts_identity"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, top_k, hidden, num_experts",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
