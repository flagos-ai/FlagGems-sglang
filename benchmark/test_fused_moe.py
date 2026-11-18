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

"""Benchmark for moe/fused_moe."""

import pytest

import flaggems_sglang
from baseline import fused_moe as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# (num_tokens, hidden, inter, num_experts, topk).


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, hidden, inter, num_experts, topk = shape
    yield BenchmarkCasePlan(
        shape={
            "hidden_states": (num_tokens, hidden),
            "w1": (num_experts, 2 * inter, hidden),
            "w2": (num_experts, hidden, inter),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "hidden": hidden,
            "inter": inter,
            "num_experts": num_experts,
            "topk": topk,
        },
        builder_args=(shape,),
        baseline_key=workload.key_shape(workload.binding_for_case(shape)),
    )


@pytest.mark.fused_moe
def test_perf_fused_moe():
    bench = OpBenchmark(
        op_name="fused_moe",
        torch_op=get_reference("fused_moe"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, hidden, inter, num_experts, topk",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
