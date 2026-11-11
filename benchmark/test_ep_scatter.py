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

"""Benchmark for moe/ep_scatter."""

import pytest

import flaggems_sglang
from baseline import ep_scatter as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, topk, hidden, num_experts = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (num_tokens, hidden),
            "topk_ids": (num_tokens, topk),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "topk": topk,
            "hidden": hidden,
            "num_experts": num_experts,
        },
        baseline_key=workload.key_shape(
            {
                "t": num_tokens,
                "topk": topk,
                "hidden": hidden,
                "num_experts": num_experts,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.ep_scatter
def test_perf_ep_scatter():
    bench = OpBenchmark(
        op_name="ep_scatter",
        torch_op=get_reference("ep_scatter"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, topk, hidden, num_experts",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
