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

"""Benchmark for moe/moe_topk_sum."""

import pytest

import flaggems_sglang
from baseline import moe_topk_sum as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/moe_topk_sum.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    m, topk, k = shape
    yield BenchmarkCasePlan(
        shape={"x": (m, topk, k), "out": (m, k)},
        params={
            "configuration": shape,
            "seed": 0,
            "m": m,
            "topk": topk,
            "k": k,
        },
        builder_args=(shape,),
    )


@pytest.mark.moe_topk_sum
def test_perf_moe_topk_sum():
    bench = OpBenchmark(
        op_name="moe_topk_sum",
        torch_op=get_reference("moe_topk_sum"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="m, topk, k",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
