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

"""Benchmark for quantization/per_group_transpose."""

import pytest

import flaggems_sglang
from baseline import per_group_transpose as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/quantization/per_group_transpose;
# every expert gets ``rows_per_expert`` rows.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    k, rows_per_expert = shape
    yield BenchmarkCasePlan(
        shape={
            "a": (rows_per_expert * workload.NUM_EXPERTS, k),
            "expert_offsets": (workload.NUM_EXPERTS + 1,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "NUM_EXPERTS": workload.NUM_EXPERTS,
            "k": k,
            "rows_per_expert": rows_per_expert,
        },
        builder_args=(shape,),
    )


@pytest.mark.per_group_transpose
def test_perf_per_group_transpose():
    bench = OpBenchmark(
        op_name="per_group_transpose",
        torch_op=get_reference("per_group_transpose"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        shape_desc="k, rows_per_expert",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
