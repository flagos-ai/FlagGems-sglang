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

"""Benchmark for moe/moe_sum_reduce."""

import pytest

import flaggems_sglang
from baseline import moe_sum_reduce as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/moe_sum_reduce.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    T, top_k, H = shape
    yield BenchmarkCasePlan(
        shape={"x": (T, top_k, H)},
        params={
            "configuration": shape,
            "seed": 0,
            "ROUTED_SCALING_FACTOR": workload.ROUTED_SCALING_FACTOR,
            "T": T,
            "top_k": top_k,
            "H": H,
        },
        builder_args=(shape,),
    )


@pytest.mark.moe_sum_reduce
def test_perf_moe_sum_reduce():
    bench = OpBenchmark(
        op_name="moe_sum_reduce",
        torch_op=get_reference("moe_sum_reduce"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=(),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="T, top_k, H",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
