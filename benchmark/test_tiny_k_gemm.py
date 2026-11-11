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

"""Benchmark for gemm/tiny_k_gemm."""

import pytest

import flaggems_sglang
from baseline import tiny_k_gemm as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/gemm/tiny_k_gemm.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    m, n, k = shape
    yield BenchmarkCasePlan(
        shape={"x": (m, k), "w": (n, k)},
        params={"configuration": shape, "seed": 9, "m": m, "n": n, "k": k},
        builder_args=(shape,),
        baseline_key=workload.key_shape(workload.binding_for_case(shape)),
    )


@pytest.mark.tiny_k_gemm
def test_perf_tiny_k_gemm():
    bench = OpBenchmark(
        op_name="tiny_k_gemm",
        torch_op=get_reference("tiny_k_gemm"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="m, n, k",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
