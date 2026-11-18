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

"""Benchmark for moe/fused_moe_gemm."""

import pytest

import flaggems_sglang
from baseline import fused_moe_gemm as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/fused_moe_gemm.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    T, E, N, K, top_k = shape
    yield BenchmarkCasePlan(
        shape={
            "A": (T, K),
            "B": (E, N, K),
            "topk_weights": (T, top_k),
            "topk_ids": (T, top_k),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "T": T,
            "E": E,
            "N": N,
            "K": K,
            "top_k": top_k,
        },
        baseline_key=workload.key_shape(
            {
                "T": shape[0],
                "E": shape[1],
                "N": shape[2],
                "K": shape[3],
                "top_k": shape[4],
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.fused_moe_gemm
def test_perf_fused_moe_gemm():
    bench = OpBenchmark(
        op_name="fused_moe_gemm",
        torch_op=get_reference("fused_moe_gemm"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=(),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="T, E, N, K, top_k",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
