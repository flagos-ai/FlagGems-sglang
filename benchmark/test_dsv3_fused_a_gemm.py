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

"""Benchmark for moe/dsv3_fused_a_gemm."""

import pytest

import flaggems_sglang
from baseline import dsv3_fused_a_gemm as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    num_tokens, hd_in, hd_out = plan.builder_args[0]
    return workload.build_inputs(
        {"num_tokens": num_tokens, "hd_in": hd_in, "hd_out": hd_out},
        cur_dtype,
        device,
    )[0]


def _case_fn(shape, cur_dtype):
    m, hd_in, hd_out = shape
    yield BenchmarkCasePlan(
        shape={
            "a": (m, hd_in),
            "b": (hd_in, hd_out),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "m": m,
            "hd_in": hd_in,
            "hd_out": hd_out,
        },
        builder_args=(shape,),
    )


@pytest.mark.dsv3_fused_a_gemm
def test_perf_dsv3_fused_a_gemm():
    bench = OpBenchmark(
        op_name="dsv3_fused_a_gemm",
        torch_op=get_reference("dsv3_fused_a_gemm"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="m, hd_in, hd_out",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
