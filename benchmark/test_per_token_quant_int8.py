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

"""Benchmark for quantization/per_token_quant_int8."""

import pytest

import flaggems_sglang
from baseline import per_token_quant_int8 as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark, is_case_listing

# Shapes match kernel-comp-baseline/problems/quantization/
# per_token_quant_int8.


def _build_inputs(plan, cur_dtype, device):
    m, k = plan.builder_args[0]
    args, kwargs = workload.build_inputs({"m": m, "k": k}, cur_dtype, device)
    return (*args, kwargs) if kwargs else args


def _case_fn(shape, cur_dtype):
    m, k = shape
    yield BenchmarkCasePlan(
        shape={"x": (m, k)},
        params={"configuration": shape, "seed": 0, "m": m, "k": k},
        builder_args=(shape,),
    )


@pytest.mark.per_token_quant_int8
def test_perf_per_token_quant_int8():
    # This op has a reference but no Triton implementation yet.
    gems_op = None
    if not is_case_listing():
        gems_op = flaggems_sglang.get_op("per_token_quant_int8")
        if gems_op is None:
            pytest.skip(
                "quantization/per_token_quant_int8 not implemented yet"
            )
    bench = OpBenchmark(
        op_name="per_token_quant_int8",
        torch_op=get_reference("per_token_quant_int8"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="m, k",
    )
    bench.set_gems(gems_op)
    bench.run()
