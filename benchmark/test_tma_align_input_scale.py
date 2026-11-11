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

"""Benchmark for moe/tma_align_input_scale."""

import pytest

import flaggems_sglang
from baseline import tma_align_input_scale as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/tma_align_input_scale.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    m, k_div_block = shape
    yield BenchmarkCasePlan(
        shape={"input_scale": (m, k_div_block)},
        params={
            "configuration": shape,
            "seed": 9,
            "m": m,
            "k_div_block": k_div_block,
        },
        builder_args=(shape,),
    )


@pytest.mark.tma_align_input_scale
def test_perf_tma_align_input_scale():
    bench = OpBenchmark(
        op_name="tma_align_input_scale",
        torch_op=get_reference("tma_align_input_scale"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="m, k_div_block",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
