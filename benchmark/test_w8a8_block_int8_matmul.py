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

"""Benchmark for quantization/w8a8_block_int8_matmul."""

import pytest

import flaggems_sglang
from baseline import w8a8_block_int8_matmul as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/quantization/w8a8_block_int8_matmul.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    m, n, k = shape
    yield BenchmarkCasePlan(
        shape={
            "A": (m, k),
            "B": (n, k),
            "As": (m, k // workload._BLOCK[1]),
            "Bs": (n // workload._BLOCK[0], k // workload._BLOCK[1]),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "_BLOCK": workload._BLOCK,
            "m": m,
            "n": n,
            "k": k,
        },
        builder_args=(shape,),
    )


@pytest.mark.w8a8_block_int8_matmul
def test_perf_w8a8_block_int8_matmul():
    bench = OpBenchmark(
        op_name="w8a8_block_int8_matmul",
        torch_op=get_reference("w8a8_block_int8_matmul"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="m, n, k",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
