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

"""Benchmark for quantization/hadamard_transform."""

import pytest

import flaggems_sglang
from baseline import hadamard_transform as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    m, n = plan.builder_args[0]
    args, kwargs = workload.build_inputs({"m": m, "n": n}, cur_dtype, device)
    return (*args, kwargs) if kwargs else args


def _case_fn(shape, cur_dtype):
    num_rows, hadamard_size = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (num_rows, hadamard_size),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_rows": num_rows,
            "hadamard_size": hadamard_size,
        },
        builder_args=(shape,),
    )


@pytest.mark.hadamard_transform
def test_perf_hadamard_transform():
    bench = OpBenchmark(
        op_name="hadamard_transform",
        torch_op=get_reference("hadamard_transform"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_rows, hadamard_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
