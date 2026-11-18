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

"""Benchmark for diffusion/indexed_scale_shift."""

import pytest

import flaggems_sglang
from baseline import indexed_scale_shift as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/diffusion/indexed_scale_shift.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    rows, hidden, variants = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (rows, hidden),
            "shift": (variants, hidden),
            "scale": (variants, hidden),
            "indices": (rows,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "rows": rows,
            "hidden": hidden,
            "variants": variants,
        },
        builder_args=(shape,),
    )


@pytest.mark.indexed_scale_shift
def test_perf_indexed_scale_shift():
    bench = OpBenchmark(
        op_name="indexed_scale_shift",
        torch_op=get_reference("indexed_scale_shift"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="rows, hidden, variants",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
