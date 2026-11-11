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

"""Benchmark for attention/log_scaling_tau."""

import pytest

import flaggems_sglang
from baseline import log_scaling_tau as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    t, heads, d = plan.builder_args[0]
    return workload.build_inputs(
        {"t": t, "heads": heads, "d": d}, cur_dtype, device
    )[0]


def _case_fn(shape, cur_dtype):
    yield BenchmarkCasePlan(
        shape={
            "x": tuple(
                d for i, d in enumerate(shape) if not (i == 1 and d == 1)
            ),
            "tau": (shape[0],),
        },
        params={"configuration": shape, "seed": 0},
        builder_args=(shape,),
    )


@pytest.mark.log_scaling_tau
def test_perf_log_scaling_tau():
    bench = OpBenchmark(
        op_name="log_scaling_tau",
        torch_op=get_reference("log_scaling_tau"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="t, heads, d",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
