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

"""Benchmark for norm/group_norm_silu."""

import pytest

import flaggems_sglang
from baseline import group_norm_silu as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    n, c, spatial, num_groups = shape
    yield BenchmarkCasePlan(
        shape={"x": (n, c, spatial, spatial), "w": (c,), "b": (c,)},
        params={
            "configuration": shape,
            "seed": 0,
            "n": n,
            "c": c,
            "spatial": spatial,
            "num_groups": num_groups,
        },
        builder_args=(shape,),
    )


@pytest.mark.group_norm_silu
def test_perf_group_norm_silu():
    bench = OpBenchmark(
        op_name="group_norm_silu",
        torch_op=get_reference("group_norm_silu"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="n, c, spatial, num_groups",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
