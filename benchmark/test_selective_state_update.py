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

"""Benchmark for mamba/selective_state_update."""

import pytest

import flaggems_sglang
from baseline import selective_state_update as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/selective_state_update.
# The trailing entries are ngroups, then the has_z / has_d / has_dt_bias /
# dt_softplus switches.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    (
        batch,
        nheads,
        dim,
        dstate,
        ngroups,
        has_z,
        has_d,
        has_dt_bias,
        dt_softplus,
    ) = shape
    yield BenchmarkCasePlan(
        shape={
            "state": (batch, nheads, dim, dstate),
            "x": (batch, nheads, dim),
            "dt": (batch, nheads, dim),
            "a": (nheads, dim, dstate),
            "b": (batch, ngroups, dstate),
            "c": (batch, ngroups, dstate),
            "D": (nheads, dim) if has_d else None,
            "z": (batch, nheads, dim) if has_z else None,
            "dt_bias": (nheads, dim) if has_dt_bias else None,
        },
        params={
            "configuration": shape,
            "seed": 0,
            "batch": batch,
            "nheads": nheads,
            "dim": dim,
            "dstate": dstate,
            "ngroups": ngroups,
            "has_z": has_z,
            "has_d": has_d,
            "has_dt_bias": has_dt_bias,
            "dt_softplus": dt_softplus,
        },
        baseline_key=workload.key_shape(
            {
                "batch": batch,
                "nheads": nheads,
                "dim": dim,
                "dstate": dstate,
                "ngroups": ngroups,
                "has_z": has_z,
                "has_d": has_d,
                "has_dt_bias": has_dt_bias,
                "dt_softplus": dt_softplus,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.selective_state_update
def test_perf_selective_state_update():
    bench = OpBenchmark(
        op_name="selective_state_update",
        torch_op=get_reference("selective_state_update"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=(0,),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc=(
            "batch, nheads, dim, dstate, ngroups, "
            "has_z, has_d, has_dt_bias, dt_softplus"
        ),
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
