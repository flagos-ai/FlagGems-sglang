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

"""Benchmark for mamba/state_passing."""

import pytest

import flaggems_sglang
from baseline import state_passing as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/state_passing. The last
# entry is the has_init switch for the optional initial state.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    batch, nchunks, chunk_size, nheads, dim, has_init = shape
    yield BenchmarkCasePlan(
        shape={
            "states": (batch, nchunks, nheads, dim),
            "dA_cumsum": (batch, nheads, nchunks, chunk_size),
            "initial_states": (batch, nheads, dim) if has_init else None,
        },
        params={
            "configuration": shape,
            "seed": 0,
            "batch": batch,
            "nchunks": nchunks,
            "chunk_size": chunk_size,
            "nheads": nheads,
            "dim": dim,
            "has_init": has_init,
        },
        builder_args=(shape,),
    )


@pytest.mark.state_passing
def test_perf_state_passing():
    bench = OpBenchmark(
        op_name="state_passing",
        torch_op=get_reference("state_passing"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="batch, nchunks, chunk_size, nheads, dim, has_init",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
