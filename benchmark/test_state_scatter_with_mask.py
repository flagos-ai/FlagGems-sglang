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

"""Benchmark for mamba/state_scatter_with_mask."""

import pytest

import flaggems_sglang
from baseline import state_scatter_with_mask as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/state_scatter_with_mask.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    layers, cache, requests, draft, s0, s1 = shape
    yield BenchmarkCasePlan(
        shape={
            "dst": (layers, cache, s0, s1),
            "src": (layers, requests, draft, s0, s1),
            "dst_idx": (requests,),
            "step": (requests,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "layers": layers,
            "cache": cache,
            "requests": requests,
            "draft": draft,
            "s0": s0,
            "s1": s1,
        },
        baseline_key=workload.key_shape(
            {
                "layers": layers,
                "cache": cache,
                "requests": requests,
                "draft": draft,
                "s0": s0,
                "s1": s1,
                **{"state_shape": (s0, s1)},
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.state_scatter_with_mask
def test_perf_state_scatter_with_mask():
    bench = OpBenchmark(
        op_name="state_scatter_with_mask",
        torch_op=get_reference("state_scatter_with_mask"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="layers, cache, requests, draft, state0, state1",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
