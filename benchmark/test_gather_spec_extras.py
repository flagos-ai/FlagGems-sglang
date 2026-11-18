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

"""Benchmark for speculative/gather_spec_extras."""

import pytest

import flaggems_sglang
from baseline import gather_spec_extras as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/speculative/gather_spec_extras.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    m, hidden = shape
    yield BenchmarkCasePlan(
        shape={
            "indices": (m,),
            "topk_p_buf": (workload._POOL, workload._TOPK),
            "topk_index_buf": (workload._POOL, workload._TOPK),
            "output_tokens_buf": (workload._POOL,),
            "hidden_states_buf": (workload._POOL, hidden),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_POOL": workload._POOL,
            "_TOPK": workload._TOPK,
            "m": m,
            "hidden": hidden,
        },
        builder_args=(shape,),
    )


@pytest.mark.gather_spec_extras
def test_perf_gather_spec_extras():
    bench = OpBenchmark(
        op_name="gather_spec_extras",
        torch_op=get_reference("gather_spec_extras"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="m, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
