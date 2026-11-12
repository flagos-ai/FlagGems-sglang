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

"""Benchmark for fla/fused_gdn_gating."""

import pytest

import flaggems_sglang
from baseline import fused_gdn_gating as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/fla/fused_gdn_gating.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    batch, num_heads = shape
    yield BenchmarkCasePlan(
        shape={
            "a_log": (num_heads,),
            "a": (batch, num_heads),
            "b": (batch, num_heads),
            "dt_bias": (num_heads,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "batch": batch,
            "num_heads": num_heads,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_gdn_gating
def test_perf_fused_gdn_gating():
    bench = OpBenchmark(
        op_name="fused_gdn_gating",
        torch_op=get_reference("fused_gdn_gating"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="batch, num_heads",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
