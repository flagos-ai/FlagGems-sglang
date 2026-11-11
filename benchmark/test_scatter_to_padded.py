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

"""Benchmark for diffusion/scatter_to_padded."""

import pytest

import flaggems_sglang
from baseline import scatter_to_padded as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    batch, seqlen, heads, head_dim = shape
    yield BenchmarkCasePlan(
        shape={
            "src": (batch, seqlen, heads, head_dim),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "batch": batch,
            "seqlen": seqlen,
            "heads": heads,
            "head_dim": head_dim,
        },
        baseline_key=workload.key_shape(
            {
                "batch": batch,
                "seqlen": seqlen,
                "heads": heads,
                "head_dim": head_dim,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.scatter_to_padded
def test_perf_scatter_to_padded():
    bench = OpBenchmark(
        op_name="scatter_to_padded",
        torch_op=get_reference("scatter_to_padded"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="batch, seqlen, heads, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
