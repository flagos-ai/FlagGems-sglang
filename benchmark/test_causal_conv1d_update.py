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

"""Benchmark for mamba/causal_conv1d_update."""

import pytest

import flaggems_sglang
from baseline import causal_conv1d_update as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/causal_conv1d_update.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    batch, dim, width, seqlen = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (batch, dim) if seqlen == 1 else (batch, dim, seqlen),
            "conv_state": (batch, dim, width - 1),
            "weight": (dim, width),
            "bias": (dim,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "batch": batch,
            "dim": dim,
            "width": width,
            "seqlen": seqlen,
        },
        builder_args=(shape,),
    )


@pytest.mark.causal_conv1d_update
def test_perf_causal_conv1d_update():
    bench = OpBenchmark(
        op_name="causal_conv1d_update",
        torch_op=get_reference("causal_conv1d_update"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="batch, dim, width, seqlen",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
