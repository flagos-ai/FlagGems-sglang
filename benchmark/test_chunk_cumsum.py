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

"""Benchmark for mamba/chunk_cumsum."""

import pytest

import flaggems_sglang
from baseline import chunk_cumsum as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/chunk_cumsum; every case
# benchmarks the fully-featured path (dt_bias + dt_softplus).


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    batch, nchunks, chunk_size, nheads = shape
    yield BenchmarkCasePlan(
        shape={
            "dt": (batch, nchunks * chunk_size, nheads),
            "a": (nheads,),
            "dt_bias": (nheads,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "batch": batch,
            "nchunks": nchunks,
            "chunk_size": chunk_size,
            "nheads": nheads,
        },
        builder_args=(shape,),
    )


@pytest.mark.chunk_cumsum
def test_perf_chunk_cumsum():
    bench = OpBenchmark(
        op_name="chunk_cumsum",
        torch_op=get_reference("chunk_cumsum"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="batch, nchunks, chunk_size, nheads",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
