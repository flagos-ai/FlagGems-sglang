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

"""Benchmark for mamba/bmm_chunk."""

import pytest

import flaggems_sglang
from baseline import bmm_chunk as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/bmm_chunk.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    batch, nchunks, chunk_size, ngroups, k = shape
    yield BenchmarkCasePlan(
        shape={
            "a": (batch, nchunks * chunk_size, ngroups, k),
            "b": (batch, nchunks * chunk_size, ngroups, k),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "CAUSAL": workload.CAUSAL,
            "batch": batch,
            "nchunks": nchunks,
            "chunk_size": chunk_size,
            "ngroups": ngroups,
            "k": k,
        },
        builder_args=(shape,),
    )


@pytest.mark.bmm_chunk
def test_perf_bmm_chunk():
    bench = OpBenchmark(
        op_name="bmm_chunk",
        torch_op=get_reference("bmm_chunk"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="batch, nchunks, chunk_size, ngroups, k",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
