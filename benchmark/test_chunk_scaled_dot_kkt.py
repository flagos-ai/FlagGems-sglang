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

"""Benchmark for fla/chunk_scaled_dot_kkt."""

import pytest

import flaggems_sglang
from baseline import chunk_scaled_dot_kkt as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/fla/chunk_scaled_dot_kkt.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    batch, nchunks, chunk_size, hg, h, k_dim, use_g = shape
    yield BenchmarkCasePlan(
        shape={
            "k": (batch, nchunks * chunk_size, hg, k_dim),
            "beta": (batch, nchunks * chunk_size, h),
            "g_cumsum": (batch, nchunks * chunk_size, h) if use_g else None,
        },
        params={
            "configuration": shape,
            "seed": 0,
            "batch": batch,
            "nchunks": nchunks,
            "chunk_size": chunk_size,
            "hg": hg,
            "h": h,
            "k_dim": k_dim,
            "use_g": use_g,
        },
        builder_args=(shape,),
        baseline_key=workload.key_shape(workload.binding_for_case(shape)),
    )


@pytest.mark.chunk_scaled_dot_kkt
def test_perf_chunk_scaled_dot_kkt():
    bench = OpBenchmark(
        op_name="chunk_scaled_dot_kkt",
        torch_op=get_reference("chunk_scaled_dot_kkt"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="batch, nchunks, chunk_size, hg, h, k_dim, use_g",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
