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

"""Benchmark for fla/chunk_local_cumsum_scalar."""

from functools import partial

import pytest

import flaggems_sglang
from baseline import chunk_local_cumsum_scalar as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/fla/chunk_local_cumsum_scalar.


def _case_fn(shape, cur_dtype, reverse, scale=None):
    batch, nchunks, chunk_size, nheads = shape
    yield BenchmarkCasePlan(
        shape={"x": (batch, nchunks * chunk_size, nheads)},
        params={
            "configuration": shape,
            "seed": 0,
            "reverse": reverse,
            "scale": scale,
            "batch": batch,
            "nchunks": nchunks,
            "chunk_size": chunk_size,
            "nheads": nheads,
        },
        builder_args=(shape,),
        baseline_key=workload.key_shape(
            workload.binding_for_case(shape, reverse)
        ),
    )


@pytest.mark.chunk_local_cumsum_scalar
@pytest.mark.parametrize("reverse", [False, True])
def test_perf_chunk_local_cumsum_scalar(reverse):
    bench = OpBenchmark(
        op_name="chunk_local_cumsum_scalar",
        torch_op=get_reference("chunk_local_cumsum_scalar"),
        case_fn=partial(_case_fn, reverse=reverse),
        build_inputs_fn=single_input_builder(
            partial(workload.input_cases, reverse=reverse)
        ),
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="batch, nchunks, chunk_size, nheads",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
