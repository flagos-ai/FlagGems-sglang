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

"""Benchmark for attention/seqlens_expand."""

import pytest

import flaggems_sglang
from baseline import seqlens_expand as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    (n,) = shape
    yield BenchmarkCasePlan(
        shape={"cu_seqlens_q": (n + 1,)},
        params={"configuration": shape, "seed": 0, "n": n},
        builder_args=(shape,),
    )


@pytest.mark.seqlens_expand
def test_perf_seqlens_expand():
    bench = OpBenchmark(
        op_name="seqlens_expand",
        torch_op=get_reference("seqlens_expand"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="n",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
