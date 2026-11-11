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

"""Benchmark for moe/gate_topk."""

import pytest

import flaggems_sglang
from baseline import gate_topk as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    m, n, k = shape
    yield BenchmarkCasePlan(
        shape={"x": (m, n)},
        params={"configuration": shape, "seed": 0, "m": m, "n": n, "k": k},
        baseline_key=workload.key_shape({"m": m, "n": n, "k": k}),
        builder_args=(shape,),
    )


@pytest.mark.gate_topk
def test_perf_gate_topk():
    bench = OpBenchmark(
        op_name="gate_topk",
        torch_op=get_reference("gate_topk"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="m, n, k",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
