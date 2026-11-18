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

"""Benchmark for activation/sigmoid_gate_mul."""

import pytest

import flaggems_sglang
from baseline import sigmoid_gate_mul as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, hidden = shape
    yield BenchmarkCasePlan(
        shape={"gate": (bs, hidden), "up": (bs, hidden)},
        params={"configuration": shape, "seed": 0, "bs": bs, "hidden": hidden},
        builder_args=(shape,),
    )


@pytest.mark.sigmoid_gate_mul
def test_perf_sigmoid_gate_mul():
    bench = OpBenchmark(
        op_name="sigmoid_gate_mul",
        torch_op=get_reference("sigmoid_gate_mul"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
