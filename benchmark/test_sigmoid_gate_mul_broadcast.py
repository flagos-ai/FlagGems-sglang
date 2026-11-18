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

"""Benchmark for elementwise/sigmoid_gate_mul_broadcast."""

import pytest

import flaggems_sglang
from baseline import sigmoid_gate_mul_broadcast as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/elementwise/
# sigmoid_gate_mul_broadcast.


def _build_inputs(plan, cur_dtype, device):
    bs, hidden = plan.builder_args[0]
    args, kwargs = workload.build_inputs(
        {"bs": bs, "hidden": hidden}, cur_dtype, device
    )
    return (*args, kwargs) if kwargs else args


def _case_fn(shape, cur_dtype):
    num_tokens, hidden = shape
    yield BenchmarkCasePlan(
        shape={"gate": (num_tokens, hidden), "up": (num_tokens, 1)},
        params={
            "configuration": shape,
            "seeds": [0, 1],
            "num_tokens": num_tokens,
            "hidden": hidden,
        },
        builder_args=(shape,),
    )


@pytest.mark.sigmoid_gate_mul_broadcast
def test_perf_sigmoid_gate_mul_broadcast():
    bench = OpBenchmark(
        op_name="sigmoid_gate_mul_broadcast",
        torch_op=get_reference("sigmoid_gate_mul_broadcast"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
