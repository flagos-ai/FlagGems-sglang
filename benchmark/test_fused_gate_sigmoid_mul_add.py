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

"""Benchmark for activation_norm/fused_gate_sigmoid_mul_add."""

import pytest

import flaggems_sglang
from baseline import fused_gate_sigmoid_mul_add as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/activation_norm/
# fused_gate_sigmoid_mul_add.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, hidden = shape
    yield BenchmarkCasePlan(
        shape={
            "hidden_states": (num_tokens, hidden),
            "gate_weight": (hidden,),
            "shared_output": (num_tokens, hidden),
            "final_hidden_states": (num_tokens, hidden),
        },
        params={
            "configuration": shape,
            "seeds": [0, 1, 2, 3],
            "num_tokens": num_tokens,
            "hidden": hidden,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_gate_sigmoid_mul_add
def test_perf_fused_gate_sigmoid_mul_add():
    bench = OpBenchmark(
        op_name="fused_gate_sigmoid_mul_add",
        torch_op=get_reference("fused_gate_sigmoid_mul_add"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
