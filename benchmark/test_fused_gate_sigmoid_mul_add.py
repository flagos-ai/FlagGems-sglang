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
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/activation_norm/
# fused_gate_sigmoid_mul_add.
SHAPES = [
    (bs, hidden) for bs in (1, 8, 64, 512) for hidden in (2048, 4096, 7168)
]
MORE_SHAPES = [(bs, hidden) for bs in (4096,) for hidden in (2048, 4096, 7168)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, hidden = shape

    def _rand(*size, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(
            *size, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    yield (
        _rand(num_tokens, hidden, seed=0),
        _rand(hidden, seed=1),
        _rand(num_tokens, hidden, seed=2),
        _rand(num_tokens, hidden, seed=3),
    )


def _case_fn(shape, cur_dtype):
    (num_tokens, hidden) = shape
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
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
