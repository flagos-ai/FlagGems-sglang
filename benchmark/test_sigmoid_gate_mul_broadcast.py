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
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/elementwise/
# sigmoid_gate_mul_broadcast.
SHAPES = [
    (bs, hidden) for bs in (1, 8, 64, 512) for hidden in (1024, 4096, 8192)
]
MORE_SHAPES = [(bs, hidden) for bs in (4096,) for hidden in (1024, 4096, 8192)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, hidden = shape

    def _rand(*size, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(
            *size, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    yield _rand(num_tokens, hidden, seed=0), _rand(num_tokens, 1, seed=1)


def _case_fn(shape, cur_dtype):
    (num_tokens, hidden) = shape
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
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
