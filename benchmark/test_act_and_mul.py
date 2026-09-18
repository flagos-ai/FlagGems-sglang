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

"""Benchmark for moe/act_and_mul."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/act_and_mul.
SHAPES = [(m, 4096, "silu", None) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [
    (1, 37, "silu", None),
    (83, 1024, "gelu", None),
    (7, 512, "silu", 7.0),
    (256, 4096, "gelu", 10.0),
]


def _input_fn(shape, cur_dtype, device):
    m, half_hidden, activation, swiglu_limit = shape
    g = torch.Generator(device=device).manual_seed(0)
    # The op reads both halves from a single [m, 2 * half_hidden] tensor.
    gateup_output = torch.randn(
        m,
        half_hidden * 2,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    # ``activation`` is a str and ``swiglu_limit`` may be None-or-float;
    # a trailing dict keeps them out of the positional args.
    yield gateup_output, {
        "activation": activation,
        "swiglu_limit": swiglu_limit,
    }


@pytest.mark.act_and_mul
def test_perf_act_and_mul():
    bench = OpBenchmark(
        op_name="act_and_mul",
        torch_op=get_reference("act_and_mul"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, half_hidden, activation, swiglu_limit",
    )
    bench.set_gems(flaggems_sglang.act_and_mul)
    bench.run()
