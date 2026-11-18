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

"""Benchmark for activation/bias_gelu_tanh."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (rows, cols). The wide rows are the MLP projections where the fused
# bias + gelu_tanh pass is bandwidth bound.
SHAPES = [
    (4096, 11008),
    (512, 4096),
]

# The single-row shapes isolate launch cost and the non-power-of-two widths
# exercise the masked tail path.
MORE_SHAPES = [
    (1, 11008),
    (16, 14336),
    (128, 14336),
    (1, 4096),
]


def _input_fn(shape, cur_dtype, device):
    rows, cols = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        (rows, cols), generator=g, device=device, dtype=torch.float32
    )
    bias = 0.5 * torch.randn(cols, device=device, generator=g)
    yield x.to(cur_dtype), bias.to(cur_dtype)


def _case_fn(shape, cur_dtype):
    rows, cols = shape
    yield BenchmarkCasePlan(
        shape={"x": (rows, cols)},
        params={
            "configuration": shape,
            "rows": rows,
            "cols": cols,
        },
        builder_args=(shape,),
    )


@pytest.mark.bias_gelu_tanh
def test_perf_bias_gelu_tanh():
    bench = OpBenchmark(
        op_name="bias_gelu_tanh",
        torch_op=get_reference("bias_gelu_tanh"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="rows, cols",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
