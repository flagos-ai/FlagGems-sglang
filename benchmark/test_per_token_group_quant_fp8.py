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

"""Benchmark for per_token_group_quant_fp8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (rows, hidden, group_size)
SHAPES = [
    (1, 7168, 128),
    (32, 7168, 128),
    (256, 2048, 128),
]

# The extremes: a single group, the smallest supported group, the largest
# supported group, and a non-power-of-two group that takes the masked branch.
MORE_SHAPES = [
    (1, 128, 128),
    (64, 1024, 16),
    (128, 4096, 256),
    (8, 96, 48),
]


def _input_fn(shape, cur_dtype, device):
    rows, hidden, group_size = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        (rows, hidden), generator=g, device=device, dtype=cur_dtype
    )
    yield x, group_size


def _case_fn(shape, cur_dtype):
    rows, hidden, group_size = shape
    yield BenchmarkCasePlan(
        shape={"x": (rows, hidden)},
        params={
            "configuration": shape,
            "rows": rows,
            "hidden": hidden,
            "group_size": group_size,
        },
        builder_args=(shape,),
    )


@pytest.mark.per_token_group_quant_fp8
def test_perf_per_token_group_quant_fp8():
    bench = OpBenchmark(
        op_name="per_token_group_quant_fp8",
        torch_op=get_reference("per_token_group_quant_fp8"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="rows, hidden, group_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
