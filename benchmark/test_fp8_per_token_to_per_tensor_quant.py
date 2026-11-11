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

"""Benchmark for moe/fp8_per_token_to_per_tensor_quant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

GROUP = 128

# Entries are (hidden, num_experts, m), taken from the upstream registered
# benchmark: DeepSeek-V3 at EP8, then 3584 hidden at a low and a high local
# expert count.
SHAPES = [
    (7168, 8, 1024),
    (3584, 8, 1024),
    (3584, 56, 256),
]
MORE_SHAPES = [
    (3584, 1, 16),
    (3584, 4, 48),
    (1152, 4, 48),
]


def _row_counts(num_experts, m):
    """Every expert gets a different live count, as a real dispatch would."""
    return [(e * 7) % (m + 1) for e in range(num_experts)]


def _input_fn(shape, cur_dtype, device):
    hidden, num_experts, m = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(num_experts, m, hidden, generator=g, device=device).to(
        cur_dtype
    )
    # DeepEP hands the last two scale dims over column-major, for TMA.
    x_scale = (
        torch.rand(num_experts, m, hidden // GROUP, generator=g, device=device)
        .add_(0.5)
        .permute(0, 2, 1)
        .contiguous()
        .permute(0, 2, 1)
    )
    masked_m = torch.tensor(
        _row_counts(num_experts, m), dtype=torch.int32, device=device
    )
    output_scale = torch.tensor([2.0], dtype=torch.float32, device=device)
    yield x, x_scale, masked_m, output_scale


def _case_fn(shape, cur_dtype):
    hidden, num_experts, m = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (num_experts, m, hidden),
            "x_scale": (num_experts, m, hidden // GROUP),
            "masked_m": (num_experts,),
            "output_scale": (1,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "hidden": hidden,
            "num_experts": num_experts,
            "m": m,
        },
        builder_args=(shape,),
    )


@pytest.mark.fp8_per_token_to_per_tensor_quant
def test_perf_fp8_per_token_to_per_tensor_quant():
    bench = OpBenchmark(
        op_name="fp8_per_token_to_per_tensor_quant",
        torch_op=get_reference("fp8_per_token_to_per_tensor_quant"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.float8_e4m3fn],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="hidden, num_experts, m",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
