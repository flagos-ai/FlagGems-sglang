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

"""Benchmark for activation_norm/wan_rmsnorm_silu."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (batch, channels, seq_len) flattened Wan feature maps.
SHAPES = [
    (16, 128, 2048),
    (8, 512, 1024),
    (1, 5120, 512),
]

MORE_SHAPES = [
    (1, 2048, 2048),
    (64, 128, 512),
    (1, 512, 4096),
]


def _input_fn(shape, cur_dtype, device):
    batch, channels, seq = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        (batch, channels, seq),
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    gamma = (
        0.5
        * torch.randn(
            channels, generator=g, device=device, dtype=torch.float32
        )
        + 1.0
    ).to(cur_dtype)
    bias = (
        0.25
        * torch.randn(
            channels, generator=g, device=device, dtype=torch.float32
        )
    ).to(cur_dtype)
    yield x, gamma, bias, 1.0, 1e-6


def _case_fn(shape, cur_dtype):
    batch, channels, seq = shape
    yield BenchmarkCasePlan(
        shape={"x": (batch, channels, seq)},
        params={
            "configuration": shape,
            "batch": batch,
            "channels": channels,
            "seq": seq,
        },
        builder_args=(shape,),
    )


@pytest.mark.wan_rmsnorm_silu
def test_perf_wan_rmsnorm_silu():
    bench = OpBenchmark(
        op_name="wan_rmsnorm_silu",
        torch_op=get_reference("wan_rmsnorm_silu"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, channels, seq",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
