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

"""Benchmark for activation_norm/grouped_gemma_rmsnorm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (batch, hidden, group_size). gpt-oss-like hidden sizes with a per-group
# weight layout; the wide-row cases exercise the streaming path.
SHAPES = [
    (1, 2880, 160),
    (64, 2880, 160),
    (4096, 2880, 160),
]

MORE_SHAPES = [
    (512, 4096, 128),
    (4096, 4096, 128),
    (2048, 1024, 64),
    (256, 8192, 256),
]


def _input_fn(shape, cur_dtype, device):
    batch, hidden, group_size = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        (batch, hidden), generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    weight = (
        0.5
        * torch.randn(
            hidden, generator=g, device=device, dtype=torch.float32
        )
    ).to(cur_dtype)
    yield x, weight, group_size, 1e-6


def _case_fn(shape, cur_dtype):
    batch, hidden, group_size = shape
    yield BenchmarkCasePlan(
        shape={"x": (batch, hidden)},
        params={
            "configuration": shape,
            "batch": batch,
            "hidden": hidden,
            "group_size": group_size,
        },
        builder_args=(shape,),
    )


@pytest.mark.grouped_gemma_rmsnorm
def test_perf_grouped_gemma_rmsnorm():
    bench = OpBenchmark(
        op_name="grouped_gemma_rmsnorm",
        torch_op=get_reference("grouped_gemma_rmsnorm"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, hidden, group_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
