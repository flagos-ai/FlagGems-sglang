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

"""Benchmark for embedding/timestep_embedding."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (n, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period).
SHAPES = [
    (2048, 1280, True, 0.0, 1.0, 10000),
    (1024, 512, True, 0.0, 1.0, 10000),
]

# The odd half sizes and the shifted/scaled variants exercise the frequency
# exponent path, and the tiny row count isolates the launch cost.
MORE_SHAPES = [
    (128, 1280, True, 1.0, 10.0, 10000),
    (16, 16, False, 0.0, 1000.0, 100),
    (64, 256, False, 0.0, 1.0, 10000),
    (1, 32, True, 0.0, 1.0, 10000),
]


def _input_fn(shape, cur_dtype, device):
    n, dim, flip, shift, scale, max_period = shape
    g = torch.Generator(device=device).manual_seed(0)
    t = 1000.0 * torch.rand(n, device=device, generator=g)
    yield (
        t.to(cur_dtype),
        dim,
        flip,
        shift,
        scale,
        max_period,
    )


def _case_fn(shape, cur_dtype):
    n, dim, flip, shift, scale, max_period = shape
    yield BenchmarkCasePlan(
        shape={"t": (n,)},
        params={
            "configuration": shape,
            "rows": n,
            "dim": dim,
            "flip_sin_to_cos": flip,
            "downscale_freq_shift": shift,
            "scale": scale,
            "max_period": max_period,
        },
        builder_args=(shape,),
    )


@pytest.mark.timestep_embedding
def test_perf_timestep_embedding():
    bench = OpBenchmark(
        op_name="timestep_embedding",
        torch_op=get_reference("timestep_embedding"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="n, dim, flip, shift, scale, max_period",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
