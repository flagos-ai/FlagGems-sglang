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

"""Benchmark for diffusion/timestep_embedding."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.utils.rng_utils import create_generator

from .op_benchmark import OpBenchmark

# (batch, dim) — dim must be even.
SHAPES = [(b, d) for b in (1, 8, 64, 512) for d in (256, 1024)]
MORE_SHAPES = [(16, 320), (256, 2048), (2048, 512)]


def _input_fn(shape, cur_dtype, device):
    batch, dim = shape
    g = create_generator(device, seed=0)
    t = (
        torch.rand(batch, dtype=torch.float32, device=device, generator=g)
        * 1000
    ).to(cur_dtype)
    yield dict(
        t=t,
        dim=dim,
        flip_sin_to_cos=False,
        downscale_freq_shift=0.0,
        scale=1.0,
        max_period=10000,
    )


@pytest.mark.timestep_embedding
def test_perf_timestep_embedding():
    gems_op = flaggems_sglang.get_op("timestep_embedding")
    if gems_op is None:
        pytest.skip("diffusion/timestep_embedding not implemented yet")
    bench = OpBenchmark(
        op_name="timestep_embedding",
        torch_op=get_reference("timestep_embedding"),
        input_fn=_input_fn,
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, dim",
    )
    bench.set_gems(gems_op)
    bench.run()
