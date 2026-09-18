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

"""Benchmark for mamba/chunk_cumsum."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/chunk_cumsum; every case
# benchmarks the fully-featured path (dt_bias + dt_softplus).
SHAPES = [(8, 16, 256, 32), (32, 4, 256, 64)]
MORE_SHAPES = [(1, 1, 8, 4), (2, 3, 16, 8), (3, 2, 32, 16)]


def _input_fn(shape, cur_dtype, device):
    batch, nchunks, chunk_size, nheads = shape
    g = torch.Generator(device=device).manual_seed(0)
    seqlen = nchunks * chunk_size
    dt = torch.randn(
        batch,
        seqlen,
        nheads,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    # A must stay negative for the decay to be stable.
    a = (
        -torch.rand(nheads, generator=g, device=device, dtype=torch.float32)
        - 0.1
    )
    dt_bias = torch.randn(
        nheads, generator=g, device=device, dtype=torch.float32
    )
    yield dt, a, chunk_size, dict(dt_bias=dt_bias, dt_softplus=True)


@pytest.mark.chunk_cumsum
def test_perf_chunk_cumsum():
    bench = OpBenchmark(
        op_name="chunk_cumsum",
        torch_op=get_reference("chunk_cumsum"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, nchunks, chunk_size, nheads",
    )
    bench.set_gems(flaggems_sglang.chunk_cumsum)
    bench.run()
