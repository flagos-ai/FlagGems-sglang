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

"""Benchmark for fla/chunk_scaled_dot_kkt."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/fla/chunk_scaled_dot_kkt.
SHAPES = [
    (8, 16, 64, 8, 32, 128, 1),
    (32, 4, 64, 8, 32, 128, 1),
]
MORE_SHAPES = [
    (1, 1, 16, 2, 2, 32, 1),
    (2, 3, 16, 2, 4, 32, 0),
    (3, 2, 32, 4, 8, 64, 1),
]


def _input_fn(shape, cur_dtype, device):
    batch, nchunks, chunk_size, hg, h, k_dim, use_g = shape
    g = torch.Generator(device=device).manual_seed(0)
    t = nchunks * chunk_size
    k = torch.randn(
        batch,
        t,
        hg,
        k_dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    beta = torch.rand(
        batch, t, h, generator=g, device=device, dtype=torch.float32
    )
    g_cumsum = None
    if use_g:
        raw = (
            -torch.rand(
                batch, t, h, generator=g, device=device, dtype=torch.float32
            )
            * 0.1
        )
        g_cumsum = (
            raw.view(batch, nchunks, chunk_size, h)
            .cumsum(dim=2)
            .view(batch, t, h)
        )
    yield k, beta, g_cumsum, {"chunk_size": chunk_size}


@pytest.mark.chunk_scaled_dot_kkt
def test_perf_chunk_scaled_dot_kkt():
    bench = OpBenchmark(
        op_name="chunk_scaled_dot_kkt",
        torch_op=get_reference("chunk_scaled_dot_kkt"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, nchunks, chunk_size, hg, h, k_dim, use_g",
    )
    bench.set_gems(flaggems_sglang.chunk_scaled_dot_kkt)
    bench.run()
