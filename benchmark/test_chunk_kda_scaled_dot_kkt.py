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

"""Benchmark for fla/chunk_kda_scaled_dot_kkt."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(1, t, 8, 128) for t in (64, 256, 1024)]
MORE_SHAPES = [(1, 4096, 8, 128)]

_CHUNK = 64


def _input_fn(shape, cur_dtype, device):
    b, t, h, kdim = shape
    g = torch.Generator(device=device).manual_seed(9)
    q = torch.randn(
        b, t, h, kdim, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    k = torch.randn(
        b, t, h, kdim, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    gate = -0.05 * torch.rand(
        b, t, h, kdim, dtype=torch.float32, device=device, generator=g
    )
    gk = (
        gate.reshape(b, -1, _CHUNK, h, kdim)
        .cumsum(dim=2)
        .reshape(b, t, h, kdim)
    )
    beta = torch.rand(b, t, h, dtype=torch.float32, device=device, generator=g)
    yield q, k, gk, beta, kdim**-0.5, 1.0, _CHUNK


@pytest.mark.chunk_kda_scaled_dot_kkt
def test_perf_chunk_kda_scaled_dot_kkt():
    bench = OpBenchmark(
        op_name="chunk_kda_scaled_dot_kkt",
        torch_op=get_reference("chunk_kda_scaled_dot_kkt"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, seq, heads, head_dim",
    )
    bench.set_gems(flaggems_sglang.chunk_kda_scaled_dot_kkt)
    bench.run()
