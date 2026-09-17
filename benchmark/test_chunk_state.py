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

"""Benchmark for mamba/chunk_state."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference.chunk_cumsum import (
    reference as chunk_cumsum_reference,
)

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/chunk_state.
SHAPES = [(8, 16, 256, 32, 8, 64, 128), (32, 4, 256, 64, 8, 64, 128)]
MORE_SHAPES = [
    (1, 1, 8, 4, 2, 16, 8),
    (2, 3, 16, 8, 2, 32, 16),
    (3, 2, 32, 16, 4, 64, 32),
]


def _input_fn(shape, cur_dtype, device):
    batch, nchunks, chunk_size, nheads, ngroups, headdim, dstate = shape
    g = torch.Generator(device=device).manual_seed(0)
    seqlen = nchunks * chunk_size
    x = torch.randn(
        batch,
        seqlen,
        nheads,
        headdim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    b = torch.randn(
        batch,
        seqlen,
        ngroups,
        dstate,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    raw_dt = torch.rand(
        batch,
        seqlen,
        nheads,
        generator=g,
        device=device,
        dtype=torch.float32,
    )
    a = (
        -torch.rand(nheads, generator=g, device=device, dtype=torch.float32)
        - 0.1
    )
    # dt/dA_cumsum come from the chunk_cumsum stage that precedes this op.
    dt_out, dA_cumsum = chunk_cumsum_reference(raw_dt, a, chunk_size)
    yield b, x, dt_out, dA_cumsum


@pytest.mark.chunk_state
def test_perf_chunk_state():
    bench = OpBenchmark(
        op_name="chunk_state",
        torch_op=get_reference("chunk_state"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc=(
            "batch, nchunks, chunk_size, nheads, ngroups, headdim, dstate"
        ),
    )
    bench.set_gems(flaggems_sglang.chunk_state)
    bench.run()
