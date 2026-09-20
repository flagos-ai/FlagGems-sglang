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

"""Benchmark for mamba/state_passing."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference.chunk_cumsum import (
    reference as chunk_cumsum_reference,
)

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/state_passing. The last
# entry is the has_init switch for the optional initial state.
SHAPES = [
    (8, 16, 256, 32, 8192, False),
    (32, 4, 256, 64, 8192, False),
]
MORE_SHAPES = [
    (1, 1, 8, 4, 16, False),
    (2, 3, 16, 8, 32, True),
    (3, 5, 32, 16, 64, False),
]


def _input_fn(shape, cur_dtype, device):
    batch, nchunks, chunk_size, nheads, dim, has_init = shape
    g = torch.Generator(device=device).manual_seed(0)
    states = torch.randn(
        batch,
        nchunks,
        nheads,
        dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    raw_dt = torch.rand(
        batch,
        nchunks * chunk_size,
        nheads,
        generator=g,
        device=device,
        dtype=torch.float32,
    )
    # A must stay negative for the decay to be stable.
    a = (
        -torch.rand(nheads, generator=g, device=device, dtype=torch.float32)
        - 0.1
    )
    # dA_cumsum comes from the chunk_cumsum stage that precedes this op.
    _, dA_cumsum = chunk_cumsum_reference(raw_dt, a, chunk_size)
    initial_states = None
    if has_init:
        initial_states = torch.randn(
            batch, nheads, dim, generator=g, device=device, dtype=torch.float32
        ).to(cur_dtype)
    yield states, dA_cumsum, initial_states


@pytest.mark.state_passing
def test_perf_state_passing():
    bench = OpBenchmark(
        op_name="state_passing",
        torch_op=get_reference("state_passing"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, nchunks, chunk_size, nheads, dim, has_init",
    )
    bench.set_gems(flaggems_sglang.state_passing)
    bench.run()
