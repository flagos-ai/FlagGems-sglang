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

"""Benchmark for fla/chunk_local_cumsum_vector."""

from functools import partial

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/fla/chunk_local_cumsum_vector.
# chunk_size must be >= 16: the kernel implements the cumsum via a triangular
# matmul whose contraction dim is chunk_size (tl.dot requires K >= 16).
SHAPES = [(8, 16, 64, 8, 64), (32, 4, 64, 8, 128)]
MORE_SHAPES = [(1, 1, 16, 2, 16), (2, 3, 16, 4, 32), (3, 2, 32, 4, 64)]


def _input_fn(shape, cur_dtype, device, reverse, scale=None):
    batch, nchunks, chunk_size, nheads, s = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        batch,
        nchunks * chunk_size,
        nheads,
        s,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    yield x, chunk_size, reverse, scale


@pytest.mark.chunk_local_cumsum_vector
@pytest.mark.parametrize("reverse", [False, True])
def test_perf_chunk_local_cumsum_vector(reverse):
    bench = OpBenchmark(
        op_name="chunk_local_cumsum_vector",
        torch_op=get_reference("chunk_local_cumsum_vector"),
        input_fn=partial(_input_fn, reverse=reverse),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, nchunks, chunk_size, nheads, s",
    )
    bench.set_gems(flaggems_sglang.chunk_local_cumsum_vector)
    bench.run()
