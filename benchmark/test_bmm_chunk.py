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

"""Benchmark for mamba/bmm_chunk."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

CAUSAL = True

# Shapes match kernel-comp-baseline/problems/mamba/bmm_chunk.
SHAPES = [(8, 16, 256, 8, 64), (32, 4, 256, 8, 64)]
MORE_SHAPES = [(1, 1, 8, 2, 16), (2, 3, 16, 2, 32), (3, 2, 32, 4, 64)]


def _input_fn(shape, cur_dtype, device):
    batch, nchunks, chunk_size, ngroups, k = shape
    g = torch.Generator(device=device).manual_seed(0)
    seqlen = nchunks * chunk_size
    a = torch.randn(
        batch,
        seqlen,
        ngroups,
        k,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    b = torch.randn(
        batch,
        seqlen,
        ngroups,
        k,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    yield a, b, chunk_size, CAUSAL


@pytest.mark.bmm_chunk
def test_perf_bmm_chunk():
    bench = OpBenchmark(
        op_name="bmm_chunk",
        torch_op=get_reference("bmm_chunk"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, nchunks, chunk_size, ngroups, k",
    )
    bench.set_gems(flaggems_sglang.bmm_chunk)
    bench.run()
