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

"""Benchmark for fla/chunk_local_cumsum_scalar."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

reference = get_reference("chunk_local_cumsum_scalar")


class ChunkLocalCumsumScalarBenchmark(OpBenchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_DESC = "batch, nchunks, chunk_size, nheads"
    # Shapes match kernel-comp-baseline/problems/fla/
    # chunk_local_cumsum_scalar.
    CORE_SHAPES = [
        (8, 16, 64, 32),
        (32, 4, 64, 64),
    ]
    MORE_SHAPES = [
        (2, 3, 16, 8),
        (3, 2, 32, 16),
    ]

    def __init__(self, *args, reverse=False, scale=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reverse = reverse
        self.scale = scale

    def get_input_iter(self, cur_dtype):
        for batch, nchunks, chunk_size, nheads in self.shapes:
            g = torch.Generator(device=self.device).manual_seed(0)
            x = torch.randn(
                batch,
                nchunks * chunk_size,
                nheads,
                generator=g,
                device=self.device,
                dtype=torch.float32,
            ).to(cur_dtype)
            yield x, chunk_size, self.reverse, self.scale


@pytest.mark.chunk_local_cumsum_scalar
@pytest.mark.parametrize("reverse", [False, True])
def test_perf_chunk_local_cumsum_scalar(reverse):
    bench = ChunkLocalCumsumScalarBenchmark(
        op_name="chunk_local_cumsum_scalar",
        torch_op=reference,
        reverse=reverse,
    )
    bench.set_gems(flaggems_sglang.chunk_local_cumsum_scalar)
    bench.run()
