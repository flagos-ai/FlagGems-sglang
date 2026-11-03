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

"""Benchmark for diffusion/scatter_to_padded."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(b, 1024, 16, 128) for b in (1, 4, 16)]
MORE_SHAPES = [(64, 1024, 16, 128)]


def _input_fn(shape, cur_dtype, device):
    batch, seqlen, heads, head_dim = shape
    g = torch.Generator(device=device).manual_seed(9)
    lens = torch.randint(
        1,
        seqlen + 1,
        (batch,),
        device=device,
        generator=g,
        dtype=torch.int64,
    )
    total = int(lens.sum())
    inv = torch.full((batch * seqlen,), -1, dtype=torch.int32, device=device)
    cursor = 0
    for b in range(batch):
        n = int(lens[b])
        inv[b * seqlen : b * seqlen + n] = torch.arange(
            cursor, cursor + n, dtype=torch.int32, device=device
        )
        cursor += n
    out_unpad = torch.randn(
        total,
        heads,
        head_dim,
        dtype=torch.float32,
        device=device,
        generator=g,
    ).to(cur_dtype)
    yield out_unpad, inv, batch, seqlen


@pytest.mark.scatter_to_padded
def test_perf_scatter_to_padded():
    bench = OpBenchmark(
        op_name="scatter_to_padded",
        torch_op=get_reference("scatter_to_padded"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, seqlen, heads, head_dim",
    )
    bench.set_gems(flaggems_sglang.scatter_to_padded)
    bench.run()
