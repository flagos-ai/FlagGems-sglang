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

"""Benchmark for diffusion/rotary_embedding."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

INTERLEAVED = False

# Shapes match kernel-comp-baseline/problems/diffusion/rotary_embedding.
SHAPES = [(t, 16, 128) for t in (1, 64, 1024, 8192, 32768)]
MORE_SHAPES = [(37, 16, 128), (1024, 8, 128), (4096, 16, 64)]


def _input_fn(shape, cur_dtype, device):
    tokens, heads, head_size = shape
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(tokens, heads, head_size, generator=g, device=device).to(
        cur_dtype
    )
    ang = torch.randn(tokens, head_size // 2, generator=g, device=device)
    yield x, torch.cos(ang).to(cur_dtype), torch.sin(ang).to(
        cur_dtype
    ), INTERLEAVED


@pytest.mark.rotary_embedding
def test_perf_rotary_embedding():
    bench = OpBenchmark(
        op_name="rotary_embedding",
        torch_op=get_reference("rotary_embedding"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, heads, head_size",
    )
    bench.set_gems(flaggems_sglang.rotary_embedding)
    bench.run()
