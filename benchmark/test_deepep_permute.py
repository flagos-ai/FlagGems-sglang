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

"""Benchmark for moe/deepep_permute."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(t, 8, 7168) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 8, 7168) for t in (2048,)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, topk, hidden = shape
    g = torch.Generator(device=device).manual_seed(0)
    n = num_tokens * topk
    x = torch.randn(
        num_tokens, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    dst = torch.randperm(
        n, device="cpu", generator=torch.Generator(device="cpu").manual_seed(0)
    ).to(device)
    yield x, dst


@pytest.mark.deepep_permute
def test_perf_deepep_permute():
    bench = OpBenchmark(
        op_name="deepep_permute",
        torch_op=get_reference("deepep_permute"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, topk, hidden",
    )
    bench.set_gems(flaggems_sglang.deepep_permute)
    bench.run()
