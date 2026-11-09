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

"""Benchmark for moe/silu_and_mul_routed."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, n) for m in (1, 8, 64, 512) for n in (1024, 4096)]
MORE_SHAPES = [(4096, n) for n in (1024, 4096)]


def _input_fn(shape, cur_dtype, device):
    m, n = shape
    g = torch.Generator(device=device).manual_seed(9)
    gateup = torch.randn(
        m, 2 * n, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    weights = torch.rand(
        m, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    # out_dtype None, use_interleaved True: a bool only reaches the op
    # through a kwargs dict, and the split layout is covered by the test.
    yield gateup, weights, None, {"use_interleaved": True}


@pytest.mark.silu_and_mul_routed
def test_perf_silu_and_mul_routed():
    bench = OpBenchmark(
        op_name="silu_and_mul_routed",
        torch_op=get_reference("silu_and_mul_routed"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, intermediate",
    )
    bench.set_gems(flaggems_sglang.silu_and_mul_routed)
    bench.run()
