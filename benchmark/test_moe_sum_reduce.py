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

"""Benchmark for moe/moe_sum_reduce."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

ROUTED_SCALING_FACTOR = 2.5

# Shapes match kernel-comp-baseline/problems/moe/moe_sum_reduce.
SHAPES = [(t, 8, 7168) for t in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(7, 2, 128), (83, 4, 512), (3, 8, 2048)]


def _input_fn(shape, cur_dtype, device):
    T, top_k, H = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        T, top_k, H, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    yield x, ROUTED_SCALING_FACTOR


@pytest.mark.moe_sum_reduce
def test_perf_moe_sum_reduce():
    bench = OpBenchmark(
        op_name="moe_sum_reduce",
        torch_op=get_reference("moe_sum_reduce"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="T, top_k, H",
    )
    bench.set_gems(flaggems_sglang.moe_sum_reduce)
    bench.run()
