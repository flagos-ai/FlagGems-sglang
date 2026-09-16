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

"""Benchmark for quantization/per_group_transpose."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

NUM_EXPERTS = 8

# Shapes match kernel-comp-baseline/problems/quantization/per_group_transpose;
# every expert gets ``rows_per_expert`` rows.
SHAPES = [(k, n) for k in (128, 512, 4096) for n in (16, 128, 1024)]


def _input_fn(shape, cur_dtype, device):
    k, rows_per_expert = shape
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(
        rows_per_expert * NUM_EXPERTS,
        k,
        generator=g,
        device=device,
        dtype=cur_dtype,
    ).contiguous()
    expert_offsets = torch.tensor(
        [i * rows_per_expert for i in range(NUM_EXPERTS + 1)],
        dtype=torch.int32,
        device=device,
    )
    yield a, expert_offsets


@pytest.mark.per_group_transpose
def test_perf_per_group_transpose():
    bench = OpBenchmark(
        op_name="per_group_transpose",
        torch_op=get_reference("per_group_transpose"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        shape_desc="k, rows_per_expert",
    )
    bench.set_gems(flaggems_sglang.per_group_transpose)
    bench.run()
