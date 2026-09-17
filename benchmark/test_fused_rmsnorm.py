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

"""Benchmark for activation_norm/fused_rmsnorm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

EPS = 1e-6

# Shapes match kernel-comp-baseline/problems/activation_norm/fused_rmsnorm.
SHAPES = [(bs, h) for bs in (1, 8, 64, 512, 4096) for h in (1024, 4096, 8192)]
MORE_SHAPES = [(7, 16), (83, 1024), (48, 3072), (1, 8192)]


def _input_fn(shape, cur_dtype, device):
    bs, hidden = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        bs, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    weight = torch.randn(
        hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    yield x, weight, EPS


@pytest.mark.fused_rmsnorm
def test_perf_fused_rmsnorm():
    bench = OpBenchmark(
        op_name="fused_rmsnorm",
        torch_op=get_reference("fused_rmsnorm"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, hidden",
    )
    bench.set_gems(flaggems_sglang.fused_rmsnorm)
    bench.run()
