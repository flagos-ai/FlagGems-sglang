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

"""Benchmark for fla/fused_gdn_gating."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/fla/fused_gdn_gating.
SHAPES = [(m, 64) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(1, 4), (37, 32), (256, 8)]


def _input_fn(shape, cur_dtype, device):
    batch, num_heads = shape
    g = torch.Generator(device=device).manual_seed(0)
    # This op is float32 throughout; cur_dtype is not applied.
    a_log = (
        torch.randn(num_heads, generator=g, device=device, dtype=torch.float32)
        * 0.5
    )
    a = torch.randn(
        batch, num_heads, generator=g, device=device, dtype=torch.float32
    )
    b = torch.randn(
        batch, num_heads, generator=g, device=device, dtype=torch.float32
    )
    dt_bias = torch.randn(
        num_heads, generator=g, device=device, dtype=torch.float32
    )
    yield a_log, a, b, dt_bias


@pytest.mark.fused_gdn_gating
def test_perf_fused_gdn_gating():
    bench = OpBenchmark(
        op_name="fused_gdn_gating",
        torch_op=get_reference("fused_gdn_gating"),
        input_fn=_input_fn,
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, num_heads",
    )
    bench.set_gems(flaggems_sglang.fused_gdn_gating)
    bench.run()
