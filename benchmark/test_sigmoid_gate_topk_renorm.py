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

"""Benchmark for moe/sigmoid_gate_topk_renorm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

ROUTE_SCALE = 1.5
GLOBAL_SCALE = 0.8

# Shapes match kernel-comp-baseline/problems/moe/sigmoid_gate_topk_renorm.
SHAPES = [(m, 128, 3, 5) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(1, 32, 1, 3), (83, 64, 1, 6), (9, 256, 4, 12)]


def _input_fn(shape, cur_dtype, device):
    m, n_routed, n_shared, k = shape
    g = torch.Generator(device=device).manual_seed(0)
    n = n_routed + n_shared
    logits = torch.randn(
        m, n, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    bias = 0.1 * torch.randn(
        n_routed, generator=g, device=device, dtype=torch.float32
    )
    global_scale = torch.tensor(
        GLOBAL_SCALE, device=device, dtype=torch.float32
    )
    yield logits, k, n_shared, ROUTE_SCALE, global_scale, bias


@pytest.mark.sigmoid_gate_topk_renorm
def test_perf_sigmoid_gate_topk_renorm():
    bench = OpBenchmark(
        op_name="sigmoid_gate_topk_renorm",
        torch_op=get_reference("sigmoid_gate_topk_renorm"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, n_routed, n_shared, k",
    )
    bench.set_gems(flaggems_sglang.sigmoid_gate_topk_renorm)
    bench.run()
