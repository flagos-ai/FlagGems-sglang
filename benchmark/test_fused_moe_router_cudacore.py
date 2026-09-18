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

"""Benchmark for moe/fused_moe_router_cudacore."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/fused_moe_router_cudacore.
SHAPES = [(m, 4096, 256, 8, 0.0, False) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [
    (1, 64, 8, 1, 0.0, False),
    (37, 256, 16, 2, 30.0, False),
    (83, 512, 32, 3, 0.0, True),
]


def _input_fn(shape, cur_dtype, device):
    bs, hidden_dim, num_experts, topk, softcap, has_bias = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        bs, hidden_dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    router_weight = torch.randn(
        num_experts,
        hidden_dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    bias = None
    if has_bias:
        bias = (
            torch.randn(
                num_experts, generator=g, device=device, dtype=torch.float32
            )
            * 0.1
        )
    yield x, router_weight, topk, softcap, bias


@pytest.mark.fused_moe_router_cudacore
def test_perf_fused_moe_router_cudacore():
    # This op has a reference but no Triton implementation yet.
    gems_op = flaggems_sglang.get_op("fused_moe_router_cudacore")
    if gems_op is None:
        pytest.skip("moe/fused_moe_router_cudacore not implemented yet")
    bench = OpBenchmark(
        op_name="fused_moe_router_cudacore",
        torch_op=get_reference("fused_moe_router_cudacore"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc=(
            "bs, hidden_dim, num_experts, topk, moe_softcapping, has_bias"
        ),
    )
    bench.set_gems(gems_op)
    bench.run()
