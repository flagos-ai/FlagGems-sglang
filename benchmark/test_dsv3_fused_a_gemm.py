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

"""Benchmark for moe/dsv3_fused_a_gemm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, 4096, 64, 8) for m in (1, 2, 4, 8)]
MORE_SHAPES = [(m, 4096, 64, 8) for m in (16,)]


def _input_fn(shape, cur_dtype, device):
    m, d, num_experts, top_k = shape
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, d, dtype=torch.float32, device=device, generator=g).to(
        cur_dtype
    )
    b = torch.randn(
        num_experts, d, d, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    topk_weights = torch.randn(
        m, top_k, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    topk_ids = torch.randint(
        0,
        num_experts,
        (m, top_k),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    sorted_token_ids = torch.arange(m, dtype=torch.int32, device=device)
    expert_ids = torch.arange(num_experts, dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.full(
        (num_experts,), m, dtype=torch.int32, device=device
    )
    yield a, b, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded


@pytest.mark.dsv3_fused_a_gemm
def test_perf_dsv3_fused_a_gemm():
    bench = OpBenchmark(
        op_name="dsv3_fused_a_gemm",
        torch_op=get_reference("dsv3_fused_a_gemm"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, d, num_experts, top_k",
    )
    bench.set_gems(flaggems_sglang.dsv3_fused_a_gemm)
    bench.run()
