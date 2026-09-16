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

"""Benchmark for moe/fused_moe_gemm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/fused_moe_gemm.
SHAPES = [(t, 8, 4096, 4096, 2) for t in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(8, 4, 64, 128, 2), (17, 8, 128, 256, 2), (5, 4, 64, 128, 1)]


def _input_fn(shape, cur_dtype, device):
    T, E, N, K, top_k = shape
    g = torch.Generator(device=device).manual_seed(0)
    A = torch.randn(T, K, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    B = torch.randn(
        E, N, K, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    topk_ids = torch.randint(
        0, E, (T, top_k), dtype=torch.int32, device=device, generator=g
    )
    topk_weights = torch.rand(
        T, top_k, device=device, generator=g, dtype=torch.float32
    )
    yield A, B, topk_weights, topk_ids, top_k


@pytest.mark.fused_moe_gemm
def test_perf_fused_moe_gemm():
    bench = OpBenchmark(
        op_name="fused_moe_gemm",
        torch_op=get_reference("fused_moe_gemm"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="T, E, N, K, top_k",
    )
    bench.set_gems(flaggems_sglang.fused_moe_gemm)
    bench.run()
