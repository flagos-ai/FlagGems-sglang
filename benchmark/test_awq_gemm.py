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

"""Benchmark for quantization/awq_gemm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# (m, n) — k fixed at 4096, group_size=128, split_k=8.
SHAPES = [(m, n) for m in (1, 8, 64, 512) for n in (1024, 4096)]
MORE_SHAPES = [(16, 32), (24, 64), (32, 128)]


def _input_fn(shape, cur_dtype, device):
    m, n = shape
    k, group_size, split_k = 4096, 128, 8
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand(m, k, dtype=torch.float32, device=device, generator=g)
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (k, n // 8),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (k // group_size, n // 8),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    scales = torch.rand(
        k // group_size, n, dtype=torch.float32, device=device, generator=g
    )
    yield dict(
        input=x,
        qweight=qweight,
        scales=scales,
        qzeros=qzeros,
        split_k_iters=split_k,
    )


@pytest.mark.awq_gemm
def test_perf_awq_gemm():
    gems_op = flaggems_sglang.get_op("awq_gemm")
    if gems_op is None:
        pytest.skip("quantization/awq_gemm not implemented yet")
    bench = OpBenchmark(
        op_name="awq_gemm",
        torch_op=get_reference("awq_gemm"),
        input_fn=_input_fn,
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, n",
    )
    bench.set_gems(gems_op)
    bench.run()
