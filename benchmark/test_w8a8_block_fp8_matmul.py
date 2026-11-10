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

"""Benchmark for quantization/w8a8_block_fp8_matmul."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_BLOCK = [128, 128]
_FP8_DTYPE = torch.float8_e4m3fn
SHAPES = [
    (m, n, k)
    for m in (1, 8, 64, 512, 4096)
    for n, k in ((1024, 4096), (4096, 4096), (7168, 4096))
]
MORE_SHAPES = [(7, 256, 512), (64, 1024, 512), (256, 1024, 4096)]


def _input_fn(shape, cur_dtype, device):
    m, n, k = shape
    block_n, block_k = _BLOCK
    fp8_max = torch.finfo(_FP8_DTYPE).max
    g = torch.Generator(device=device).manual_seed(0)
    A = (
        torch.randn(m, k, dtype=torch.float32, device=device, generator=g)
        .clamp(-fp8_max, fp8_max)
        .to(_FP8_DTYPE)
    )
    B = (
        torch.randn(n, k, dtype=torch.float32, device=device, generator=g)
        .clamp(-fp8_max, fp8_max)
        .to(_FP8_DTYPE)
    )
    As = (
        1e-2
        * torch.rand(
            m, k // block_k, dtype=torch.float32, device=device, generator=g
        )
    ).contiguous()
    Bs = (
        1e-2
        * torch.rand(
            n // block_n,
            k // block_k,
            dtype=torch.float32,
            device=device,
            generator=g,
        )
    ).contiguous()
    yield dict(
        A=A, B=B, As=As, Bs=Bs, block_size=_BLOCK, output_dtype=cur_dtype
    )


@pytest.mark.w8a8_block_fp8_matmul
def test_perf_w8a8_block_fp8_matmul():
    gems_op = flaggems_sglang.get_op("w8a8_block_fp8_matmul")
    if gems_op is None:
        pytest.skip("quantization/w8a8_block_fp8_matmul not implemented yet")
    bench = OpBenchmark(
        op_name="w8a8_block_fp8_matmul",
        torch_op=get_reference("w8a8_block_fp8_matmul"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, n, k",
    )
    bench.set_gems(gems_op)
    bench.run()
