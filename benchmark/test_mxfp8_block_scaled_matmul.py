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

"""Benchmark for quantization/mxfp8_block_scaled_matmul."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_FP8_MAX = 448.0
SHAPES = [(m, 4096, n) for m in (128, 1024, 4096) for n in (4096, 7168)]
MORE_SHAPES = [(64, 128, 128), (130, 256, 384)]


def _quantize_mx(x):
    m, k = x.shape
    xg = x.reshape(m, k // 32, 32).to(torch.float32)
    amax = xg.abs().amax(-1).clamp(min=1e-12)
    exp = torch.ceil(torch.log2(amax / _FP8_MAX)).clamp(-127, 127)
    xq = (xg / torch.exp2(exp).unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX).to(
        torch.float8_e4m3fn
    )
    return xq.reshape(m, k), (exp + 127).to(torch.uint8)


def _input_fn(shape, cur_dtype, device):
    m, k, n = shape
    g = torch.Generator(device=device).manual_seed(0)
    a_f = torch.randn(m, k, dtype=torch.float32, device=device, generator=g)
    b_f = torch.randn(n, k, dtype=torch.float32, device=device, generator=g)
    a, a_scale = _quantize_mx(a_f)
    b, b_scale = _quantize_mx(b_f)
    yield dict(a=a, a_scale=a_scale, b=b, b_scale=b_scale, output_dtype=torch.bfloat16)


@pytest.mark.mxfp8_block_scaled_matmul
def test_perf_mxfp8_block_scaled_matmul():
    gems_op = flaggems_sglang.get_op("mxfp8_block_scaled_matmul")
    if gems_op is None:
        pytest.skip("quantization/mxfp8_block_scaled_matmul not implemented yet")
    bench = OpBenchmark(
        op_name="mxfp8_block_scaled_matmul",
        torch_op=get_reference("mxfp8_block_scaled_matmul"),
        input_fn=_input_fn,
        dtypes=[torch.float8_e4m3fn],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, k, n",
    )
    bench.set_gems(gems_op)
    bench.run()
