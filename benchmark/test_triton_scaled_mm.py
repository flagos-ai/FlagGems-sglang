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

"""Benchmark for quantization/triton_scaled_mm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [
    (m, k, n)
    for m in (1, 8, 64, 512, 4096)
    for k, n in ((4096, 4096), (4096, 7168))
]
MORE_SHAPES = [(32, 32, 32), (17, 64, 96)]


def _input_fn(shape, cur_dtype, device):
    m, k, n = shape
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.clamp(
        0.1
        * torch.randn(m, k, dtype=torch.float16, device=device, generator=g),
        -0.3,
        0.3,
    ).to(torch.float8_e4m3fn)
    b = torch.clamp(
        0.1
        * torch.randn(k, n, dtype=torch.float16, device=device, generator=g),
        -0.3,
        0.3,
    ).to(torch.float8_e4m3fn)
    scale_a = 0.1 + 0.05 * torch.rand(
        m, 1, dtype=torch.float32, device=device, generator=g
    )
    scale_b = 0.1 + 0.05 * torch.rand(
        n, 1, dtype=torch.float32, device=device, generator=g
    )
    yield dict(
        input=a,
        weight=b,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.float16,
    )


@pytest.mark.triton_scaled_mm
def test_perf_triton_scaled_mm():
    gems_op = flaggems_sglang.get_op("triton_scaled_mm")
    if gems_op is None:
        pytest.skip("quantization/triton_scaled_mm not implemented yet")
    bench = OpBenchmark(
        op_name="triton_scaled_mm",
        torch_op=get_reference("triton_scaled_mm"),
        input_fn=_input_fn,
        dtypes=[torch.float8_e4m3fn],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, k, n",
    )
    bench.set_gems(gems_op)
    bench.run()
