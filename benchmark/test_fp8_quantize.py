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

"""Benchmark for quantization/fp8_quantize."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, n) for m in (1, 8, 64, 512, 4096) for n in (1024, 4096, 8192)]
MORE_SHAPES = [(7, 128), (83, 512), (256, 4096)]


def _input_fn(shape, cur_dtype, device):
    g = torch.Generator(device=device).manual_seed(0)
    x = (
        1.0
        * torch.randn(*shape, generator=g, device=device, dtype=torch.float32)
    ).to(cur_dtype)
    yield dict(x=x, scale_inv=1.0, fp8_dtype=torch.float8_e4m3fn)


@pytest.mark.fp8_quantize
def test_perf_fp8_quantize():
    gems_op = flaggems_sglang.get_op("fp8_quantize")
    if gems_op is None:
        pytest.skip("quantization/fp8_quantize not implemented yet")
    bench = OpBenchmark(
        op_name="fp8_quantize",
        torch_op=get_reference("fp8_quantize"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, n",
    )
    bench.set_gems(gems_op)
    bench.run()
