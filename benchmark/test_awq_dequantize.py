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

"""Benchmark for quantization/awq_dequantize."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# (rows, cols) — group_size fixed at 128.
SHAPES = [(rows, cols) for rows in (3584, 18944) for cols in (448, 4736)]
MORE_SHAPES = [(3584, 448, -1), (128, 16, 32), (256, 32, 64)]


def _input_fn(shape, cur_dtype, device):
    rows, cols = shape
    group_size = 128
    g = torch.Generator(device=device).manual_seed(0)
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (rows, cols),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    scales = torch.rand(
        rows // group_size,
        cols * 8,
        dtype=torch.float16,
        device=device,
        generator=g,
    )
    zeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (rows // group_size, cols),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    yield dict(qweight=qweight, scales=scales, zeros=zeros)


@pytest.mark.awq_dequantize
def test_perf_awq_dequantize():
    gems_op = flaggems_sglang.get_op("awq_dequantize")
    if gems_op is None:
        pytest.skip("quantization/awq_dequantize not implemented yet")
    bench = OpBenchmark(
        op_name="awq_dequantize",
        torch_op=get_reference("awq_dequantize"),
        input_fn=_input_fn,
        dtypes=[torch.float16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="rows, cols",
    )
    bench.set_gems(gems_op)
    bench.run()
