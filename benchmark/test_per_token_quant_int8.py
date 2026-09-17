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

"""Benchmark for quantization/per_token_quant_int8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/quantization/
# per_token_quant_int8.
SHAPES = [(m, k) for m in (1, 8, 64, 512, 4096) for k in (2048, 4096, 8192)]
MORE_SHAPES = [(7, 128), (83, 512), (256, 4096), (3, 4736)]


def _input_fn(shape, cur_dtype, device):
    m, k = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = (
        (
            torch.rand(m, k, generator=g, device=device, dtype=torch.float32)
            * 2
            - 1
        )
        .to(cur_dtype)
        .contiguous()
    )
    yield (x,)


@pytest.mark.per_token_quant_int8
def test_perf_per_token_quant_int8():
    # This op has a reference but no Triton implementation yet.
    gems_op = flaggems_sglang.get_op("per_token_quant_int8")
    if gems_op is None:
        pytest.skip("quantization/per_token_quant_int8 not implemented yet")
    bench = OpBenchmark(
        op_name="per_token_quant_int8",
        torch_op=get_reference("per_token_quant_int8"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, k",
    )
    bench.set_gems(gems_op)
    bench.run()
