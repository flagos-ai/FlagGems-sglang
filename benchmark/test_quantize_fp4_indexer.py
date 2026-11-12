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

"""Benchmark for quantization/quantize_fp4_indexer."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(n,) for n in (16, 128, 1024, 8192, 65536)]
MORE_SHAPES = [(7,), (333,), (256,)]


def _input_fn(shape, cur_dtype, device):
    (n,) = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = (
        torch.randn(n, 128, dtype=torch.float32, device=device, generator=g)
        .to(cur_dtype)
        .contiguous()
    )
    yield dict(x=x)


@pytest.mark.quantize_fp4_indexer
def test_perf_quantize_fp4_indexer():
    gems_op = flaggems_sglang.get_op("quantize_fp4_indexer")
    if gems_op is None:
        pytest.skip("quantization/quantize_fp4_indexer not implemented yet")
    bench = OpBenchmark(
        op_name="quantize_fp4_indexer",
        torch_op=get_reference("quantize_fp4_indexer"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="n",
    )
    bench.set_gems(gems_op)
    bench.run()
