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

"""Benchmark for quantization/per_token_group_quant_8bit_v2."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, k) for m in (1, 8, 64, 512, 4096) for k in (2048, 4096, 7168)]
MORE_SHAPES = [(1, 512), (37, 2048), (512, 4096)]
_GROUP = 128


def _input_fn(shape, cur_dtype, device):
    m, k = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = (
        torch.randn(m, k, device=device, generator=g)
        .to(cur_dtype)
        .contiguous()
    )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    yield dict(
        input=x,
        output_q=torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=device),
        output_s=torch.zeros(
            m, k // _GROUP, dtype=torch.float32, device=device
        ),
        group_size=_GROUP,
        eps=1e-10,
        min_8bit=-fp8_max,
        max_8bit=fp8_max,
    )


@pytest.mark.per_token_group_quant_8bit_v2
def test_perf_per_token_group_quant_8bit_v2():
    gems_op = flaggems_sglang.get_op("per_token_group_quant_8bit_v2")
    if gems_op is None:
        pytest.skip(
            "quantization/per_token_group_quant_8bit_v2 not implemented yet"
        )
    bench = OpBenchmark(
        op_name="per_token_group_quant_8bit_v2",
        torch_op=get_reference("per_token_group_quant_8bit_v2"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, k",
    )
    bench.set_gems(gems_op)
    bench.run()
