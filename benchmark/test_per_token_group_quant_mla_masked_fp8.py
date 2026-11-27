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

"""Benchmark for quantization/per_token_group_quant_mla_masked_fp8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.utils.rng_utils import create_generator

from .op_benchmark import OpBenchmark

_GROUP = 128
_EPS = 1e-12
# (b, m) — k fixed at 512.
SHAPES = [(b, m) for b in (1, 8, 32) for m in (16, 128, 1024)]
MORE_SHAPES = [(1, 7), (4, 300), (8, 2048)]


def _input_fn(shape, cur_dtype, device):
    b, m = shape
    k = 512
    g = create_generator(device, seed=0)
    x = torch.randn(
        b, m, k, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    yield dict(x=x.contiguous(), group_size=_GROUP, eps=_EPS)


@pytest.mark.per_token_group_quant_mla_masked_fp8
def test_perf_per_token_group_quant_mla_masked_fp8():
    gems_op = flaggems_sglang.get_op("per_token_group_quant_mla_masked_fp8")
    if gems_op is None:
        pytest.skip(
            "quantization/per_token_group_quant_mla_masked_fp8 not implemented yet"
        )
    bench = OpBenchmark(
        op_name="per_token_group_quant_mla_masked_fp8",
        torch_op=get_reference("per_token_group_quant_mla_masked_fp8"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="b, m",
    )
    bench.set_gems(gems_op)
    bench.run()
