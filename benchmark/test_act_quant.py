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

"""Benchmark for dsa/act_quant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark
from flaggems_sglang.utils.rng_utils import create_generator

# (m, n) — block_size fixed at 128, plain (non-round) scales.
SHAPES = [(m, 4096) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(1, 128), (37, 512), (83, 1024)]


def _input_fn(shape, cur_dtype, device):
    m, n = shape
    g = create_generator(device, seed=0)
    x = torch.randn(m, n, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    yield dict(x=x, block_size=128, scale_fmt=None)


@pytest.mark.act_quant
def test_perf_act_quant():
    gems_op = flaggems_sglang.get_op("act_quant")
    if gems_op is None:
        pytest.skip("dsa/act_quant not implemented yet")
    bench = OpBenchmark(
        op_name="act_quant",
        torch_op=get_reference("act_quant"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, n",
    )
    bench.set_gems(gems_op)
    bench.run()
