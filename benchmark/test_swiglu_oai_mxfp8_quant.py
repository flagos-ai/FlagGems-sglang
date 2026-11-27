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

"""Benchmark for moe/swiglu_oai_mxfp8_quant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.utils.rng_utils import create_generator

from .op_benchmark import OpBenchmark

# (m, n_inter) — n_inter must be a multiple of 32, the MXFP8 group width.
SHAPES = [(m, n) for m in (1, 8, 64, 512, 4096) for n in (1024, 4096, 12288)]
MORE_SHAPES = [(7, 128), (333, 1024), (256, 512)]


def _input_fn(shape, cur_dtype, device):
    m, n_inter = shape
    g = create_generator(device, seed=0)
    gate_up = (
        2.0
        * torch.randn(
            m,
            2 * n_inter,
            dtype=torch.float32,
            device=device,
            generator=g,
        )
    ).to(cur_dtype)
    yield dict(gate_up=gate_up.contiguous(), alpha=1.702, beta=1.0, limit=None)


@pytest.mark.swiglu_oai_mxfp8_quant
def test_perf_swiglu_oai_mxfp8_quant():
    gems_op = flaggems_sglang.get_op("swiglu_oai_mxfp8_quant")
    if gems_op is None:
        pytest.skip("moe/swiglu_oai_mxfp8_quant not implemented yet")
    bench = OpBenchmark(
        op_name="swiglu_oai_mxfp8_quant",
        torch_op=get_reference("swiglu_oai_mxfp8_quant"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, n_inter",
    )
    bench.set_gems(gems_op)
    bench.run()
