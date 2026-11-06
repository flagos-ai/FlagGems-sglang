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

"""Benchmark for moe/swiglu_oai_split."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, n) for m in (1, 8, 64, 512) for n in (1024, 4096, 12288)]
MORE_SHAPES = [(4096, n) for n in (1024, 4096, 12288)]


def _input_fn(shape, cur_dtype, device):
    m, n_inter = shape
    g = torch.Generator(device=device).manual_seed(9)
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
    # limit and out_dtype stay None; the test covers both.
    yield gate_up, 1.702, 1.0, None, None


@pytest.mark.swiglu_oai_split
def test_perf_swiglu_oai_split():
    bench = OpBenchmark(
        op_name="swiglu_oai_split",
        torch_op=get_reference("swiglu_oai_split"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, intermediate",
    )
    bench.set_gems(flaggems_sglang.swiglu_oai_split)
    bench.run()
