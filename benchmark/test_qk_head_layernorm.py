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

"""Benchmark for diffusion/qk_head_layernorm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(1, s, 32, 128) for s in (128, 1024, 4360)]
MORE_SHAPES = [(1, 16384, 32, 128)]

_EPS = 1e-6


def _input_fn(shape, cur_dtype, device):
    b, s, heads, head_dim = shape
    g = torch.Generator(device=device).manual_seed(9)

    def _t():
        return torch.randn(
            b,
            s,
            heads,
            head_dim,
            dtype=torch.float32,
            device=device,
            generator=g,
        ).to(cur_dtype)

    yield _t(), _t(), _EPS


@pytest.mark.qk_head_layernorm
def test_perf_qk_head_layernorm():
    bench = OpBenchmark(
        op_name="qk_head_layernorm",
        torch_op=get_reference("qk_head_layernorm"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, seq, heads, head_dim",
    )
    bench.set_gems(flaggems_sglang.qk_head_layernorm)
    bench.run()
