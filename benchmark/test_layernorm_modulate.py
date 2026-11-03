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

"""Benchmark for diffusion/layernorm_modulate."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(b, s, 2240) for b in (1, 2) for s in (77, 1024, 4096)]
MORE_SHAPES = [(8, s, 2240) for s in (77, 1024, 4096)]

_EPS = 1e-6


def _input_fn(shape, cur_dtype, device):
    batch, seq, hidden = shape
    g = torch.Generator(device=device).manual_seed(9)

    def _t(*s):
        return torch.randn(
            *s, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    yield _t(batch, seq, hidden), _t(batch, hidden), _t(batch, hidden), _EPS


@pytest.mark.layernorm_modulate
def test_perf_layernorm_modulate():
    bench = OpBenchmark(
        op_name="layernorm_modulate",
        torch_op=get_reference("layernorm_modulate"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, seq, hidden",
    )
    bench.set_gems(flaggems_sglang.layernorm_modulate)
    bench.run()
