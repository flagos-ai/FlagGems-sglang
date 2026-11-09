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

"""Benchmark for fla/kda_gate_chunk_cumsum."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(1, t, 16, 128) for t in (64, 512, 2048)]
MORE_SHAPES = [(1, 8192, 16, 128)]

_CHUNK = 64


def _input_fn(shape, cur_dtype, device):
    b, t, h, s = shape
    gen = torch.Generator(device=device).manual_seed(9)
    g = torch.randn(
        b, t, h, s, dtype=torch.float32, device=device, generator=gen
    ).to(cur_dtype)
    a_log = (
        torch.randn(h, dtype=torch.float32, device=device, generator=gen) * 0.5
    )
    # dt_bias, scale and lower_bound stay None here; the test covers each.
    yield g, a_log, _CHUNK, None, None, None


@pytest.mark.kda_gate_chunk_cumsum
def test_perf_kda_gate_chunk_cumsum():
    bench = OpBenchmark(
        op_name="kda_gate_chunk_cumsum",
        torch_op=get_reference("kda_gate_chunk_cumsum"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, seq, heads, state",
    )
    bench.set_gems(flaggems_sglang.kda_gate_chunk_cumsum)
    bench.run()
