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

"""Benchmark for diffusion/rmsnorm_scale."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(r, d) for r in (8, 64, 1024) for d in (2048, 4096)]
MORE_SHAPES = [(8192, d) for d in (2048, 4096)]

_EPS = 1e-6


def _input_fn(shape, cur_dtype, device):
    rows, dim = shape
    g = torch.Generator(device=device).manual_seed(9)

    def _t(*s):
        return torch.randn(
            *s, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    # scale has one row per x row here; the broadcast path is covered by the
    # test.
    yield _t(rows, dim), _t(dim), _t(rows, dim), _EPS


@pytest.mark.rmsnorm_scale
def test_perf_rmsnorm_scale():
    bench = OpBenchmark(
        op_name="rmsnorm_scale",
        torch_op=get_reference("rmsnorm_scale"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_rows, hidden",
    )
    bench.set_gems(flaggems_sglang.rmsnorm_scale)
    bench.run()
