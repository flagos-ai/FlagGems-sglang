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

"""Benchmark for diffusion/wan_rmsnorm_silu."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_T = 4
# (channels, spatial) — 0 < C <= 1024, channels_last_3d input.
SHAPES = [(c, s) for c in (128, 512) for s in (16, 32, 64)]
MORE_SHAPES = [(64, 8), (256, 32), (1024, 16)]


def _input_fn(shape, cur_dtype, device):
    c, spatial = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = (
        torch.randn(1, c, _T, spatial, spatial, device=device, generator=g)
        .to(cur_dtype)
        .contiguous(memory_format=torch.channels_last_3d)
    )
    gamma = torch.randn(c, device=device, generator=g).to(cur_dtype)
    yield dict(x=x, gamma=gamma, bias=None, rms_scale=None, eps=1e-12)


@pytest.mark.wan_rmsnorm_silu
def test_perf_wan_rmsnorm_silu():
    gems_op = flaggems_sglang.get_op("wan_rmsnorm_silu")
    if gems_op is None:
        pytest.skip("diffusion/wan_rmsnorm_silu not implemented yet")
    bench = OpBenchmark(
        op_name="wan_rmsnorm_silu",
        torch_op=get_reference("wan_rmsnorm_silu"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="channels, spatial",
    )
    bench.set_gems(gems_op)
    bench.run()
