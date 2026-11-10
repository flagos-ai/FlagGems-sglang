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

"""Benchmark for diffusion/zimage_qk_rmsnorm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_HEAD_DIM = 128
SHAPES = [(s,) for s in (128, 1024, 4360, 16384)]
MORE_SHAPES = [(64,), (2048,)]


def _input_fn(shape, cur_dtype, device):
    (s,) = shape
    heads = 32
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        1, s, heads, _HEAD_DIM, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    weight = torch.randn(_HEAD_DIM, dtype=torch.float32, device=device, generator=g).to(
        cur_dtype
    ).contiguous()
    yield dict(x=x, weight=weight, eps=1e-6)


@pytest.mark.zimage_qk_rmsnorm
def test_perf_zimage_qk_rmsnorm():
    gems_op = flaggems_sglang.get_op("zimage_qk_rmsnorm")
    if gems_op is None:
        pytest.skip("diffusion/zimage_qk_rmsnorm not implemented yet")
    bench = OpBenchmark(
        op_name="zimage_qk_rmsnorm",
        torch_op=get_reference("zimage_qk_rmsnorm"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seq_len",
    )
    bench.set_gems(gems_op)
    bench.run()
