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

"""Benchmark for norm/group_norm_silu."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(n, c, s, 32) for n in (1, 2) for c, s in ((128, 64), (256, 32))]
MORE_SHAPES = [(n, c, s, 32) for n in (1, 2) for c, s in ((512, 16),)]


def _input_fn(shape, cur_dtype, device):
    n, c, spatial, num_groups = shape
    eps = 1e-5
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        n, c, spatial, spatial, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    w = torch.randn(c, dtype=torch.float32, device=device, generator=g).to(
        cur_dtype
    )
    b = torch.randn(c, dtype=torch.float32, device=device, generator=g).to(
        cur_dtype
    )
    yield x, w, b, {"num_groups": num_groups, "eps": eps}


@pytest.mark.group_norm_silu
def test_perf_group_norm_silu():
    bench = OpBenchmark(
        op_name="group_norm_silu",
        torch_op=get_reference("group_norm_silu"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="n, c, spatial, num_groups",
    )
    bench.set_gems(flaggems_sglang.group_norm_silu)
    bench.run()
