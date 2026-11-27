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

"""Benchmark for activation_norm/grouped_gemma_rmsnorm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.utils.rng_utils import create_generator

from .op_benchmark import OpBenchmark

# (rows, hidden, group_size) — group_size is a multiple of 512 dividing H.
SHAPES = [
    (1, 10240, 2560),
    (33, 1024, 512),
    (128, 10240, 2560),
    (1024, 2048, 1024),
]
MORE_SHAPES = [(5, 512, 512), (4096, 1024, 512), (16, 4096, 1024)]


def _input_fn(shape, cur_dtype, device):
    *dims, group_size = shape
    hidden = dims[-1]
    g = create_generator(device, seed=0)
    # The generator is bound to device, so every draw must name it too.
    x = torch.randn(*dims, generator=g, device=device).to(cur_dtype)
    weight = (
        torch.randn(hidden, generator=g, device=device).to(cur_dtype) * 0.2
    ).to(cur_dtype)
    yield dict(x=x, weight=weight, group_size=group_size, eps=1e-6)


@pytest.mark.grouped_gemma_rmsnorm
def test_perf_grouped_gemma_rmsnorm():
    gems_op = flaggems_sglang.get_op("grouped_gemma_rmsnorm")
    if gems_op is None:
        pytest.skip(
            "activation_norm/grouped_gemma_rmsnorm not implemented yet"
        )
    bench = OpBenchmark(
        op_name="grouped_gemma_rmsnorm",
        torch_op=get_reference("grouped_gemma_rmsnorm"),
        input_fn=_input_fn,
        dtypes=[torch.float16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="rows, hidden",
    )
    bench.set_gems(gems_op)
    bench.run()
