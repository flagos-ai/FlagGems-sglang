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

"""Benchmark for utility/murmur_hash32."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark
from flaggems_sglang.utils.rng_utils import create_generator

# (n, m) — n seed/position rows against m column indices.
SHAPES = [(n, 1024) for n in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(1, 8), (37, 128), (64, 4096)]


def _input_fn(shape, cur_dtype, device):
    n, m = shape
    g = create_generator(device, seed=0)
    s = torch.randint(
        0, 1 << 30, (n,), dtype=torch.int32, device=device, generator=g
    )
    p = torch.randint(
        0, 1 << 20, (n,), dtype=torch.int32, device=device, generator=g
    )
    c = torch.arange(m, dtype=torch.int32, device=device)
    yield dict(seed=s, positions=p, col_indices=c)


@pytest.mark.murmur_hash32
def test_perf_murmur_hash32():
    gems_op = flaggems_sglang.get_op("murmur_hash32")
    if gems_op is None:
        pytest.skip("utility/murmur_hash32 not implemented yet")
    bench = OpBenchmark(
        op_name="murmur_hash32",
        torch_op=get_reference("murmur_hash32"),
        input_fn=_input_fn,
        dtypes=[torch.int32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="n, m",
    )
    bench.set_gems(gems_op)
    bench.run()
