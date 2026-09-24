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

"""Benchmark for attention/fill_padded_rows."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(rows, 8, rows // 2) for rows in (8, 64, 512, 4096)]
MORE_SHAPES = [(rows, 8, rows // 2) for rows in (16384,)]


def _input_fn(shape, cur_dtype, device):
    rows, cols, mask_len = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        rows, cols, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    yield (x, mask_len)


@pytest.mark.fill_padded_rows
def test_perf_fill_padded_rows():
    bench = OpBenchmark(
        op_name="fill_padded_rows",
        torch_op=get_reference("fill_padded_rows"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="rows, cols, mask_len",
    )
    bench.set_gems(flaggems_sglang.fill_padded_rows)
    bench.run()
