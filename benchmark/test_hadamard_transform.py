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

"""Benchmark for quantization/hadamard_transform."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, n) for m in (1, 8, 64, 512) for n in (128, 512, 2048)]
MORE_SHAPES = [(4096, n) for n in (128, 512, 2048)]


def _input_fn(shape, cur_dtype, device):
    m, n = shape
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(m, n, device=device, generator=g).to(cur_dtype)
    yield x, 1.0


@pytest.mark.hadamard_transform
def test_perf_hadamard_transform():
    bench = OpBenchmark(
        op_name="hadamard_transform",
        torch_op=get_reference("hadamard_transform"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_rows, hadamard_size",
    )
    bench.set_gems(flaggems_sglang.hadamard_transform)
    bench.run()
