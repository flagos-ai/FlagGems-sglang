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

"""Benchmark for rope/interleaved_rope."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/rope/interleaved_rope. The last
# entry is the mrope section split.
SHAPES = [(s, 128, (16, 24, 24)) for s in (1, 128, 2048, 8192)]
MORE_SHAPES = [
    (1, 48, (8, 8, 8)),
    (37, 96, (16, 16, 16)),
    (257, 128, (16, 24, 24)),
]


def _input_fn(shape, cur_dtype, device):
    s, d, mrope_section = shape
    # The op reads the three rope variants from a single [3, s, d] tensor.
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        3, s, d, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    yield x, list(mrope_section)


@pytest.mark.interleaved_rope
def test_perf_interleaved_rope():
    # This op has a reference but no Triton implementation yet.
    gems_op = flaggems_sglang.get_op("interleaved_rope")
    if gems_op is None:
        pytest.skip("rope/interleaved_rope not implemented yet")
    bench = OpBenchmark(
        op_name="interleaved_rope",
        torch_op=get_reference("interleaved_rope"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="s, d, mrope_section",
    )
    bench.set_gems(gems_op)
    bench.run()
