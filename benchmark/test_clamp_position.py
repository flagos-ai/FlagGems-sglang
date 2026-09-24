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

"""Benchmark for attention/clamp_position."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(bs,) for bs in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(bs,) for bs in (16384,)]


def _input_fn(shape, cur_dtype, device):
    (bs,) = shape
    g = torch.Generator(device=device).manual_seed(0)
    seq_lens = torch.randint(
        0, 8192, (bs,), dtype=torch.int32, device=device, generator=g
    )
    yield (seq_lens,)


@pytest.mark.clamp_position
def test_perf_clamp_position():
    bench = OpBenchmark(
        op_name="clamp_position",
        torch_op=get_reference("clamp_position"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs",
    )
    bench.set_gems(flaggems_sglang.clamp_position)
    bench.run()
