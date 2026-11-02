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

"""Benchmark for diffusion/pack_qkv_destination_major."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(r, 32, 128, 8) for r in (64, 1024, 4096)]
MORE_SHAPES = [(16384, 32, 128, 8)]


def _input_fn(shape, cur_dtype, device):
    rows, global_heads, head_size, world_size = shape
    g = torch.Generator(device=device).manual_seed(9)

    def _t():
        return torch.randn(
            rows,
            global_heads,
            head_size,
            dtype=torch.float32,
            device=device,
            generator=g,
        ).to(cur_dtype)

    # out=None lets the operator allocate; the passed-out path is covered by
    # the test.
    yield _t(), _t(), _t(), world_size, None


@pytest.mark.pack_qkv_destination_major
def test_perf_pack_qkv_destination_major():
    bench = OpBenchmark(
        op_name="pack_qkv_destination_major",
        torch_op=get_reference("pack_qkv_destination_major"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="rows, global_heads, head_size, world_size",
    )
    bench.set_gems(flaggems_sglang.pack_qkv_destination_major)
    bench.run()
