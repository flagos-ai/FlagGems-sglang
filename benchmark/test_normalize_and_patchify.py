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

"""Benchmark for utility/normalize_and_patchify."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(b, 3, 448, 448, 14) for b in (1, 4)]
MORE_SHAPES = [(16, 3, 448, 448, 14)]


def _input_fn(shape, cur_dtype, device):
    batch, channels, h, w, patch = shape
    g = torch.Generator(device=device).manual_seed(9)
    image = torch.rand(
        batch,
        channels,
        h,
        w,
        dtype=torch.float32,
        device=device,
        generator=g,
    ).to(cur_dtype)
    scale = torch.rand(
        channels, 1, 1, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    bias = torch.randn(
        channels, 1, 1, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    # No padding here; the pad-to-a-patch-multiple path is covered by the
    # test.
    yield image, scale, bias, patch, h, w


@pytest.mark.normalize_and_patchify
def test_perf_normalize_and_patchify():
    bench = OpBenchmark(
        op_name="normalize_and_patchify",
        torch_op=get_reference("normalize_and_patchify"),
        input_fn=_input_fn,
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, channels, height, width, patch_size",
    )
    bench.set_gems(flaggems_sglang.normalize_and_patchify)
    bench.run()
