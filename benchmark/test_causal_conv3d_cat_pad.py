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

"""Benchmark for diffusion/causal_conv3d_cat_pad."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(1, 128, t, 32, 32, 2) for t in (4, 8)]
MORE_SHAPES = [(1, 128, 16, 32, 32, 2)]

_PADDING = (1, 1, 1, 1, 2, 0)


def _input_fn(shape, cur_dtype, device):
    b, c, t, h, w, cache_t = shape
    g = torch.Generator(device=device).manual_seed(9)

    def _t(*s):
        return torch.randn(
            *s, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    # padding is a tuple, which unpack_to_args_kwargs forwards positionally.
    yield _t(b, c, t, h, w), _t(b, c, cache_t, h, w), _PADDING


@pytest.mark.causal_conv3d_cat_pad
def test_perf_causal_conv3d_cat_pad():
    bench = OpBenchmark(
        op_name="causal_conv3d_cat_pad",
        torch_op=get_reference("causal_conv3d_cat_pad"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, channels, frames, height, width, cache_frames",
    )
    bench.set_gems(flaggems_sglang.causal_conv3d_cat_pad)
    bench.run()
