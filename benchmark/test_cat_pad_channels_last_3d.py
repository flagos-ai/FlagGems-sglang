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

"""Benchmark for diffusion/cat_pad_channels_last_3d."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_C, _CACHE_T = 128, 2
_PADDING = (1, 1, 1, 1, 2, 0)
# (t, spatial) — 32x32 frames with a 2-frame cache.
SHAPES = [(t, 32) for t in (4, 8, 16)]
MORE_SHAPES = [(4, 16), (8, 64), (32, 32)]


def _input_fn(shape, cur_dtype, device):
    t, spatial = shape
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*dims):
        # Materialized channels_last_3d from the start: the layout is the
        # point of this operator.
        return (
            torch.randn(*dims, dtype=torch.float32, device=device, generator=g)
            .to(cur_dtype)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .permute(0, 4, 1, 2, 3)
        )

    yield dict(
        x=_t(1, _C, t, spatial, spatial),
        cache_x=_t(1, _C, _CACHE_T, spatial, spatial),
        padding=_PADDING,
        keep_cache_t=0,
    )


@pytest.mark.cat_pad_channels_last_3d
def test_perf_cat_pad_channels_last_3d():
    gems_op = flaggems_sglang.get_op("cat_pad_channels_last_3d")
    if gems_op is None:
        pytest.skip("diffusion/cat_pad_channels_last_3d not implemented yet")
    bench = OpBenchmark(
        op_name="cat_pad_channels_last_3d",
        torch_op=get_reference("cat_pad_channels_last_3d"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="t, spatial",
    )
    bench.set_gems(gems_op)
    bench.run()
