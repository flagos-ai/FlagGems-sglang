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

"""Benchmark for quantization/store_sf_interleaved."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_PAGE = 128
_PAGES = 256
SHAPES = [(t,) for t in (1, 8, 64, 512, 4096, 16384)]
MORE_SHAPES = [(1,), (37,), (512,)]


def _input_fn(shape, cur_dtype, device):
    (t,) = shape
    nheads = 8
    g = torch.Generator(device=device).manual_seed(0)
    sf_in = torch.randint(
        0, 256, (t, nheads, 4), dtype=torch.uint8, device=device, generator=g
    )
    sf_out = torch.zeros(_PAGES, nheads, 32, 4, 4, dtype=torch.uint8, device=device)
    loc = torch.randperm(_PAGES * _PAGE, device=device, generator=g)[:t].to(torch.int64)
    yield dict(sf_in=sf_in, sf_out=sf_out, loc=loc, page_size=_PAGE)


@pytest.mark.store_sf_interleaved
def test_perf_store_sf_interleaved():
    gems_op = flaggems_sglang.get_op("store_sf_interleaved")
    if gems_op is None:
        pytest.skip("quantization/store_sf_interleaved not implemented yet")
    bench = OpBenchmark(
        op_name="store_sf_interleaved",
        torch_op=get_reference("store_sf_interleaved"),
        input_fn=_input_fn,
        dtypes=[torch.uint8],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="t",
    )
    bench.set_gems(gems_op)
    bench.run()
