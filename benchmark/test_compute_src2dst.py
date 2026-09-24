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

"""Benchmark for attention/compute_src2dst."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(n,) for n in (8, 512, 8192, 131072)]
MORE_SHAPES = [(n,) for n in (1048576,)]


def _input_fn(shape, cur_dtype, device):
    (n,) = shape
    g = torch.Generator(device=device).manual_seed(0)
    src_to_dst = torch.randint(
        0, n, (n,), dtype=torch.int32, device=device, generator=g
    )
    yield (src_to_dst,)


@pytest.mark.compute_src2dst
def test_perf_compute_src2dst():
    bench = OpBenchmark(
        op_name="compute_src2dst",
        torch_op=get_reference("compute_src2dst"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="n",
    )
    bench.set_gems(flaggems_sglang.compute_src2dst)
    bench.run()
