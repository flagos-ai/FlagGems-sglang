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

"""Benchmark for activation_norm/hc_head."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/activation_norm/hc_head.
SHAPES = [(t, 4, 7168) for t in (1, 128, 2048, 8192)]
MORE_SHAPES = [(1, 2, 128), (37, 4, 256), (129, 2, 512)]

_NORM_EPS = 1e-6
_HC_EPS = 1e-3


def _input_fn(shape, cur_dtype, device):
    t, hc_mult, hidden_size = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        t,
        hc_mult,
        hidden_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    hc_fn = (
        torch.randn(
            hc_mult,
            hc_mult * hidden_size,
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    )
    hc_scale = torch.tensor([1.5], device=device, dtype=torch.float32)
    hc_base = 0.1 * torch.randn(
        hc_mult, generator=g, device=device, dtype=torch.float32
    )
    yield x, hc_fn, hc_scale, hc_base, _NORM_EPS, _HC_EPS


@pytest.mark.hc_head
def test_perf_hc_head():
    bench = OpBenchmark(
        op_name="hc_head",
        torch_op=get_reference("hc_head"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="t, hc_mult, hidden_size",
    )
    bench.set_gems(flaggems_sglang.hc_head)
    bench.run()
