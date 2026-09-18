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

"""Benchmark for attention/log_scaling_tau."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/log_scaling_tau.
SHAPES = [(t, 16, 128) for t in (1, 128, 2048, 8192)]
# (37, 8, 64) and (129, 512) from the correctness cases; the 2-D case is
# padded with a 1 so every shape tuple has the same arity.
MORE_SHAPES = [(1, 1, 128), (37, 8, 64), (129, 1, 512)]


def _input_fn(shape, cur_dtype, device):
    g = torch.Generator(device=device).manual_seed(0)
    # A leading 1 in the middle position marks a genuinely 2-D case.
    dims = tuple(d for i, d in enumerate(shape) if not (i == 1 and d == 1))
    x = torch.randn(*dims, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    # tau is per-row (one entry per index along dim 0).
    tau = 0.5 + torch.rand(
        shape[0], generator=g, device=device, dtype=torch.float32
    )
    yield x, tau


@pytest.mark.log_scaling_tau
def test_perf_log_scaling_tau():
    bench = OpBenchmark(
        op_name="log_scaling_tau",
        torch_op=get_reference("log_scaling_tau"),
        input_fn=_input_fn,
        dtypes=[torch.float16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="t, heads, d",
    )
    bench.set_gems(flaggems_sglang.log_scaling_tau)
    bench.run()
