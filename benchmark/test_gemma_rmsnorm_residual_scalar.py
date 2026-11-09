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

"""Benchmark for activation_norm/gemma_rmsnorm_residual_scalar."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, n) for m in (1, 8, 64, 512) for n in (1024, 4096, 8192)]
MORE_SHAPES = [(4096, n) for n in (1024, 4096, 8192)]

_EPS = 1e-6


def _input_fn(shape, cur_dtype, device):
    m, n = shape
    g = torch.Generator(device=device).manual_seed(9)

    def _t(*s):
        return torch.randn(
            *s, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    yield (
        _t(m, n),
        _t(n),
        _t(m, n),
        torch.full((1,), 0.75, dtype=torch.float32, device=device),
        _EPS,
    )


@pytest.mark.gemma_rmsnorm_residual_scalar
def test_perf_gemma_rmsnorm_residual_scalar():
    bench = OpBenchmark(
        op_name="gemma_rmsnorm_residual_scalar",
        torch_op=get_reference("gemma_rmsnorm_residual_scalar"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems(flaggems_sglang.gemma_rmsnorm_residual_scalar)
    bench.run()
