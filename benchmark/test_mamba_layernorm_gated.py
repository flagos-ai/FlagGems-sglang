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

"""Benchmark for mamba/layernorm_gated."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

EPS = 1e-5

# Shapes match kernel-comp-baseline/problems/mamba/layernorm_gated.
SHAPES = [(m, 4096, 128) for m in (1, 8, 64, 512, 4096)]
# ``group_size == n`` is equivalent to the single-group default.
MORE_SHAPES = [(1, 64, 64), (37, 256, 64), (83, 512, 128), (4, 1024, 1024)]


def _input_fn(shape, cur_dtype, device):
    m, n, group_size = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, n, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    weight = torch.randn(
        n, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    bias = torch.randn(n, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    z = torch.randn(m, n, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    # ``norm_before_gate``/``is_rms_norm`` keep their defaults; the vendored
    # unpack_to_args_kwargs routes the dict into kwargs.
    yield x, weight, bias, EPS, dict(z=z, group_size=group_size)


@pytest.mark.mamba_layernorm_gated
def test_perf_mamba_layernorm_gated():
    bench = OpBenchmark(
        op_name="mamba_layernorm_gated",
        torch_op=get_reference("mamba_layernorm_gated"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, n, group_size",
    )
    bench.set_gems(flaggems_sglang.mamba_layernorm_gated)
    bench.run()
