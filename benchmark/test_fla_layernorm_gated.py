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

"""Benchmark for fla/layernorm_gated."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/fla/layernorm_gated.
SHAPES = [(t, 2048, "swish", 1, 1) for t in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [
    (1, 64, "swish", 1, 1),
    (37, 256, "sigmoid", 1, 1),
    (83, 512, "swish", 0, 0),
]

_EPS = 1e-5


def _input_fn(shape, cur_dtype, device):
    t, d, activation, is_rms_norm, has_bias = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(t, d, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    gate = torch.randn(
        t, d, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    weight = torch.randn(
        d, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    bias = None
    if has_bias:
        bias = torch.randn(
            d, generator=g, device=device, dtype=torch.float32
        ).to(cur_dtype)
    # ``activation`` is a str, so it rides in a trailing dict along with the
    # remaining keyword-only style options.
    yield x, gate, weight, bias, {
        "activation": activation,
        "eps": _EPS,
        "is_rms_norm": bool(is_rms_norm),
    }


@pytest.mark.fla_layernorm_gated
def test_perf_fla_layernorm_gated():
    bench = OpBenchmark(
        op_name="fla_layernorm_gated",
        torch_op=get_reference("fla_layernorm_gated"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="t, d, activation, is_rms_norm, has_bias",
    )
    bench.set_gems(flaggems_sglang.fla_layernorm_gated)
    bench.run()
