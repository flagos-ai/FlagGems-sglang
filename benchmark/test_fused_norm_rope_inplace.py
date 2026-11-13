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

"""Benchmark for norm/fused_norm_rope_inplace."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (tokens, heads, head_dim, rotary_dim).
SHAPES = [
    (2048, 8, 128, 64),
    (2048, 32, 128, 128),
]

# The full-rotation and small-rotation cases bound the fused rope segment, and
# the narrow head isolates launch cost.
MORE_SHAPES = [
    (512, 8, 64, 64),
    (8192, 8, 64, 32),
    (1, 8, 128, 64),
]


def _input_fn(shape, cur_dtype, device):
    tokens, heads, head_dim, rotary = shape
    npairs = rotary // 2
    g = torch.Generator(device=device).manual_seed(0)
    ang = torch.rand(tokens, npairs, device=device, generator=g) * 6.2831853
    freqs = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
    kv = torch.randn(
        (tokens, heads, head_dim),
        generator=g,
        device=device,
        dtype=torch.float32,
    )
    weight = torch.randn(
        head_dim, generator=g, device=device, dtype=torch.float32
    )
    positions = torch.arange(tokens, device=device)
    yield (
        kv.to(cur_dtype),
        weight.to(cur_dtype),
        1e-6,
        freqs,
        positions,
    )


def _case_fn(shape, cur_dtype):
    tokens, heads, head_dim, rotary = shape
    yield BenchmarkCasePlan(
        shape={"kv": (tokens, heads, head_dim)},
        params={
            "configuration": shape,
            "tokens": tokens,
            "heads": heads,
            "head_dim": head_dim,
            "rotary_dim": rotary,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_norm_rope_inplace
def test_perf_fused_norm_rope_inplace():
    bench = OpBenchmark(
        op_name="fused_norm_rope_inplace",
        torch_op=get_reference("fused_norm_rope_inplace"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, heads, head_dim, rotary_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
