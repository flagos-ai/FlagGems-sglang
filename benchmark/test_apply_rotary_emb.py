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

"""Benchmark for embedding/apply_rotary_emb."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (tokens, heads, head_dim, use_positions, inverse).
SHAPES = [
    (2048, 32, 128, True, False),
    (2048, 8, 64, True, False),
]

# The position-free variant times the identity row selection, and the inverse
# flag exercises the conjugate path.
MORE_SHAPES = [
    (8192, 8, 128, False, False),
    (4096, 16, 64, False, True),
    (512, 8, 64, True, True),
    (1, 8, 64, True, False),
]


def _input_fn(shape, cur_dtype, device):
    tokens, heads, head_dim, use_pos, inverse = shape
    pairs = head_dim // 2
    g = torch.Generator(device=device).manual_seed(0)
    ang = torch.rand(tokens, pairs, device=device, generator=g) * 6.2831853
    freqs = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
    x = torch.randn(
        (tokens, heads, head_dim),
        generator=g,
        device=device,
        dtype=torch.float32,
    )
    positions = (
        torch.arange(tokens, device=device) if use_pos else None
    )
    yield x.to(cur_dtype), freqs, positions, inverse


def _case_fn(shape, cur_dtype):
    tokens, heads, head_dim, use_pos, inverse = shape
    yield BenchmarkCasePlan(
        shape={"x": (tokens, heads, head_dim)},
        params={
            "configuration": shape,
            "tokens": tokens,
            "heads": heads,
            "head_dim": head_dim,
            "use_positions": use_pos,
            "inverse": inverse,
        },
        builder_args=(shape,),
    )


@pytest.mark.apply_rotary_emb
def test_perf_apply_rotary_emb():
    bench = OpBenchmark(
        op_name="apply_rotary_emb",
        torch_op=get_reference("apply_rotary_emb"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, heads, head_dim, use_positions, inverse",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
