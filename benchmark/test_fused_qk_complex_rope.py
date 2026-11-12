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

"""Benchmark for rope/fused_qk_complex_rope."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (tokens, q_heads, k_heads, head_dim).
SHAPES = [
    (2048, 16, 16, 128),
    (2048, 32, 8, 128),
]

# The GQA asymmetry and the narrow head bound the broadcast pattern and the
# launch floor.
MORE_SHAPES = [
    (512, 8, 8, 64),
    (4096, 8, 8, 64),
    (1, 8, 8, 64),
]


def _input_fn(shape, cur_dtype, device):
    tokens, hq, hk, head_dim = shape
    pairs = head_dim // 2
    g = torch.Generator(device=device).manual_seed(0)
    ang = torch.rand(tokens, pairs, device=device, generator=g) * 6.2831853
    freqs = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
    q = torch.randn(
        (tokens, hq, head_dim),
        generator=g,
        device=device,
        dtype=torch.float32,
    )
    k = torch.randn(
        (tokens, hk, head_dim),
        generator=g,
        device=device,
        dtype=torch.float32,
    )
    yield q.to(cur_dtype), k.to(cur_dtype), freqs


def _case_fn(shape, cur_dtype):
    tokens, hq, hk, head_dim = shape
    yield BenchmarkCasePlan(
        shape={"q": (tokens, hq, head_dim), "k": (tokens, hk, head_dim)},
        params={
            "configuration": shape,
            "tokens": tokens,
            "q_heads": hq,
            "k_heads": hk,
            "head_dim": head_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_qk_complex_rope
def test_perf_fused_qk_complex_rope():
    bench = OpBenchmark(
        op_name="fused_qk_complex_rope",
        torch_op=get_reference("fused_qk_complex_rope"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, q_heads, k_heads, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
