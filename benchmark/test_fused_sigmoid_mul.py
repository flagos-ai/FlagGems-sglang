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

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/activation_norm/fused_sigmoid_mul.
# The strided path (3D gate over a 2D attn_output) is the production shape,
# so the benchmark sweeps num_tokens with a fixed 32 x 128 head layout.
SHAPES = [(t, 32, 128) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 32, 128) for t in (1024, 4096)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, num_heads, head_dim = shape
    g = torch.Generator(device=device).manual_seed(9)
    attn_output = torch.randn(
        num_tokens, num_heads * head_dim, device=device, generator=g
    ).to(cur_dtype)
    gate = torch.randn(
        num_tokens, num_heads, head_dim, device=device, generator=g
    ).to(cur_dtype)
    yield attn_output, gate


def _case_fn(shape, cur_dtype):
    (num_tokens, num_heads, head_dim) = shape
    yield BenchmarkCasePlan(
        shape={
            "attn_output": (num_tokens, num_heads * head_dim),
            "gate": (num_tokens, num_heads, head_dim),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_sigmoid_mul
def test_perf_fused_sigmoid_mul():
    bench = OpBenchmark(
        op_name="fused_sigmoid_mul",
        torch_op=get_reference("fused_sigmoid_mul"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, num_heads, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
