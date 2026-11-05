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

"""Benchmark for diffusion/ltx2_split_rotary."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/diffusion/ltx2_split_rotary.
SHAPES = [(1, t, 16, 128) for t in (1, 64, 1024)]
MORE_SHAPES = [(1, t, 16, 128) for t in (8192,)]


def _input_fn(shape, cur_dtype, device):
    batch, seq_len, num_heads, head_dim = shape
    half = head_dim // 2
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(
        batch, seq_len, num_heads * head_dim, device=device, generator=g
    ).to(cur_dtype)
    ang = torch.randn(
        batch, num_heads, seq_len, half, device=device, generator=g
    )
    yield (
        x,
        torch.cos(ang).to(cur_dtype),
        torch.sin(ang).to(cur_dtype),
    )


def _case_fn(shape, cur_dtype):
    (batch, seq_len, num_heads, head_dim) = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (batch, seq_len, num_heads * head_dim),
            "cos": (batch, num_heads, seq_len, head_dim // 2),
            "sin": (batch, num_heads, seq_len, head_dim // 2),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "batch": batch,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.ltx2_split_rotary
def test_perf_ltx2_split_rotary():
    bench = OpBenchmark(
        op_name="ltx2_split_rotary",
        torch_op=get_reference("ltx2_split_rotary"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, seq_len, num_heads, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
