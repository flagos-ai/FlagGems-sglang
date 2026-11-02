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

"""Benchmark for sglang/sglang_fl_rotary_embedding."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (num_tokens, num_q_heads, num_kv_heads, head_size).
SHAPES = [(t, 32, 8, 128) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(4096, 32, 8, 128)]

_MAX_POSITION = 8192


def _input_fn(shape, cur_dtype, device):
    num_tokens, num_q_heads, num_kv_heads, head_size = shape
    rotary_dim = head_size
    g = torch.Generator(device=device).manual_seed(9)
    positions = torch.randint(
        0, _MAX_POSITION, (num_tokens,), device=device, generator=g
    )
    query = torch.randn(
        num_tokens, num_q_heads * head_size, device=device, generator=g
    ).to(cur_dtype)
    key = torch.randn(
        num_tokens, num_kv_heads * head_size, device=device, generator=g
    ).to(cur_dtype)
    cos_sin_cache = torch.randn(
        _MAX_POSITION, rotary_dim, device=device, generator=g
    )
    yield positions, query, key, head_size, cos_sin_cache, {"is_neox": True}


def _case_fn(shape, cur_dtype):
    num_tokens, num_q_heads, num_kv_heads, head_size = shape
    yield BenchmarkCasePlan(
        shape={
            "query": (num_tokens, num_q_heads * head_size),
            "key": (num_tokens, num_kv_heads * head_size),
            "cos_sin_cache": (_MAX_POSITION, head_size),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
        },
        builder_args=(shape,),
    )


@pytest.mark.sglang_fl_rotary_embedding
def test_perf_sglang_fl_rotary_embedding():
    bench = OpBenchmark(
        op_name="sglang_fl_rotary_embedding",
        torch_op=get_reference("sglang_fl_rotary_embedding"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, num_q_heads, num_kv_heads, head_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
