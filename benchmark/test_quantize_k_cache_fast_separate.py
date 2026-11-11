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

"""Benchmark for kv_cache/quantize_k_cache_fast_separate."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (tokens, dim_nope, dim_rope). The first two are the DSA geometry the operator
# ships for at a decode-sized and a prefill-sized token count.
SHAPES = [
    (1024, 512, 64),
    (4096, 512, 64),
    (64, 512, 64),
]

# Small token counts, a NoPE-only row, and a wider NoPE with a wider rotary
# segment, so every segment length gets timed.
MORE_SHAPES = [
    (1, 512, 64),
    (16, 512, 64),
    (8, 256, 0),
    (8, 1024, 128),
]


def _input_fn(shape, cur_dtype, device):
    tokens, dim_nope, dim_rope = shape
    g = torch.Generator(device=device).manual_seed(0)

    k_nope = torch.randn(tokens, dim_nope, generator=g, device=device).to(
        cur_dtype
    )
    k_rope = torch.randn(tokens, dim_rope, generator=g, device=device).to(
        cur_dtype
    )

    yield k_nope, k_rope, 128


def _case_fn(shape, cur_dtype):
    tokens, dim_nope, dim_rope = shape
    yield BenchmarkCasePlan(
        shape={
            "k_nope": (tokens, dim_nope),
            "k_rope": (tokens, dim_rope),
        },
        params={
            "configuration": shape,
            "tokens": tokens,
            "dim_nope": dim_nope,
            "dim_rope": dim_rope,
            "group_size": 128,
        },
        builder_args=(shape,),
    )


@pytest.mark.quantize_k_cache_fast_separate
def test_perf_quantize_k_cache_fast_separate():
    bench = OpBenchmark(
        op_name="quantize_k_cache_fast_separate",
        torch_op=get_reference("quantize_k_cache_fast_separate"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, dim_nope, dim_rope",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
