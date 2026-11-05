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

"""Benchmark for attention/concat_mla_absorb_q."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/concat_mla_absorb_q.
SHAPES = [(t, 128, 512, 64) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 128, 512, 64) for t in (4096,)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, num_heads, a_last, b_last = shape
    g = torch.Generator(device=device).manual_seed(9)
    a = torch.randn(
        num_tokens, num_heads, a_last, device=device, generator=g
    ).to(cur_dtype)
    b = torch.randn(
        num_tokens, num_heads, b_last, device=device, generator=g
    ).to(cur_dtype)
    yield a, b


def _case_fn(shape, cur_dtype):
    (num_tokens, num_heads, a_last, b_last) = shape
    yield BenchmarkCasePlan(
        shape={
            "a": (num_tokens, num_heads, a_last),
            "b": (num_tokens, num_heads, b_last),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "num_heads": num_heads,
            "a_last": a_last,
            "b_last": b_last,
        },
        builder_args=(shape,),
    )


@pytest.mark.concat_mla_absorb_q
def test_perf_concat_mla_absorb_q():
    bench = OpBenchmark(
        op_name="concat_mla_absorb_q",
        torch_op=get_reference("concat_mla_absorb_q"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, num_heads, a_last, b_last",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
