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

"""Benchmark for utility/vocab_parallel_embedding."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/utility/vocab_parallel_embedding.
SHAPES = [(t, h) for t in (1, 8, 64, 512) for h in (2048, 7168)]
MORE_SHAPES = [(t, h) for t in (4096,) for h in (2048, 7168)]

_ORG_START, _ORG_END = 1000, 5000
_PADDING = 8
_ADDED_START, _ADDED_END = 20000, 20064


def _input_fn(shape, cur_dtype, device):
    num_tokens, hidden = shape
    g = torch.Generator(device=device).manual_seed(9)
    local_vocab = (
        (_ORG_END - _ORG_START) + _PADDING + (_ADDED_END - _ADDED_START)
    )
    weight = torch.randn(local_vocab, hidden, device=device, generator=g).to(
        cur_dtype
    )
    # A mix of in-window, added-window and out-of-window tokens.
    tokens = torch.randint(
        0,
        30000,
        (num_tokens,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    tokens[::3] = torch.randint(
        _ORG_START,
        _ORG_END,
        (tokens[::3].numel(),),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    tokens[1::3] = torch.randint(
        _ADDED_START,
        _ADDED_END,
        (tokens[1::3].numel(),),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    yield (
        tokens.contiguous(),
        weight,
        _ORG_START,
        _ORG_END,
        _PADDING,
        _ADDED_START,
        _ADDED_END,
    )


def _case_fn(shape, cur_dtype):
    (num_tokens, hidden) = shape
    yield BenchmarkCasePlan(
        shape={
            "tokens": (num_tokens,),
            "weight": (
                _ORG_END - _ORG_START + _PADDING + _ADDED_END - _ADDED_START,
                hidden,
            ),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_PADDING": _PADDING,
            "num_tokens": num_tokens,
            "hidden": hidden,
        },
        builder_args=(shape,),
    )


@pytest.mark.vocab_parallel_embedding
def test_perf_vocab_parallel_embedding():
    bench = OpBenchmark(
        op_name="vocab_parallel_embedding",
        torch_op=get_reference("vocab_parallel_embedding"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
