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

"""Benchmark for speculative/fill_bonus_tokens."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/speculative/fill_bonus_tokens.
SHAPES = [(batch, 8) for batch in (1, 8, 64, 512)]
MORE_SHAPES = [(batch, 8) for batch in (4096,)]


def _input_fn(shape, cur_dtype, device):
    batch, width = shape
    g = torch.Generator(device=device).manual_seed(9)
    tokens = torch.randint(
        0,
        128000,
        (batch, width),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    lens = torch.randint(
        1,
        width + 1,
        (batch,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    bonus = torch.zeros(batch, dtype=torch.int64, device=device)
    yield tokens, lens, bonus, width


def _case_fn(shape, cur_dtype):
    (batch, width) = shape
    yield BenchmarkCasePlan(
        shape={"tokens": (batch, width), "lens": (batch,), "bonus": (batch,)},
        params={
            "configuration": shape,
            "seed": 9,
            "batch": batch,
            "width": width,
        },
        builder_args=(shape,),
    )


@pytest.mark.fill_bonus_tokens
def test_perf_fill_bonus_tokens():
    bench = OpBenchmark(
        op_name="fill_bonus_tokens",
        torch_op=get_reference("fill_bonus_tokens"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, accept_stride",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
