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

"""Benchmark for moe/moe_topk_sum."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/moe_topk_sum.
SHAPES = [(m, topk, 7168) for m in (1, 8, 64, 512) for topk in (4, 8)]
MORE_SHAPES = [(m, topk, 7168) for m in (4096,) for topk in (4, 8)]


def _input_fn(shape, cur_dtype, device):
    m, topk, k = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, topk, k, device=device, generator=g).to(cur_dtype)
    out = torch.zeros(m, k, dtype=cur_dtype, device=device)
    yield x, out


def _case_fn(shape, cur_dtype):
    (m, topk, k) = shape
    yield BenchmarkCasePlan(
        shape={"x": (m, topk, k), "out": (m, k)},
        params={
            "configuration": shape,
            "seed": 0,
            "m": m,
            "topk": topk,
            "k": k,
        },
        builder_args=(shape,),
    )


@pytest.mark.moe_topk_sum
def test_perf_moe_topk_sum():
    bench = OpBenchmark(
        op_name="moe_topk_sum",
        torch_op=get_reference("moe_topk_sum"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, topk, k",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
