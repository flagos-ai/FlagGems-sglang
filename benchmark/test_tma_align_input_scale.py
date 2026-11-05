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

"""Benchmark for moe/tma_align_input_scale."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/tma_align_input_scale.
SHAPES = [(m, 56) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(m, 56) for m in (16384,)]


def _input_fn(shape, cur_dtype, device):
    m, k_div_block = shape
    g = torch.Generator(device=device).manual_seed(9)
    # input_scale is an fp32 activation-scale tensor regardless of cur_dtype.
    input_scale = torch.rand(
        m, k_div_block, dtype=torch.float32, device=device, generator=g
    )
    yield (input_scale,)


def _case_fn(shape, cur_dtype):
    (m, k_div_block) = shape
    yield BenchmarkCasePlan(
        shape={"input_scale": (m, k_div_block)},
        params={
            "configuration": shape,
            "seed": 9,
            "m": m,
            "k_div_block": k_div_block,
        },
        builder_args=(shape,),
    )


@pytest.mark.tma_align_input_scale
def test_perf_tma_align_input_scale():
    bench = OpBenchmark(
        op_name="tma_align_input_scale",
        torch_op=get_reference("tma_align_input_scale"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, k_div_block",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
