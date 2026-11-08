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

"""Benchmark for kv_cache/reduce_head_scores."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

NUM_HEADS = 32

_CORE_SHAPES = ((64, 2048), (512, 2048), (2048, 2048), (4096, 1024))
SHAPES = list(_CORE_SHAPES)
MORE_SHAPES = [(8, 256), (1, 1024), (2048, 4096)]


def _input_fn(shape, dtype, device):
    rows, width = shape
    g = torch.Generator(device=device).manual_seed(0)
    scores = torch.randn(
        rows, NUM_HEADS, width, generator=g, device=device
    ).to(torch.bfloat16)
    weights = torch.randn(rows, NUM_HEADS, generator=g, device=device).to(
        torch.bfloat16
    )
    yield scores, {
        "weights": weights,
    }


def _case_fn(shape, cur_dtype):
    rows, width = shape
    yield BenchmarkCasePlan(
        shape={"scores": (rows, NUM_HEADS, width)},
        params={
            "configuration": shape,
            "seed": 0,
            "rows": rows,
            "width": width,
        },
        builder_args=(shape,),
    )


@pytest.mark.reduce_head_scores
def test_perf_reduce_head_scores():
    bench = OpBenchmark(
        op_name="reduce_head_scores",
        torch_op=get_reference("reduce_head_scores"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="rows, width",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
