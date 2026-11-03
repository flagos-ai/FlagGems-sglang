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

"""Benchmark for diffusion/one_pass_rms_norm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

SHAPES = [(s, d) for s in (1, 64, 1024) for d in (1536, 3072)]
MORE_SHAPES = [(16384, d) for d in (1536, 3072)]

_EPS = 1e-6


def _input_fn(shape, cur_dtype, device):
    s, d = shape
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(s, d, device=device, generator=g).to(cur_dtype)
    w = torch.randn(d, device=device, generator=g).to(cur_dtype)
    yield x, w, _EPS


def _case_fn(shape, cur_dtype):
    (num_tokens, hidden) = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (num_tokens, hidden),
            "weight": (hidden,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "hidden": hidden,
        },
        builder_args=(shape,),
    )


@pytest.mark.one_pass_rms_norm
def test_perf_one_pass_rms_norm():
    bench = OpBenchmark(
        op_name="one_pass_rms_norm",
        torch_op=get_reference("one_pass_rms_norm"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
