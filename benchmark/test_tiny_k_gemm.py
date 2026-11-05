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

"""Benchmark for gemm/tiny_k_gemm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/gemm/tiny_k_gemm.
SHAPES = [(m, n, 128) for m in (1, 2, 4, 8) for n in (1536, 3072)]
MORE_SHAPES = [(m, n, 128) for m in (16,) for n in (1536, 3072)]


def _input_fn(shape, cur_dtype, device):
    m, n, k = shape
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(m, k, device=device, generator=g).to(cur_dtype)
    w = torch.randn(n, k, device=device, generator=g).to(cur_dtype)
    # out_dtype is a torch.dtype, which unpack_to_args_kwargs forwards
    # positionally.
    yield x, w, cur_dtype


def _case_fn(shape, cur_dtype):
    (m, n, k) = shape
    yield BenchmarkCasePlan(
        shape={"x": (m, k), "w": (n, k)},
        params={"configuration": shape, "seed": 9, "m": m, "n": n, "k": k},
        builder_args=(shape,),
    )


@pytest.mark.tiny_k_gemm
def test_perf_tiny_k_gemm():
    bench = OpBenchmark(
        op_name="tiny_k_gemm",
        torch_op=get_reference("tiny_k_gemm"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, n, k",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
