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

"""Benchmark for moe/dsv3_fused_a_gemm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

SHAPES = [(m, 7168, 2112) for m in (1, 2, 4, 8)]
MORE_SHAPES = [(m, 7168, 2112) for m in (16,)]


def _input_fn(shape, cur_dtype, device):
    m, hd_in, hd_out = shape
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(
        m, hd_in, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    # Column-major B: build the row-major [hd_out, hd_in] weight and transpose.
    w = torch.randn(
        hd_out, hd_in, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    yield a, w.t()


def _case_fn(shape, cur_dtype):
    (m, d, num_experts, top_k) = shape
    yield BenchmarkCasePlan(
        shape={
            "a": (m, d),
            "b": (num_experts, d, d),
            "topk_weights": (m, top_k),
            "topk_ids": (m, top_k),
            "sorted_token_ids": (m,),
            "expert_ids": (num_experts,),
            "num_tokens_post_padded": (num_experts,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "m": m,
            "d": d,
            "num_experts": num_experts,
            "top_k": top_k,
        },
        builder_args=(shape,),
    )


@pytest.mark.dsv3_fused_a_gemm
def test_perf_dsv3_fused_a_gemm():
    bench = OpBenchmark(
        op_name="dsv3_fused_a_gemm",
        torch_op=get_reference("dsv3_fused_a_gemm"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, hd_in, hd_out",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
