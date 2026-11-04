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

"""Benchmark for linear-attn/chunk_gated_delta_rule_fwd."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (B, T, H, K, V).
SHAPES = [
    (1, 512, 16, 128, 128),
    (1, 2048, 16, 128, 128),
    (2, 2048, 16, 128, 128),
    (1, 4096, 16, 128, 128),
]
MORE_SHAPES = [(4, 4096, 16, 128, 128)]


def _input_fn(shape, cur_dtype, device):
    B, T, H, K, V = shape
    g = torch.Generator(device=device).manual_seed(9)
    q = torch.randn(B, T, H, K, device=device, generator=g).to(cur_dtype)
    k = torch.randn(B, T, H, K, device=device, generator=g).to(cur_dtype)
    v = torch.randn(B, T, H, V, device=device, generator=g).to(cur_dtype)
    # g = logsigmoid(u), matching the operator's own data generation.
    u = torch.rand(B, T, H, device=device, generator=g)
    gate = torch.nn.functional.logsigmoid(u)
    beta = torch.rand(B, T, H, device=device, generator=g)
    scale = K**-0.5
    yield q, k, v, gate, beta, scale, None, False, None, {
        "use_qk_l2norm_in_kernel": True
    }


def _case_fn(shape, cur_dtype):
    B, T, H, K, V = shape
    yield BenchmarkCasePlan(
        shape={
            "q": (B, T, H, K),
            "k": (B, T, H, K),
            "v": (B, T, H, V),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "B": B,
            "T": T,
            "H": H,
            "K": K,
            "V": V,
        },
        builder_args=(shape,),
    )


@pytest.mark.chunk_gated_delta_rule_fwd
def test_perf_chunk_gated_delta_rule_fwd():
    bench = OpBenchmark(
        op_name="chunk_gated_delta_rule_fwd",
        torch_op=get_reference("chunk_gated_delta_rule_fwd"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="B, T, H, K, V",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
