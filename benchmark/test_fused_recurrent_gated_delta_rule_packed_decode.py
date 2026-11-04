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

"""Benchmark for linear-attn/packed decode gated delta rule."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (B, HV, H, K, V).
SHAPES = [
    (1, 32, 16, 128, 128),
    (8, 32, 16, 128, 128),
    (64, 32, 16, 128, 128),
    (256, 32, 16, 128, 128),
]
MORE_SHAPES = [(512, 32, 16, 128, 128)]


def _input_fn(shape, cur_dtype, device):
    B, HV, H, K, V = shape
    num_slots = B + 1
    qk = H * K
    g = torch.Generator(device=device).manual_seed(9)
    mixed_qkv = (
        torch.randn(B, 2 * qk + HV * V, device=device, generator=g) * 0.3
    ).to(cur_dtype)
    a = torch.randn(B, HV, device=device, generator=g).to(cur_dtype)
    b = torch.randn(B, HV, device=device, generator=g).to(cur_dtype)
    A_log = torch.randn(HV, device=device, generator=g)
    dt_bias = torch.randn(HV, device=device, generator=g)
    initial_state = (
        torch.randn(num_slots, HV, V, K, device=device, generator=g) * 0.1
    )
    out = torch.zeros(B, 1, HV, V, device=device, dtype=cur_dtype)
    ssm_state_indices = torch.arange(
        1, B + 1, device=device, dtype=torch.int32
    )
    scale = K**-0.5
    yield mixed_qkv, a, b, A_log, dt_bias, scale, initial_state, out, (
        ssm_state_indices
    ), {"use_qk_l2norm_in_kernel": False}


def _case_fn(shape, cur_dtype):
    B, HV, H, K, V = shape
    qk = H * K
    yield BenchmarkCasePlan(
        shape={
            "mixed_qkv": (B, 2 * qk + HV * V),
            "initial_state": (B + 1, HV, V, K),
            "out": (B, 1, HV, V),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "B": B,
            "HV": HV,
            "H": H,
            "K": K,
            "V": V,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
def test_perf_fused_recurrent_gated_delta_rule_packed_decode():
    bench = OpBenchmark(
        op_name="fused_recurrent_gated_delta_rule_packed_decode",
        torch_op=get_reference(
            "fused_recurrent_gated_delta_rule_packed_decode"
        ),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="B, HV, H, K, V",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
