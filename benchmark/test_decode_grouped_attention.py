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

"""Benchmark for attention/decode_grouped_attention."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/attention/decode_grouped_attention. The KV
# buffer scales with B * seq_len, so the generator builds one shape's tensors
# at a time rather than holding every case in memory.
SHAPES = [
    (B, 128, 1, 128, seq_len)
    for B, seq_len in (
        (1, 2048),
        (8, 2048),
        (64, 512),
        (512, 128),
        (4096, 128),
    )
]
# H_Q:H_KV ratios stay well above the MHA/small-GQA range already exercised by
# attention/decode_attention, so these route through the grouped kernel
# (kv_group_num > 1) with a realistic MLA-style head count.
MORE_SHAPES = [
    (2, 32, 1, 64, 10),
    (2, 128, 1, 64, 10),
    (2, 32, 4, 80, 10),
    (2, 128, 8, 512, 128),
]


def _input_fn(shape, cur_dtype, device):
    B, H_Q, H_KV, D, seq_len = shape
    g = torch.Generator(device=device).manual_seed(0)
    total_tokens = B * seq_len
    q = torch.randn(
        B, H_Q, D, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    k_buffer = torch.randn(
        total_tokens, H_KV, D, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    v_buffer = torch.randn(
        total_tokens, H_KV, D, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    # Every request holds seq_len contiguous KV slots.
    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(
        torch.full((B,), seq_len, dtype=torch.int32, device=device), dim=0
    )
    kv_indices = torch.arange(total_tokens, device=device, dtype=torch.int32)
    yield q, k_buffer, v_buffer, kv_indptr, kv_indices, 1.0 / (D**0.5)


def _case_fn(shape, cur_dtype):
    (B, H_Q, H_KV, D, seq_len) = shape
    yield BenchmarkCasePlan(
        shape={
            "q": (B, H_Q, D),
            "k_buffer": (B * seq_len, H_KV, D),
            "v_buffer": (B * seq_len, H_KV, D),
            "kv_indptr": (B + 1,),
            "kv_indices": (B * seq_len,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "B": B,
            "H_Q": H_Q,
            "H_KV": H_KV,
            "D": D,
            "seq_len": seq_len,
        },
        builder_args=(shape,),
    )


@pytest.mark.decode_grouped_attention
def test_perf_decode_grouped_attention():
    bench = OpBenchmark(
        op_name="decode_grouped_attention",
        torch_op=get_reference("decode_grouped_attention"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="B, H_Q, H_KV, D, seq_len",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
