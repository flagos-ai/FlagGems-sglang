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

"""Benchmark for moe/silu_and_mul_masked."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/silu_and_mul_masked. The
# last entry is the per-expert valid token count; the benchmark shapes run
# every expert full so the timing covers the whole padded buffer.
SHAPES = [(e, 256, 4096, (256,) * e) for e in (4, 8, 32)]
MORE_SHAPES = [
    (4, 16, 32, (0, 3, 16, 9)),
    (8, 128, 256, (128, 0, 64, 1, 17, 128, 5, 90)),
]


def _input_fn(shape, cur_dtype, device):
    expert_num, token_num_padded, hidden_dim, counts = shape
    g = torch.Generator(device=device).manual_seed(0)
    input = torch.randn(
        expert_num,
        token_num_padded,
        hidden_dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    masked_m = torch.tensor(list(counts), dtype=torch.int32, device=device)
    yield input, masked_m


def _case_fn(shape, cur_dtype):
    (expert_num, token_num_padded, hidden_dim, counts) = shape
    yield BenchmarkCasePlan(
        shape={
            "input": (expert_num, token_num_padded, hidden_dim),
            "masked_m": (len(counts),),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "expert_num": expert_num,
            "token_num_padded": token_num_padded,
            "hidden_dim": hidden_dim,
            "counts": counts,
        },
        builder_args=(shape,),
    )


@pytest.mark.silu_and_mul_masked
def test_perf_silu_and_mul_masked():
    bench = OpBenchmark(
        op_name="silu_and_mul_masked",
        torch_op=get_reference("silu_and_mul_masked"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="expert_num, token_num_padded, hidden_dim, masked_m",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
