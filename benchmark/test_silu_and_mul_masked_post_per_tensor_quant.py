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

"""Benchmark for moe/silu_and_mul_masked_post_per_tensor_quant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark
from flaggems_sglang.utils.rng_utils import create_generator

_TOKENS_PADDED = 256
# (expert_num, hidden) — one scalar scale for the whole call.
SHAPES = [(e, 4096) for e in (4, 8, 32)]
MORE_SHAPES = [(4, 32), (8, 256), (16, 1024)]


def _input_fn(shape, cur_dtype, device):
    expert_num, hidden = shape
    g = create_generator(device, seed=0)
    inp = torch.randn(
        expert_num,
        _TOKENS_PADDED,
        hidden * 2,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    masked_m = torch.full(
        (expert_num,), _TOKENS_PADDED, dtype=torch.int32, device=device
    )
    scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    yield dict(input=inp, masked_m=masked_m, scale=scale)


@pytest.mark.silu_and_mul_masked_post_per_tensor_quant
def test_perf_silu_and_mul_masked_post_per_tensor_quant():
    gems_op = flaggems_sglang.get_op(
        "silu_and_mul_masked_post_per_tensor_quant"
    )
    if gems_op is None:
        pytest.skip(
            "moe/silu_and_mul_masked_post_per_tensor_quant not implemented yet"
        )
    bench = OpBenchmark(
        op_name="silu_and_mul_masked_post_per_tensor_quant",
        torch_op=get_reference("silu_and_mul_masked_post_per_tensor_quant"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="expert_num, hidden",
    )
    bench.set_gems(gems_op)
    bench.run()
