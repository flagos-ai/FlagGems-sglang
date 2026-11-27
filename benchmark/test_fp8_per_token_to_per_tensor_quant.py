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

"""Benchmark for moe/fp8_per_token_to_per_tensor_quant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark
from flaggems_sglang.utils.rng_utils import create_generator

_K_SCALE_BLOCK_SIZE = 128
_TOKENS_PADDED = 256
# (expert_num, k) — k must be a multiple of 1024, the kernel's K_BLOCK_SIZE.
SHAPES = [(e, 4096) for e in (4, 8, 32)]
MORE_SHAPES = [(4, 1024), (8, 1024), (16, 2048)]


def _input_fn(shape, cur_dtype, device):
    expert_num, k = shape
    g = create_generator(device, seed=0)
    num_groups = k // _K_SCALE_BLOCK_SIZE
    x = (
        2
        * torch.randn(
            expert_num,
            _TOKENS_PADDED,
            k,
            generator=g,
            device=device,
        )
    ).to(cur_dtype)
    x_scale = (
        torch.rand(
            expert_num,
            _TOKENS_PADDED,
            num_groups,
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
        + 0.01
    )
    masked_m = torch.full(
        (expert_num,), _TOKENS_PADDED, dtype=torch.int32, device=device
    )
    output_scale = torch.tensor([0.02], dtype=torch.float32, device=device)
    yield dict(
        x=x,
        x_scale=x_scale,
        masked_m=masked_m,
        output_scale=output_scale,
    )


@pytest.mark.fp8_per_token_to_per_tensor_quant
def test_perf_fp8_per_token_to_per_tensor_quant():
    gems_op = flaggems_sglang.get_op("fp8_per_token_to_per_tensor_quant")
    if gems_op is None:
        pytest.skip(
            "moe/fp8_per_token_to_per_tensor_quant not implemented yet"
        )
    bench = OpBenchmark(
        op_name="fp8_per_token_to_per_tensor_quant",
        torch_op=get_reference("fp8_per_token_to_per_tensor_quant"),
        input_fn=_input_fn,
        dtypes=[torch.float8_e4m3fn],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="expert_num, k",
    )
    bench.set_gems(gems_op)
    bench.run()
