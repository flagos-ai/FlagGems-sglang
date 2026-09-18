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

"""Benchmark for quantization/w8a8_block_int8_matmul."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/quantization/w8a8_block_int8_matmul.
SHAPES = [
    (m, n, k)
    for m in (1, 8, 64, 512, 4096)
    for n, k in ((1024, 4096), (4096, 4096), (7168, 4096))
]
MORE_SHAPES = [(7, 256, 512), (64, 1024, 512), (256, 1024, 4096)]

_BLOCK = [128, 128]


def _input_fn(shape, cur_dtype, device):
    m, n, k = shape
    block_n, block_k = _BLOCK
    g = torch.Generator(device=device).manual_seed(0)
    A = torch.randint(
        -8, 8, (m, k), dtype=torch.int8, device=device, generator=g
    )
    B = torch.randint(
        -8, 8, (n, k), dtype=torch.int8, device=device, generator=g
    )
    As = (
        1e-2
        * torch.rand(
            m,
            k // block_k,
            dtype=torch.float32,
            device=device,
            generator=g,
        )
    ).contiguous()
    Bs = (
        1e-2
        * torch.rand(
            n // block_n,
            k // block_k,
            dtype=torch.float32,
            device=device,
            generator=g,
        )
    ).contiguous()
    # block_size is a list and output_dtype a torch.dtype; both survive
    # unpack_to_args_kwargs as positionals.
    yield A, B, As, Bs, _BLOCK, cur_dtype


@pytest.mark.w8a8_block_int8_matmul
def test_perf_w8a8_block_int8_matmul():
    bench = OpBenchmark(
        op_name="w8a8_block_int8_matmul",
        torch_op=get_reference("w8a8_block_int8_matmul"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, n, k",
    )
    bench.set_gems(flaggems_sglang.w8a8_block_int8_matmul)
    bench.run()
