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

"""Benchmark for fla/kda_prologue."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(n, 16, 128) for n in (64, 512, 2048)]
MORE_SHAPES = [(8192, 16, 128)]

_BT = 64


def _input_fn(shape, cur_dtype, device):
    seqlen, hv, kdim = shape
    g = torch.Generator(device=device).manual_seed(9)
    q = torch.randn(
        seqlen, hv, kdim, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    k = torch.randn(
        seqlen, hv, kdim, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    g_act = (
        -0.02
        * torch.rand(
            seqlen,
            hv,
            kdim,
            dtype=torch.float32,
            device=device,
            generator=g,
        )
    ).to(cur_dtype)
    cu = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    num_chunks = seqlen // _BT
    chunk_indices = torch.stack(
        [
            torch.zeros(num_chunks, dtype=torch.int32, device=device),
            torch.arange(num_chunks, dtype=torch.int32, device=device),
        ],
        dim=1,
    )
    yield q, k, g_act, 0.125, cu, chunk_indices, num_chunks


@pytest.mark.kda_prologue
def test_perf_kda_prologue():
    bench = OpBenchmark(
        op_name="kda_prologue",
        torch_op=get_reference("kda_prologue"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seqlen, heads, head_dim",
    )
    bench.set_gems(flaggems_sglang.kda_prologue)
    bench.run()
