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

"""Benchmark for mamba/chunk_state_varlen."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference.chunk_cumsum import (
    reference as chunk_cumsum_reference,
)

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/chunk_state_varlen. The
# first entry is the per-sequence token count; every running cumulative sum
# that ends a chunk has to land exactly on a chunk_size multiple, so no
# sequence straddles a chunk boundary (this problem's documented scope).
SHAPES = [
    ((256,) * 8, 256, 32, 8, 64, 128),
    ((256,) * 32, 256, 64, 8, 64, 128),
]
MORE_SHAPES = [
    ((3, 5), 8, 4, 2, 16, 8),
    ((10, 6, 4, 12), 16, 8, 2, 32, 16),
    ((20, 12, 5, 27, 32), 32, 16, 4, 64, 32),
]


def _input_fn(shape, cur_dtype, device):
    seq_lens, chunk_size, nheads, ngroups, headdim, dstate = shape
    g = torch.Generator(device=device).manual_seed(0)

    cu_list = [0]
    for length in seq_lens:
        cu_list.append(cu_list[-1] + length)
    total_seqlen = cu_list[-1]
    nchunks = total_seqlen // chunk_size
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int32, device=device)

    x = torch.randn(
        total_seqlen,
        nheads,
        headdim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    b = torch.randn(
        total_seqlen,
        ngroups,
        dstate,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    raw_dt = torch.rand(
        1,
        total_seqlen,
        nheads,
        generator=g,
        device=device,
        dtype=torch.float32,
    )
    a = (
        -torch.rand(nheads, generator=g, device=device, dtype=torch.float32)
        - 0.1
    )
    # dt/dA_cumsum come from the chunk_cumsum stage that precedes this op.
    dt_out, dA_cumsum = chunk_cumsum_reference(raw_dt, a, chunk_size)
    chunk_states = torch.randn(
        nchunks,
        nheads,
        headdim,
        dstate,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    yield b, x, dt_out.squeeze(0), dA_cumsum.squeeze(
        0
    ), cu_seqlens, chunk_states


@pytest.mark.chunk_state_varlen
def test_perf_chunk_state_varlen():
    bench = OpBenchmark(
        op_name="chunk_state_varlen",
        torch_op=get_reference("chunk_state_varlen"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seq_lens, chunk_size, nheads, ngroups, headdim, dstate",
    )
    bench.set_gems(flaggems_sglang.chunk_state_varlen)
    bench.run()
