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

"""Benchmark for mamba/causal_conv1d_fn."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/causal_conv1d_fn.
SHAPES = [(8, 2048, 4096, 4), (32, 512, 2048, 4)]
MORE_SHAPES = [(1, 7, 16, 4), (4, 1, 8, 3)]


def _input_fn(shape, cur_dtype, device):
    num_seqs, seq_len, dim, width = shape
    g = torch.Generator(device=device).manual_seed(0)
    seq_lens = [seq_len] * num_seqs
    x = torch.randn(
        dim,
        sum(seq_lens),
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    weight = torch.randn(
        dim, width, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    bias = torch.randn(
        dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    query_start_loc = torch.zeros(
        num_seqs + 1, dtype=torch.int32, device=device
    )
    query_start_loc[1:] = torch.cumsum(
        torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0
    )
    # ``activation`` keeps its "silu" default: the vendored
    # unpack_to_args_kwargs drops bare str positionals.
    yield x, weight, bias, query_start_loc, seq_lens


@pytest.mark.causal_conv1d_fn
def test_perf_causal_conv1d_fn():
    bench = OpBenchmark(
        op_name="causal_conv1d_fn",
        torch_op=get_reference("causal_conv1d_fn"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_seqs, seq_len, dim, width",
    )
    bench.set_gems(flaggems_sglang.causal_conv1d_fn)
    bench.run()
