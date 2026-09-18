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

"""Benchmark for mamba/causal_conv1d_update."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/causal_conv1d_update.
SHAPES = [(64, 4096, 4, 1), (256, 2048, 4, 1)]
MORE_SHAPES = [(2, 16, 4, 1), (5, 64, 4, 1), (3, 32, 3, 4)]


def _input_fn(shape, cur_dtype, device):
    batch, dim, width, seqlen = shape
    g = torch.Generator(device=device).manual_seed(0)
    state_len = width - 1
    x = torch.randn(
        batch,
        dim,
        seqlen,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    if seqlen == 1:
        x = x.squeeze(-1)
    conv_state = torch.randn(
        batch,
        dim,
        state_len,
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
    # ``activation`` keeps its "silu" default: unpack_to_args_kwargs drops
    # bare str positionals.
    yield x, conv_state, weight, bias


@pytest.mark.causal_conv1d_update
def test_perf_causal_conv1d_update():
    bench = OpBenchmark(
        op_name="causal_conv1d_update",
        torch_op=get_reference("causal_conv1d_update"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, dim, width, seqlen",
    )
    bench.set_gems(flaggems_sglang.causal_conv1d_update)
    bench.run()
