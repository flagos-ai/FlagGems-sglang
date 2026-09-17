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

"""Benchmark for attention/context_attention."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

IS_CAUSAL = True

# Shapes match kernel-comp-baseline/problems/attention/context_attention;
# every sequence in a batch gets the same length.
SHAPES = [
    (bs, seq_len, 32, 128) for bs in (1, 8, 64) for seq_len in (128, 2048)
]
MORE_SHAPES = [(2, 8, 4, 128), (3, 30, 4, 96), (1, 20, 4, 80), (1, 9, 4, 13)]


def _input_fn(shape, cur_dtype, device):
    bs, seq_len, num_heads, head_dim = shape
    g = torch.Generator(device=device).manual_seed(0)
    seq_lens = [seq_len] * bs
    total = sum(seq_lens)
    q, k, v = (
        torch.randn(
            total,
            num_heads,
            head_dim,
            generator=g,
            device=device,
            dtype=cur_dtype,
        )
        for _ in range(3)
    )
    b_start_loc = torch.zeros(bs, dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(
        torch.tensor(seq_lens[:-1], dtype=torch.int32, device=device), 0
    )
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    yield q, k, v, b_start_loc, b_seq_len, max(seq_lens), IS_CAUSAL


@pytest.mark.context_attention
def test_perf_context_attention():
    bench = OpBenchmark(
        op_name="context_attention",
        torch_op=get_reference("context_attention"),
        input_fn=_input_fn,
        # The prefill reference runs a per-sequence fp32 SDPA.
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, seq_len, num_heads, head_dim",
    )
    bench.set_gems(flaggems_sglang.context_attention)
    bench.run()
