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

"""Benchmark for attention/pad_draft_extend_query."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/attention/pad_draft_extend_query.
SHAPES = [(bs, 8, 16, 128) for bs in (1, 8, 64, 256)]
MORE_SHAPES = [(bs, 8, 16, 128) for bs in (1024,)]


def _input_fn(shape, cur_dtype, device):
    bs, max_seq_len, num_heads, head_dim = shape
    g = torch.Generator(device=device).manual_seed(7)
    seq_lens = torch.randint(
        1,
        max_seq_len + 1,
        (bs,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(seq_lens, dim=0)
    total = int(cu[-1])
    q = torch.randn(total, num_heads, head_dim, device=device, generator=g).to(
        cur_dtype
    )
    padded_q = torch.zeros(
        bs, max_seq_len, num_heads, head_dim, dtype=cur_dtype, device=device
    )
    yield q, padded_q, seq_lens, cu


def _case_fn(shape, cur_dtype):
    (bs, max_seq_len, num_heads, head_dim) = shape
    yield BenchmarkCasePlan(
        shape={
            "q": ("sum(seq_lens)", num_heads, head_dim),
            "padded_q": (bs, max_seq_len, num_heads, head_dim),
            "seq_lens": (bs,),
            "cu": (bs + 1,),
        },
        params={
            "configuration": shape,
            "seed": 7,
            "bs": bs,
            "max_seq_len": max_seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.pad_draft_extend_query
def test_perf_pad_draft_extend_query():
    bench = OpBenchmark(
        op_name="pad_draft_extend_query",
        torch_op=get_reference("pad_draft_extend_query"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, max_seq_len, num_heads, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
