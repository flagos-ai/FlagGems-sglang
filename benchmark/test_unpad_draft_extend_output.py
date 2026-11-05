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

"""Benchmark for attention/unpad_draft_extend_output."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/
# unpad_draft_extend_output.
SHAPES = [(bs, 8, 16, 128) for bs in (1, 8, 64, 256)]
MORE_SHAPES = [(bs, 8, 16, 128) for bs in (1024,)]


def _input_fn(shape, cur_dtype, device):
    bs, token_per_batch, num_heads, head_dim = shape
    g = torch.Generator(device=device).manual_seed(7)
    seq_lens = torch.randint(
        1,
        token_per_batch + 1,
        (bs,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(seq_lens, dim=0)
    raw_out = torch.randn(
        bs, token_per_batch, num_heads, head_dim, device=device, generator=g
    ).to(cur_dtype)
    yield raw_out, cu, seq_lens, int(cu[-1])


def _case_fn(shape, cur_dtype):
    (bs, token_per_batch, num_heads, head_dim) = shape
    yield BenchmarkCasePlan(
        shape={
            "raw_out": (bs, token_per_batch, num_heads, head_dim),
            "cu": (bs + 1,),
            "seq_lens": (bs,),
        },
        params={
            "configuration": shape,
            "seed": 7,
            "bs": bs,
            "token_per_batch": token_per_batch,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.unpad_draft_extend_output
def test_perf_unpad_draft_extend_output():
    bench = OpBenchmark(
        op_name="unpad_draft_extend_output",
        torch_op=get_reference("unpad_draft_extend_output"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, token_per_batch, num_heads, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
