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

"""Benchmark for utility/set_token_filter."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/utility/set_token_filter.
SHAPES = [(16, vocab, n) for vocab in (32768,) for n in (16, 1024)]
MORE_SHAPES = [(16, vocab, n) for vocab in (152064,) for n in (16, 1024)]


def _input_fn(shape, cur_dtype, device):
    batch, vocab, num_tokens = shape
    g = torch.Generator(device=device).manual_seed(9)
    words = (vocab + 31) // 32
    vocab_mask = torch.randint(
        -(2**31),
        2**31 - 1,
        (batch, words),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    # token_ids is a Python list, so the permutation stays on the CPU.
    token_ids = torch.randperm(
        vocab, generator=torch.Generator().manual_seed(9)
    )[:num_tokens].tolist()
    yield vocab_mask, token_ids, batch // 2, True, True


def _case_fn(shape, cur_dtype):
    (batch, vocab, num_tokens) = shape
    yield BenchmarkCasePlan(
        shape={"vocab_mask": (batch, (vocab + 31) // 32)},
        params={
            "configuration": shape,
            "seed": 9,
            "batch": batch,
            "vocab": vocab,
            "num_tokens": num_tokens,
        },
        builder_args=(shape,),
    )


@pytest.mark.set_token_filter
def test_perf_set_token_filter():
    bench = OpBenchmark(
        op_name="set_token_filter",
        torch_op=get_reference("set_token_filter"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, vocab, num_tokens",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
