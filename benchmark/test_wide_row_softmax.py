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

"""Benchmark for speculative/wide_row_softmax."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/speculative/wide_row_softmax.
SHAPES = [(b, v) for b in (1, 4) for v in (32000, 131072)]
MORE_SHAPES = [(b, v) for b in (32,) for v in (32000, 131072)]


def _input_fn(shape, cur_dtype, device):
    bs, vocab = shape
    g = torch.Generator(device=device).manual_seed(9)
    logits = (
        torch.randn(bs, vocab, dtype=torch.float32, device=device, generator=g)
        * 4
    )
    out = torch.full((bs, vocab), -1.0, dtype=torch.float32, device=device)
    yield logits.contiguous(), None, out


def _case_fn(shape, cur_dtype):
    (bs, vocab) = shape
    yield BenchmarkCasePlan(
        shape={"logits": (bs, vocab), "out": (bs, vocab)},
        params={"configuration": shape, "seed": 9, "bs": bs, "vocab": vocab},
        builder_args=(shape,),
    )


@pytest.mark.wide_row_softmax
def test_perf_wide_row_softmax():
    bench = OpBenchmark(
        op_name="wide_row_softmax",
        torch_op=get_reference("wide_row_softmax"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, vocab",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
