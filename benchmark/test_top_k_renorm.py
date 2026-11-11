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

"""Benchmark for sampling/top_k_renorm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (batch, vocab, k). The vocabulary spans many blocks, so the bisection sweeps
# are the cost that matters here rather than the row count.
SHAPES = [
    (1, 32000, 100),
    (128, 32000, 100),
    (1, 128256, 10),
]

# A single block, a mid-sized row, and the k extremes, so the case where the
# whole row fits at once is timed separately from the many-block path.
MORE_SHAPES = [
    (1, 111, 10),
    (64, 1024, 100),
    (8, 32000, 500),
    (1, 130000, 1),
]


def _input_fn(shape, cur_dtype, device):
    batch, vocab, k = shape
    g = torch.Generator(device=device).manual_seed(0)
    raw = torch.rand((batch, vocab), generator=g, device=device)
    probs = raw / raw.sum(dim=-1, keepdim=True)
    yield probs.to(cur_dtype), k


def _case_fn(shape, cur_dtype):
    batch, vocab, k = shape
    yield BenchmarkCasePlan(
        shape={"probs": (batch, vocab)},
        params={
            "configuration": shape,
            "batch": batch,
            "vocab": vocab,
            "top_k": k,
        },
        builder_args=(shape,),
    )


@pytest.mark.top_k_renorm
def test_perf_top_k_renorm():
    bench = OpBenchmark(
        op_name="top_k_renorm",
        torch_op=get_reference("top_k_renorm"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, vocab, k",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
