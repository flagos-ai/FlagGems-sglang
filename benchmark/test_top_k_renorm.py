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

"""Benchmark for sampling_grammar/top_k_renorm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark
from flaggems_sglang.utils.rng_utils import create_generator

# (batch, vocab, top_k) — top_k is a Python int here; the per-row tensor form
# is covered by the correctness cases.
SHAPES = [
    (1, 111, 1),
    (7, 111, 10),
    (8, 32000, 100),
    (32, 128256, 500),
    (4, 4096, 4096),
]
MORE_SHAPES = [(16, 4096, 64), (64, 32000, 32), (2, 7, 7)]


def _input_fn(shape, cur_dtype, device):
    batch, vocab, k = shape
    g = create_generator("cpu", seed=0)
    probs = torch.rand((batch, vocab), generator=g)
    probs = (probs / probs.sum(-1, keepdim=True)).to(device).to(cur_dtype)
    yield dict(probs=probs, top_k=k)


@pytest.mark.top_k_renorm
def test_perf_top_k_renorm():
    gems_op = flaggems_sglang.get_op("top_k_renorm")
    if gems_op is None:
        pytest.skip("sampling_grammar/top_k_renorm not implemented yet")
    bench = OpBenchmark(
        op_name="top_k_renorm",
        torch_op=get_reference("top_k_renorm"),
        input_fn=_input_fn,
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, vocab, top_k",
    )
    bench.set_gems(gems_op)
    bench.run()
