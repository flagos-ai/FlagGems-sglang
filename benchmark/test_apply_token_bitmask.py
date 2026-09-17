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

"""Benchmark for sampling_grammar/apply_token_bitmask."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/sampling_grammar/apply_token_bitmask.
SHAPES = [(b, 152064) for b in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(3, 100), (7, 1000), (1, 32000)]


def _input_fn(shape, cur_dtype, device):
    B, V = shape
    g = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(B, V, generator=g, device=device, dtype=cur_dtype)
    words = (V + 31) // 32
    bitmask = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (B, words),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    yield logits, bitmask


@pytest.mark.apply_token_bitmask
def test_perf_apply_token_bitmask():
    bench = OpBenchmark(
        op_name="apply_token_bitmask",
        torch_op=get_reference("apply_token_bitmask"),
        input_fn=_input_fn,
        # The op masks logits in place of a softmax, so it runs in fp32.
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="B, V",
    )
    bench.set_gems(flaggems_sglang.apply_token_bitmask)
    bench.run()
