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

"""Benchmark for speculative/draft_topk1."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

NUM_DRAFT_COLS = 4

# Shapes match kernel-comp-baseline/problems/speculative/draft_topk1.
SHAPES = [(bs, 151936, True, 0) for bs in (1, 8, 64, 512)]
MORE_SHAPES = [
    (1, 32000, False, 0),
    (37, 12000, True, 2),
    (129, 8192 + 500, False, 0),
    (8, 151936, True, 0),
]


def _input_fn(shape, cur_dtype, device):
    bs, vocab_size, with_draft_tokens, draft_token_column = shape
    g = torch.Generator(device=device).manual_seed(0)
    # The logits stay fp32; the op only argmaxes over them.
    next_token_logits = torch.randn(
        bs, vocab_size, generator=g, device=device, dtype=torch.float32
    )
    positions = torch.randint(
        0, 4096, (bs,), generator=g, device=device, dtype=torch.int64
    )

    draft_tokens = None
    if with_draft_tokens:
        draft_tokens = torch.randint(
            0,
            vocab_size,
            (bs, NUM_DRAFT_COLS),
            generator=g,
            device=device,
            dtype=torch.int64,
        )

    yield next_token_logits, positions, draft_tokens, draft_token_column


@pytest.mark.draft_topk1
def test_perf_draft_topk1():
    # This op has a reference but no Triton implementation yet.
    gems_op = flaggems_sglang.get_op("draft_topk1")
    if gems_op is None:
        pytest.skip("speculative/draft_topk1 not implemented yet")
    bench = OpBenchmark(
        op_name="draft_topk1",
        torch_op=get_reference("draft_topk1"),
        input_fn=_input_fn,
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, vocab_size, with_draft_tokens, draft_token_column",
    )
    bench.set_gems(gems_op)
    bench.run()
