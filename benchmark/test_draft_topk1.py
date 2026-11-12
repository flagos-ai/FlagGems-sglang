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

import flaggems_sglang
from baseline import draft_topk1 as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark, is_case_listing

# Shapes match kernel-comp-baseline/problems/speculative/draft_topk1.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, vocab_size, with_draft_tokens, draft_token_column = shape
    yield BenchmarkCasePlan(
        shape={
            "next_token_logits": (bs, vocab_size),
            "positions": (bs,),
            "draft_tokens": (
                (bs, workload.NUM_DRAFT_COLS) if with_draft_tokens else None
            ),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "NUM_DRAFT_COLS": workload.NUM_DRAFT_COLS,
            "bs": bs,
            "vocab_size": vocab_size,
            "with_draft_tokens": with_draft_tokens,
            "draft_token_column": draft_token_column,
        },
        baseline_key=workload.key_shape(
            {
                "bs": bs,
                "vocab_size": vocab_size,
                "with_draft_tokens": with_draft_tokens,
                "draft_token_column": draft_token_column,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.draft_topk1
def test_perf_draft_topk1():
    # This op has a reference but no Triton implementation yet.
    gems_op = None
    if not is_case_listing():
        gems_op = flaggems_sglang.get_op("draft_topk1")
        if gems_op is None:
            pytest.skip("speculative/draft_topk1 not implemented yet")
    bench = OpBenchmark(
        op_name="draft_topk1",
        torch_op=get_reference("draft_topk1"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=(1, 2),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, vocab_size, with_draft_tokens, draft_token_column",
    )
    bench.set_gems(gems_op)
    bench.run()
