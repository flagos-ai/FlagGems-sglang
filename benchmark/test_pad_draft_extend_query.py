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

import flaggems_sglang
from baseline import pad_draft_extend_query as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/attention/pad_draft_extend_query.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, max_seq_len, num_heads, head_dim = shape
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
        baseline_key=workload.key_shape(
            {
                "bs": bs,
                "max_seq_len": max_seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.pad_draft_extend_query
def test_perf_pad_draft_extend_query():
    bench = OpBenchmark(
        op_name="pad_draft_extend_query",
        torch_op=get_reference("pad_draft_extend_query"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, max_seq_len, num_heads, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
