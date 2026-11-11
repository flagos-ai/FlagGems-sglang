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

import flaggems_sglang
from baseline import unpad_draft_extend_output as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/
# unpad_draft_extend_output.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, token_per_batch, num_heads, head_dim = shape
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
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, token_per_batch, num_heads, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
