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

"""Benchmark for attention/build_trtllm_mha_page_table."""

import pytest

import flaggems_sglang
from baseline import build_trtllm_mha_page_table as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, block_size, max_len = shape
    yield BenchmarkCasePlan(
        shape={
            "seq_lens": (bs,),
            "block_tables": (bs, (max_len + block_size - 1) // block_size),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "bs": bs,
            "block_size": block_size,
            "max_len": max_len,
        },
        baseline_key=workload.key_shape(
            {
                "bs": bs,
                "page_size": block_size,
                "max_len": max_len,
                **{
                    "max_batch": workload.MAX_BATCH,
                    "max_context": workload.MAX_CONTEXT,
                },
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.build_trtllm_mha_page_table
def test_perf_build_trtllm_mha_page_table():
    bench = OpBenchmark(
        op_name="build_trtllm_mha_page_table",
        torch_op=get_reference("build_trtllm_mha_page_table"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, page_size, max_len",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
