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

"""Benchmark for kvcache/create_flashmla_kv_indices."""

import pytest

import flaggems_sglang
from baseline import create_flashmla_kv_indices as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/kvcache/
# create_flashmla_kv_indices.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, max_len, page_size = shape
    yield BenchmarkCasePlan(
        shape={
            "req_to_token": (workload._MAX_BATCH, workload._MAX_CONTEXT),
            "req_pool_indices": (bs,),
            "lens": (bs,),
            "kv_indices": (bs, (max_len + page_size - 1) // page_size),
        },
        params={
            "configuration": shape,
            "seed": 7,
            "_MAX_BATCH": workload._MAX_BATCH,
            "_MAX_CONTEXT": workload._MAX_CONTEXT,
            "bs": bs,
            "max_len": max_len,
            "page_size": page_size,
        },
        baseline_key=workload.key_shape(
            {
                "bs": bs,
                "max_len": max_len,
                "page_size": page_size,
                **{
                    "max_batch": workload._MAX_BATCH,
                    "max_context": workload._MAX_CONTEXT,
                },
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.create_flashmla_kv_indices
def test_perf_create_flashmla_kv_indices():
    bench = OpBenchmark(
        op_name="create_flashmla_kv_indices",
        torch_op=get_reference("create_flashmla_kv_indices"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, max_len, page_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
