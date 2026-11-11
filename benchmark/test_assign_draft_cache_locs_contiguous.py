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

"""Benchmark for speculative/assign_draft_cache_locs_contiguous."""

import pytest

import flaggems_sglang
from baseline import assign_draft_cache_locs_contiguous as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/speculative/assign_draft_cache_locs_contiguous.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, topk, steps = shape
    yield BenchmarkCasePlan(
        shape={
            "req_pool_indices": (min(bs, workload._MAX_BATCH),),
            "req_to_token": (workload._MAX_BATCH, workload._POOL_LEN),
            "seq_lens": (bs,),
            "out_cache_loc": (bs * topk * steps,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_MAX_BATCH": workload._MAX_BATCH,
            "_POOL_LEN": workload._POOL_LEN,
            "bs": bs,
            "topk": topk,
            "steps": steps,
        },
        baseline_key=workload.key_shape(
            {
                "bs": bs,
                "topk": topk,
                "steps": steps,
                **{
                    "max_batch": workload._MAX_BATCH,
                    "pool_len": workload._POOL_LEN,
                },
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.assign_draft_cache_locs_contiguous
def test_perf_assign_draft_cache_locs_contiguous():
    bench = OpBenchmark(
        op_name="assign_draft_cache_locs_contiguous",
        torch_op=get_reference("assign_draft_cache_locs_contiguous"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, topk, steps",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
