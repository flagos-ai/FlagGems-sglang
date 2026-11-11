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

"""Benchmark for kvcache/store_cache_4d."""

import pytest

import flaggems_sglang
from baseline import store_cache_4d as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/kvcache/store_cache_4d.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    n, head_num, head_dim = shape
    yield BenchmarkCasePlan(
        shape={
            "k_view": (
                workload._NUM_PAGES,
                workload._PAGE_SIZE,
                head_num,
                head_dim,
            ),
            "v_view": (
                workload._NUM_PAGES,
                workload._PAGE_SIZE,
                head_num,
                head_dim,
            ),
            "cache_k": (n, head_num, head_dim),
            "cache_v": (n, head_num, head_dim),
            "loc": (n,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_NUM_PAGES": workload._NUM_PAGES,
            "_PAGE_SIZE": workload._PAGE_SIZE,
            "n": n,
            "head_num": head_num,
            "head_dim": head_dim,
        },
        builder_args=(shape,),
        baseline_key=workload.key_shape(workload.binding_for_case(shape)),
    )


@pytest.mark.store_cache_4d
def test_perf_store_cache_4d():
    bench = OpBenchmark(
        op_name="store_cache_4d",
        torch_op=get_reference("store_cache_4d"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=(0, 1),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="n, head_num, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
