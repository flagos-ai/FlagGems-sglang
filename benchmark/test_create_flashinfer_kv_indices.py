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

"""Benchmark for attention/create_flashinfer_kv_indices."""

import pytest

import flaggems_sglang
from baseline import create_flashinfer_kv_indices as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, page_size, max_len = shape
    yield BenchmarkCasePlan(
        shape={
            "kv_indptr": (bs + 1,),
            "kv_page_indices": ("kv_indptr[-1]",),
            "kv_last_page_len": (bs,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "bs": bs,
            "page_size": page_size,
            "max_len": max_len,
        },
        baseline_key=workload.key_shape(
            {
                "bs": bs,
                "page_size": page_size,
                "max_len": max_len,
                **{"max_batch": 256, "max_context": 8192},
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.create_flashinfer_kv_indices
def test_perf_create_flashinfer_kv_indices():
    bench = OpBenchmark(
        op_name="create_flashinfer_kv_indices",
        torch_op=get_reference("create_flashinfer_kv_indices"),
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
