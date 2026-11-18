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

"""Benchmark for attention/linear_decode_forward."""

import pytest

import flaggems_sglang
from baseline import linear_decode_forward as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/linear_decode_forward.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    b, h, d = shape
    yield BenchmarkCasePlan(
        shape={
            "q": (b, h, 1, d),
            "k": (b, h, 1, d),
            "v": (b, h, 1, d),
            "state": (b + 2, h, d, d),
            "slope": (h,),
            "slot_idx": (b,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_BLOCK_SIZE": workload._BLOCK_SIZE,
            "b": b,
            "h": h,
            "d": d,
        },
        builder_args=(shape,),
    )


@pytest.mark.linear_decode_forward
def test_perf_linear_decode_forward():
    bench = OpBenchmark(
        op_name="linear_decode_forward",
        torch_op=get_reference("linear_decode_forward"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="b, h, d",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
