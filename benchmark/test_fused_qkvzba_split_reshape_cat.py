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

"""Benchmark for attention/fused_qkvzba_split_reshape_cat."""

import pytest

import flaggems_sglang
from baseline import fused_qkvzba_split_reshape_cat as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/fused_qkvzba_split_reshape_cat.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    batch, nqk, nv, head_qk, head_v = shape
    yield BenchmarkCasePlan(
        shape={
            "mixed_qkvz": (
                batch,
                nqk * (2 * head_qk + 2 * (nv // nqk) * head_v),
            ),
            "mixed_ba": (batch, 2 * nv),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "batch": batch,
            "nqk": nqk,
            "nv": nv,
            "head_qk": head_qk,
            "head_v": head_v,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_qkvzba_split_reshape_cat
def test_perf_fused_qkvzba_split_reshape_cat():
    bench = OpBenchmark(
        op_name="fused_qkvzba_split_reshape_cat",
        torch_op=get_reference("fused_qkvzba_split_reshape_cat"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="batch, num_heads_qk, num_heads_v, head_qk, head_v",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
