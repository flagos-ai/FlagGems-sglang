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

"""Benchmark for fla/recompute_w_u."""

import pytest

import flaggems_sglang
from baseline import recompute_w_u as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    b, t, h, hg, k_dim, v_dim, bt = plan.builder_args[0]
    return workload.build_inputs(
        {
            "b": b,
            "t": t,
            "h": h,
            "hg": hg,
            "k_dim": k_dim,
            "v_dim": v_dim,
            "bt": bt,
        },
        cur_dtype,
        device,
    )[0]


def _case_fn(shape, cur_dtype):
    b, t, h, hg, k_dim, v_dim, bt = shape
    yield BenchmarkCasePlan(
        shape={
            "k": (b, t, hg, k_dim),
            "v": (b, t, h, v_dim),
            "beta": (b, t, h),
            "g": (b, t, h),
            "A": (b, t, h, bt),
            "cu_seqlens": None,
        },
        params={
            "configuration": shape,
            "seed": 9,
            "b": b,
            "t": t,
            "h": h,
            "hg": hg,
            "k_dim": k_dim,
            "v_dim": v_dim,
            "bt": bt,
        },
        builder_args=(shape,),
    )


@pytest.mark.recompute_w_u
def test_perf_recompute_w_u():
    bench = OpBenchmark(
        op_name="recompute_w_u",
        torch_op=get_reference("recompute_w_u"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="b, t, h, hg, k_dim, v_dim, bt",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
