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

"""Benchmark for activation_norm/softcap_out."""

import pytest

import flaggems_sglang
from baseline import softcap_out as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    m, n = plan.builder_args[0]
    args, _ = workload.build_inputs({"m": m, "n": n}, cur_dtype, device)
    return args


def _case_fn(shape, cur_dtype):
    m, n = shape
    yield BenchmarkCasePlan(
        shape={"x": (m, n)},
        params={
            "configuration": shape,
            "seed": 0,
            "SOFTCAP_CONST": workload.SOFTCAP_CONST,
            "m": m,
            "n": n,
        },
        builder_args=(shape,),
    )


@pytest.mark.softcap_out
def test_perf_softcap_out():
    bench = OpBenchmark(
        op_name="softcap_out",
        torch_op=get_reference("softcap_out"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="m, n",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
