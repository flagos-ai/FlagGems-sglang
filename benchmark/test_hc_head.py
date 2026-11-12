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

"""Benchmark for activation_norm/hc_head."""

import pytest

import flaggems_sglang
from baseline import hc_head as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/activation_norm/hc_head.

_NORM_EPS = 1e-6
_HC_EPS = 1e-3


def _build_inputs(plan, cur_dtype, device):
    t, hc_mult, hidden = plan.builder_args[0]
    args, kwargs = workload.build_inputs(
        {"t": t, "hc_mult": hc_mult, "hidden": hidden}, cur_dtype, device
    )
    return (*args, kwargs) if kwargs else args


def _case_fn(shape, cur_dtype):
    t, hc_mult, hidden_size = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (t, hc_mult, hidden_size),
            "hc_fn": (hc_mult, hc_mult * hidden_size),
            "hc_scale": (1,),
            "hc_base": (hc_mult,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "_HC_EPS": _HC_EPS,
            "_NORM_EPS": _NORM_EPS,
            "t": t,
            "hc_mult": hc_mult,
            "hidden_size": hidden_size,
        },
        builder_args=(shape,),
    )


@pytest.mark.hc_head
def test_perf_hc_head():
    bench = OpBenchmark(
        op_name="hc_head",
        torch_op=get_reference("hc_head"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="t, hc_mult, hidden_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
