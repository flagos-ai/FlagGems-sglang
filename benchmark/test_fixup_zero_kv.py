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

"""Benchmark for attention/fixup_zero_kv."""

import pytest

import flaggems_sglang
from baseline import fixup_zero_kv as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/fixup_zero_kv.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, num_heads, v_head_dim, max_q = shape
    yield BenchmarkCasePlan(
        shape={
            "out": ("sum(q_lens)", num_heads, v_head_dim),
            "lse": ("sum(q_lens)", num_heads),
            "kv_lens": (bs,),
            "cum": (bs + 1,),
        },
        params={
            "configuration": shape,
            "seed": 11,
            "_ZERO_FRAC": workload._ZERO_FRAC,
            "bs": bs,
            "num_heads": num_heads,
            "v_head_dim": v_head_dim,
            "max_q": max_q,
        },
        baseline_key=workload.key_shape(
            {
                "bs": bs,
                "num_heads": num_heads,
                "v_head_dim": v_head_dim,
                "max_q": max_q,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.fixup_zero_kv
def test_perf_fixup_zero_kv():
    bench = OpBenchmark(
        op_name="fixup_zero_kv",
        torch_op=get_reference("fixup_zero_kv"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, num_heads, v_head_dim, max_q",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
