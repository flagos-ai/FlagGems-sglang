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

"""Benchmark for speculative/fused_norm_rope_stacked."""

import pytest

import flaggems_sglang
from baseline import fused_norm_rope_stacked as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# kernel-comp-baseline/problems/speculative/fused_norm_rope_stacked.


def _build_inputs(plan, cur_dtype, device):
    t, n_layers, num_kv_heads, head_dim, rotary_dim = plan.builder_args[0]
    return workload.build_inputs(
        {
            "t": t,
            "n_layers": n_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
        },
        cur_dtype,
        device,
    )[0]


def _case_fn(shape, cur_dtype):
    t, n_layers, num_kv_heads, head_dim, rotary_dim = shape
    yield BenchmarkCasePlan(
        shape={
            "kv": (t, n_layers, 2 * num_kv_heads * head_dim),
            "k_norm_weight": (n_layers, head_dim),
            "eps": (n_layers,),
            "cos_sin_cache": (workload.MAX_POS, rotary_dim),
            "positions": (t,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "_EPS": workload.EPS,
            "_MAX_POS": workload.MAX_POS,
            "t": t,
            "n_layers": n_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_norm_rope_stacked
def test_perf_fused_norm_rope_stacked():
    bench = OpBenchmark(
        op_name="fused_norm_rope_stacked",
        torch_op=get_reference("fused_norm_rope_stacked"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="t, n_layers, num_kv_heads, head_dim, rotary_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
