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

"""Benchmark for rope/ernie45_rope_fused."""

import pytest

import flaggems_sglang
from baseline import ernie45_rope_fused as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/rope/ernie45_rope_fused.
# mrope_section is [section_h, section_w, section_t] with
# section_h == section_w and the three summing to rotary_dim // 2
# (Ernie4.5 layout), so it is derived from rotary_dim rather than listed.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, n_qh, n_kh, head_size, rotary_dim = shape
    yield BenchmarkCasePlan(
        shape={
            "q": (num_tokens, n_qh * head_size),
            "k": (num_tokens, n_kh * head_size),
            "cos_sin_cache": (workload._MAX_POS, rotary_dim),
            "positions": (3, num_tokens),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "_MAX_POS": workload._MAX_POS,
            "num_tokens": num_tokens,
            "n_qh": n_qh,
            "n_kh": n_kh,
            "head_size": head_size,
            "rotary_dim": rotary_dim,
        },
        baseline_key=workload.key_shape(
            {
                "num_tokens": num_tokens,
                "n_qh": n_qh,
                "n_kh": n_kh,
                "head_size": head_size,
                "rotary_dim": rotary_dim,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.ernie45_rope_fused
def test_perf_ernie45_rope_fused():
    bench = OpBenchmark(
        op_name="ernie45_rope_fused",
        torch_op=get_reference("ernie45_rope_fused"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, n_qh, n_kh, head_size, rotary_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
