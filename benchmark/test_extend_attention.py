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

"""Benchmark for attention/extend_attention."""

import pytest

import flaggems_sglang
from baseline import extend_attention as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/extend_attention.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    B, n_ctx, H_Q, H_KV, D = shape
    yield BenchmarkCasePlan(
        shape={
            "q_extend": ("sum(b_seq_len_extend)", H_Q, D),
            "k_extend": ("sum(b_seq_len_extend)", H_KV, D),
            "v_extend": ("sum(b_seq_len_extend)", H_KV, D),
            "k_buffer": ("sum(b_seq_len_prefix + b_seq_len_extend)", H_KV, D),
            "v_buffer": ("sum(b_seq_len_prefix + b_seq_len_extend)", H_KV, D),
            "qo_indptr": (B + 1,),
            "kv_indptr": (B + 1,),
            "kv_indices": ("sum(b_seq_len_prefix)",),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "B": B,
            "n_ctx": n_ctx,
            "H_Q": H_Q,
            "H_KV": H_KV,
            "D": D,
        },
        baseline_key=workload.key_shape(
            {"B": B, "n_ctx": n_ctx, "H_Q": H_Q, "H_KV": H_KV, "D": D}
        ),
        builder_args=(shape,),
    )


@pytest.mark.extend_attention
def test_perf_extend_attention():
    bench = OpBenchmark(
        op_name="extend_attention",
        torch_op=get_reference("extend_attention"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=(),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="B, n_ctx, H_Q, H_KV, D",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
