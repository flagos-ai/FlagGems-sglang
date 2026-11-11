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

"""Benchmark for attention/decode_attention."""

import pytest

import flaggems_sglang
from baseline import decode_attention as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/decode_attention. The
# KV buffer scales with B * seq_len, so the generator builds one shape's
# tensors at a time rather than holding every case in memory.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    B, H_Q, H_KV, D, seq_len = shape
    yield BenchmarkCasePlan(
        shape={
            "q": (B, H_Q, D),
            "k_buffer": (B * seq_len, H_KV, D),
            "v_buffer": (B * seq_len, H_KV, D),
            "kv_indptr": (B + 1,),
            "kv_indices": (B * seq_len,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "B": B,
            "H_Q": H_Q,
            "H_KV": H_KV,
            "D": D,
            "seq_len": seq_len,
        },
        baseline_key=workload.key_shape(
            {"B": B, "H_Q": H_Q, "H_KV": H_KV, "D": D, "seq_len": seq_len}
        ),
        builder_args=(shape,),
    )


@pytest.mark.decode_attention
def test_perf_decode_attention():
    bench = OpBenchmark(
        op_name="decode_attention",
        torch_op=get_reference("decode_attention"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=(),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="B, H_Q, H_KV, D, seq_len",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
