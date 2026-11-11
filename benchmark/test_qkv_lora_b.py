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

"""Benchmark for lora/qkv_lora_b."""

import pytest

import flaggems_sglang
from baseline import qkv_lora_b as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/lora/qkv_lora_b. The first entry
# is the per-segment token count.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    seg_lens, num_lora, r, dq, dkv = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (sum(seg_lens), 3 * r),
            "weights": (num_lora, dq + 2 * dkv, r),
            "base_output": (sum(seg_lens), dq + 2 * dkv),
            "output_offset": (4,),
            "batch_info.seg_indptr": (len(seg_lens) + 1,),
            "batch_info.seg_lens": (len(seg_lens),),
            "batch_info.weight_indices": (len(seg_lens),),
            "batch_info.lora_ranks": (num_lora,),
            "batch_info.scalings": (num_lora,),
            "batch_info.permutation": (
                (sum(seg_lens),) if "none" != "none" else None
            ),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "seg_lens": seg_lens,
            "num_lora": num_lora,
            "r": r,
            "dq": dq,
            "dkv": dkv,
        },
        builder_args=(shape,),
        baseline_key=workload.key_shape(workload.binding_for_case(shape)),
    )


@pytest.mark.qkv_lora_b
def test_perf_qkv_lora_b():
    bench = OpBenchmark(
        op_name="qkv_lora_b",
        torch_op=get_reference("qkv_lora_b"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=("base_output",),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="seg_lens, num_lora, r, dq, dkv",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
