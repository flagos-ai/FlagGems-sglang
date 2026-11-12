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

"""Benchmark for lora/chunked_sgmv_expand."""

import pytest

import flaggems_sglang
from baseline import chunked_sgmv_expand as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/lora/chunked_sgmv_expand.
# Entries are (seg_len, num_segs, num_lora, r, *slice_sizes): every segment
# has the same length, which covers both bench cases.
# BLOCK_M == max(seg_lens) is used directly as a Triton arange size, so the
# largest segment must stay a power of 2.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    seg_len, num_segs, num_lora, r = shape[:4]
    slice_sizes = list(shape[4:])
    yield BenchmarkCasePlan(
        shape={
            "x": (seg_len * num_segs, len(slice_sizes) * r),
            "weights": (num_lora, sum(slice_sizes), r),
            "base_output": (seg_len * num_segs, sum(slice_sizes)),
            "slice_offsets": (len(slice_sizes) + 1,),
            "batch_info.seg_indptr": (num_segs + 1,),
            "batch_info.seg_lens": (num_segs,),
            "batch_info.weight_indices": (num_segs,),
            "batch_info.lora_ranks": (num_lora,),
            "batch_info.scalings": (num_lora,),
            "batch_info.permutation": (
                (seg_len * num_segs,) if "identity" != "none" else None
            ),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "seg_len": seg_len,
            "num_segs": num_segs,
            "num_lora": num_lora,
            "r": r,
            "slice_sizes": slice_sizes,
        },
        builder_args=(shape,),
        baseline_key=workload.key_shape(workload.binding_for_case(shape)),
    )


@pytest.mark.chunked_sgmv_expand
def test_perf_chunked_sgmv_expand():
    bench = OpBenchmark(
        op_name="chunked_sgmv_expand",
        torch_op=get_reference("chunked_sgmv_expand"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=("base_output",),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="seg_len, num_segs, num_lora, r, *slice_sizes",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
