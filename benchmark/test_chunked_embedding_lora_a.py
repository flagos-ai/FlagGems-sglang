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

"""Benchmark for lora/chunked_embedding_lora_a."""

import pytest

import flaggems_sglang
from baseline import chunked_embedding_lora_a as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/lora/chunked_embedding_lora_a.
# Entries are (seg_len, num_segs, num_lora, r, vocab_size): every segment has
# the same length, which covers both bench cases.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    seg_len, num_segs, num_lora, r, vocab_size = shape
    yield BenchmarkCasePlan(
        shape={
            "input_ids": (seg_len * num_segs,),
            "weights": (num_lora, r, vocab_size),
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
            "vocab_size": vocab_size,
        },
        builder_args=(shape,),
        baseline_key=workload.key_shape(workload.binding_for_case(shape)),
    )


@pytest.mark.chunked_embedding_lora_a
def test_perf_chunked_embedding_lora_a():
    bench = OpBenchmark(
        op_name="chunked_embedding_lora_a",
        torch_op=get_reference("chunked_embedding_lora_a"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="seg_len, num_segs, num_lora, r, vocab_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
