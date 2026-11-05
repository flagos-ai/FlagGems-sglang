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
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference._lora_batch_utils import make_batch_info

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/lora/chunked_embedding_lora_a.
# Entries are (seg_len, num_segs, num_lora, r, vocab_size): every segment has
# the same length, which covers both bench cases.
SHAPES = [
    (512, 8, 4, 32, 32000),
    (2048, 4, 2, 64, 128256),
]
MORE_SHAPES = [
    (5, 1, 1, 16, 128),
    (12, 4, 2, 32, 512),
    (9, 2, 2, 16, 256),
]


def _input_fn(shape, cur_dtype, device):
    seg_len, num_segs, num_lora, r, vocab_size = shape
    g = torch.Generator(device=device).manual_seed(0)
    seg_lens = [seg_len] * num_segs
    s = sum(seg_lens)
    input_ids = torch.randint(
        0, vocab_size, (s,), generator=g, device=device, dtype=torch.int64
    )
    weights = torch.randn(
        num_lora,
        r,
        vocab_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    weight_indices = [i % num_lora for i in range(num_segs)]
    batch_info = make_batch_info(
        seg_lens,
        weight_indices,
        lora_ranks=[r] * num_lora,
        permutation="identity",
    )
    # batch_info is a dataclass, which unpack_to_args_kwargs drops from the
    # positional args -- it and every later parameter ride in a dict.
    yield input_ids, weights, {
        "batch_info": batch_info,
        "vocab_size": vocab_size,
    }


def _case_fn(shape, cur_dtype):
    (seg_len, num_segs, num_lora, r, vocab_size) = shape
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
    )


@pytest.mark.chunked_embedding_lora_a
def test_perf_chunked_embedding_lora_a():
    bench = OpBenchmark(
        op_name="chunked_embedding_lora_a",
        torch_op=get_reference("chunked_embedding_lora_a"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seg_len, num_segs, num_lora, r, vocab_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
