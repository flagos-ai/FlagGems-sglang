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

"""Benchmark for lora/embedding_lora_a."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference._lora_batch_utils import make_batch_info

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/lora/embedding_lora_a. The first
# entry is the per-segment token count.
SHAPES = [
    ((512,) * 8, 4, 32, 32000, 0),
    ((2048,) * 4, 2, 64, 128256, 0),
]
MORE_SHAPES = [
    ((5,), 1, 16, 128, 0),
    ((3, 7, 0, 12), 2, 32, 512, 0),
    ((9, 4), 2, 16, 256, 8),
]


def _input_fn(shape, cur_dtype, device):
    seg_lens, num_lora, r, vocab_size, num_extra = shape
    g = torch.Generator(device=device).manual_seed(0)
    input_ids = torch.randint(
        0,
        vocab_size + num_extra,
        (sum(seg_lens),),
        generator=g,
        device=device,
        dtype=torch.int64,
    )
    weights = torch.randn(
        num_lora,
        r,
        vocab_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    extra_embeddings = None
    if num_extra > 0:
        extra_embeddings = torch.randn(
            num_lora,
            num_extra,
            r,
            generator=g,
            device=device,
            dtype=torch.float32,
        ).to(cur_dtype)
    batch_info = make_batch_info(
        list(seg_lens),
        [i % num_lora for i in range(len(seg_lens))],
        lora_ranks=[r] * num_lora,
    )
    # ``batch_info`` is a dataclass, which unpack_to_args_kwargs would drop
    # from the positional args, so route it (and the tail) through kwargs.
    yield input_ids, weights, dict(
        batch_info=batch_info,
        vocab_size=vocab_size,
        extra_embeddings=extra_embeddings,
    )


def _case_fn(shape, cur_dtype):
    (seg_lens, num_lora, r, vocab_size, num_extra) = shape
    yield BenchmarkCasePlan(
        shape={
            "input_ids": (sum(seg_lens),),
            "weights": (num_lora, r, vocab_size),
            "extra_embeddings": (
                (num_lora, num_extra, r) if num_extra > 0 else None
            ),
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
            "vocab_size": vocab_size,
            "num_extra": num_extra,
        },
        builder_args=(shape,),
    )


@pytest.mark.embedding_lora_a
def test_perf_embedding_lora_a():
    bench = OpBenchmark(
        op_name="embedding_lora_a",
        torch_op=get_reference("embedding_lora_a"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seg_lens, num_lora, r, vocab_size, num_extra",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
