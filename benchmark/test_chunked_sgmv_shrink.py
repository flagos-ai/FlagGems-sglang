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

"""Benchmark for lora/chunked_sgmv_shrink."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference._lora_batch_utils import make_batch_info

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/lora/chunked_sgmv_shrink.
# Shape entries are (seg_len, num_segs, num_lora, r, k, num_slices): every
# segment has the same length, which covers both bench cases.
SHAPES = [
    (64, 8, 4, 32, 4096, 1),
    (256, 4, 2, 64, 4096, 1),
]
# BLOCK_M == max(seg_lens) is used directly as a Triton arange size, so the
# largest segment must stay a power of 2.
MORE_SHAPES = [
    (8, 1, 1, 16, 64, 1),
    (16, 4, 2, 16, 128, 1),
    (16, 2, 2, 32, 256, 3),
]


def _input_fn(shape, cur_dtype, device):
    seg_len, num_segs, num_lora, r, k, num_slices = shape
    g = torch.Generator(device=device).manual_seed(0)
    seg_lens = [seg_len] * num_segs
    s = sum(seg_lens)
    x = torch.randn(s, k, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    weights = torch.randn(
        num_lora,
        num_slices * r,
        k,
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
    yield x, weights, {
        "batch_info": batch_info,
        "num_slices": num_slices,
    }


@pytest.mark.chunked_sgmv_shrink
def test_perf_chunked_sgmv_shrink():
    bench = OpBenchmark(
        op_name="chunked_sgmv_shrink",
        torch_op=get_reference("chunked_sgmv_shrink"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seg_len, num_segs, num_lora, r, k, num_slices",
    )
    bench.set_gems(flaggems_sglang.chunked_sgmv_shrink)
    bench.run()
