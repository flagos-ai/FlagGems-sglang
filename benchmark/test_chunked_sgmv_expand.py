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
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference._lora_batch_utils import make_batch_info

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/lora/chunked_sgmv_expand.
# Entries are (seg_len, num_segs, num_lora, r, *slice_sizes): every segment
# has the same length, which covers both bench cases.
SHAPES = [
    (64, 8, 4, 32, 4096, 4096),
    (256, 4, 2, 64, 4096, 1024, 1024),
]
# BLOCK_M == max(seg_lens) is used directly as a Triton arange size, so the
# largest segment must stay a power of 2.
MORE_SHAPES = [
    (8, 1, 1, 16, 64),
    (16, 4, 2, 16, 128, 64),
    (16, 2, 2, 32, 256, 128, 128),
]


def _input_fn(shape, cur_dtype, device):
    seg_len, num_segs, num_lora, r = shape[:4]
    slice_sizes = list(shape[4:])
    g = torch.Generator(device=device).manual_seed(0)
    seg_lens = [seg_len] * num_segs
    s = sum(seg_lens)
    n_slices = len(slice_sizes)
    output_dim = sum(slice_sizes)
    max_slice_size = max(slice_sizes)

    offsets = [0]
    for sz in slice_sizes:
        offsets.append(offsets[-1] + sz)
    slice_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)

    x = torch.randn(
        s,
        n_slices * r,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    weights = torch.randn(
        num_lora,
        output_dim,
        r,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    base_output = torch.randn(
        s, output_dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    weight_indices = [i % num_lora for i in range(num_segs)]
    scalings = [0.5 + 0.25 * i for i in range(num_lora)]
    batch_info = make_batch_info(
        seg_lens,
        weight_indices,
        lora_ranks=[r] * num_lora,
        scalings=scalings,
        permutation="identity",
    )
    # batch_info is a dataclass, which unpack_to_args_kwargs drops from the
    # positional args -- it and every later parameter ride in a dict.
    yield x, weights, {
        "batch_info": batch_info,
        "slice_offsets": slice_offsets,
        "max_slice_size": max_slice_size,
        "base_output": base_output,
    }


@pytest.mark.chunked_sgmv_expand
def test_perf_chunked_sgmv_expand():
    bench = OpBenchmark(
        op_name="chunked_sgmv_expand",
        torch_op=get_reference("chunked_sgmv_expand"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seg_len, num_segs, num_lora, r, *slice_sizes",
    )
    bench.set_gems(flaggems_sglang.chunked_sgmv_expand)
    bench.run()
