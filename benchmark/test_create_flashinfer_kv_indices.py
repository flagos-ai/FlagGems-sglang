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

"""Benchmark for attention/create_flashinfer_kv_indices."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(bs, 16, 4096) for bs in (1, 8, 32)]
MORE_SHAPES = [(bs, 16, 4096) for bs in (128,)]


def _input_fn(shape, cur_dtype, device):
    bs, page_size, max_len = shape
    g = torch.Generator(device=device).manual_seed(0)
    num_pages = (max_len + page_size - 1) // page_size
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.randint(
        1, num_pages + 1, (bs,), dtype=torch.int32, device=device, generator=g
    ).cumsum(0)
    kv_page_indices = torch.randint(
        0,
        100000,
        (int(kv_indptr[-1].item()),),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    kv_last_page_len = torch.randint(
        1, page_size + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    yield kv_indptr, kv_page_indices, kv_last_page_len, {
        "page_size": page_size
    }


@pytest.mark.create_flashinfer_kv_indices
def test_perf_create_flashinfer_kv_indices():
    bench = OpBenchmark(
        op_name="create_flashinfer_kv_indices",
        torch_op=get_reference("create_flashinfer_kv_indices"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, page_size, max_len",
    )
    bench.set_gems(flaggems_sglang.create_flashinfer_kv_indices)
    bench.run()
