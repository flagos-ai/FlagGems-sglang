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

"""Benchmark for attention/pad_sequence_with_mask."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_MAX_LEN = 64
# (batch_size, hidden) — max_len fixed at 64, which sets the kernel's tile.
SHAPES = [(bs, 4096) for bs in (1, 8, 64, 256)]
MORE_SHAPES = [(1, 512), (9, 1024), (64, 2048)]


def _input_fn(shape, cur_dtype, device):
    bs, hidden = shape
    g = torch.Generator(device=device).manual_seed(0)
    lengths = torch.randint(
        1,
        _MAX_LEN + 1,
        (bs,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    offsets = torch.zeros(bs, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(lengths, dim=0)[:-1]
    total = int(lengths.sum())
    emb = torch.randn(total, hidden, device=device, generator=g).to(cur_dtype)
    yield dict(
        input_emb=emb,
        offsets=offsets,
        lengths=lengths,
        max_len=_MAX_LEN,
    )


@pytest.mark.pad_sequence_with_mask
def test_perf_pad_sequence_with_mask():
    gems_op = flaggems_sglang.get_op("pad_sequence_with_mask")
    if gems_op is None:
        pytest.skip("attention/pad_sequence_with_mask not implemented yet")
    bench = OpBenchmark(
        op_name="pad_sequence_with_mask",
        torch_op=get_reference("pad_sequence_with_mask"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch_size, hidden",
    )
    bench.set_gems(gems_op)
    bench.run()
