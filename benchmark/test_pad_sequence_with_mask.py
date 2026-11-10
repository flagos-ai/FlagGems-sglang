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

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (batch, max_len, hidden). Every row is fully valid, so the benchmark
# measures the gather rather than a mostly-empty pad band; the pad-band cost
# is the same kernel either way.
SHAPES = [
    (8, 512, 4096),
    (56, 128, 3584),
    (1, 2048, 1024),
    (4, 64, 3584),
]

MORE_SHAPES = [
    (1, 1, 128),
    (2, 33, 100),
    (32, 16, 512),
]


def _input_fn(shape, cur_dtype, device):
    batch, max_len, hidden = shape
    g = torch.Generator(device=device).manual_seed(0)

    lengths = torch.full((batch,), max_len, dtype=torch.int32, device=device)
    total = int(lengths.sum().item())
    offsets = torch.tensor(
        [i * max_len for i in range(batch)],
        dtype=torch.int32,
        device=device,
    )

    input_emb = torch.randn(
        total, hidden, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)

    yield input_emb, offsets, lengths, max_len


def _case_fn(shape, cur_dtype):
    batch, max_len, hidden = shape
    yield BenchmarkCasePlan(
        shape={
            "input_emb": (batch * max_len, hidden),
            "offsets": (batch,),
            "lengths": (batch,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "batch": batch,
            "max_len": max_len,
            "hidden": hidden,
        },
        builder_args=(shape,),
    )


@pytest.mark.pad_sequence_with_mask
def test_perf_pad_sequence_with_mask():
    bench = OpBenchmark(
        op_name="pad_sequence_with_mask",
        torch_op=get_reference("pad_sequence_with_mask"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.float32, torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, max_len, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
