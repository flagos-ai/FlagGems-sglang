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
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference._lora_batch_utils import make_batch_info

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/lora/qkv_lora_b. The first entry
# is the per-segment token count.
SHAPES = [
    ((64,) * 8, 4, 32, 4096, 1024),
    ((256,) * 4, 2, 64, 4096, 1024),
]
MORE_SHAPES = [
    ((5,), 1, 16, 64, 32),
    ((3, 7, 0, 12), 2, 16, 128, 64),
    ((9, 4), 2, 32, 256, 128),
]


def _input_fn(shape, cur_dtype, device):
    seg_lens, num_lora, r, dq, dkv = shape
    g = torch.Generator(device=device).manual_seed(0)
    s = sum(seg_lens)
    output_dim = dq + 2 * dkv
    x = torch.randn(
        s, 3 * r, generator=g, device=device, dtype=torch.float32
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
    output_offset = torch.tensor(
        [0, dq, dq + dkv, output_dim], dtype=torch.int32, device=device
    )
    batch_info = make_batch_info(
        list(seg_lens),
        [i % num_lora for i in range(len(seg_lens))],
        lora_ranks=[r] * num_lora,
        scalings=[0.5 + 0.25 * i for i in range(num_lora)],
    )
    # ``batch_info`` is a dataclass, which unpack_to_args_kwargs would drop
    # from the positional args, so route it (and the tail) through kwargs.
    yield x, weights, dict(
        batch_info=batch_info,
        output_offset=output_offset,
        max_qkv_out_dim=max(dq, dkv),
        base_output=base_output,
    )


@pytest.mark.qkv_lora_b
def test_perf_qkv_lora_b():
    bench = OpBenchmark(
        op_name="qkv_lora_b",
        torch_op=get_reference("qkv_lora_b"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seg_lens, num_lora, r, dq, dkv",
    )
    bench.set_gems(flaggems_sglang.qkv_lora_b)
    bench.run()
