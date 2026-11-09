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

"""Benchmark for lora/step_a_v."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference._lora_batch_utils import make_batch_info

from .op_benchmark import OpBenchmark

SHAPES = [((64,) * 8, 4, 128, 512, 32)]
MORE_SHAPES = [((256,) * 4, 2, 128, 512, 16)]


def _input_fn(shape, cur_dtype, device):
    seg_lens, num_lora, h, kv_lora_rank, rank = shape
    g = torch.Generator(device=device).manual_seed(0)
    s = sum(seg_lens)
    attn_output = torch.randn(
        s, h, kv_lora_rank, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    a_buf = torch.randn(
        num_lora,
        rank,
        kv_lora_rank,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    batch_info = make_batch_info(
        list(seg_lens),
        [i % num_lora for i in range(len(seg_lens))],
        lora_ranks=[rank] * num_lora,
    )
    # batch_info is a dataclass, which unpack_to_args_kwargs would drop from
    # the positional args, so route it through kwargs.
    yield attn_output, a_buf, dict(batch_info=batch_info)


@pytest.mark.step_a_v
def test_perf_step_a_v():
    bench = OpBenchmark(
        op_name="step_a_v",
        torch_op=get_reference("step_a_v"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seg_lens, num_lora, heads, kv_lora_rank, rank",
    )
    bench.set_gems(flaggems_sglang.step_a_v)
    bench.run()
