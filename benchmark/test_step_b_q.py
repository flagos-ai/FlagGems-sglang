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

"""Benchmark for lora/step_b_q."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference._lora_batch_utils import make_batch_info

from .op_benchmark import OpBenchmark

SHAPES = [((64,) * 8, 4, 128, 32, 512)]
MORE_SHAPES = [((256,) * 4, 2, 128, 16, 512)]


def _input_fn(shape, cur_dtype, device):
    seg_lens, num_lora, h, rank, kv_lora_rank = shape
    g = torch.Generator(device=device).manual_seed(0)
    s = sum(seg_lens)
    q_lora_a = torch.randn(
        s, h, rank, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    a_buf = torch.randn(
        num_lora,
        rank,
        kv_lora_rank,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    base_output = torch.randn(
        s, h, kv_lora_rank, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    batch_info = make_batch_info(
        list(seg_lens),
        [i % num_lora for i in range(len(seg_lens))],
        lora_ranks=[rank] * num_lora,
        scalings=[0.5 + 0.25 * i for i in range(num_lora)],
    )
    # batch_info is a dataclass, which unpack_to_args_kwargs would drop from
    # the positional args, so route it (and the tail) through kwargs.
    yield q_lora_a, a_buf, dict(batch_info=batch_info, base_output=base_output)


@pytest.mark.step_b_q
def test_perf_step_b_q():
    bench = OpBenchmark(
        op_name="step_b_q",
        torch_op=get_reference("step_b_q"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seg_lens, num_lora, heads, rank, kv_lora_rank",
    )
    bench.set_gems(flaggems_sglang.step_b_q)
    bench.run()
