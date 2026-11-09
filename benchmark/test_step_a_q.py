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

"""Benchmark for lora/step_a_q."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference._lora_batch_utils import make_batch_info

from .op_benchmark import OpBenchmark

SHAPES = [((64,) * 8, 4, 128, 128, 128, 32)]
MORE_SHAPES = [((256,) * 4, 2, 128, 64, 128, 16)]


def _input_fn(shape, cur_dtype, device):
    seg_lens, num_lora, h, qk_nope, v_head_dim, rank = shape
    g = torch.Generator(device=device).manual_seed(0)
    s = sum(seg_lens)
    full_k = qk_nope + v_head_dim
    q_nope = torch.randn(
        s, h, qk_nope, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    b_buf = torch.randn(
        num_lora,
        h * full_k,
        rank,
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
    # the positional args, so route it (and the tail) through kwargs.
    yield q_nope, b_buf, dict(batch_info=batch_info, full_K_per_head=full_k)


@pytest.mark.step_a_q
def test_perf_step_a_q():
    bench = OpBenchmark(
        op_name="step_a_q",
        torch_op=get_reference("step_a_q"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seg_lens, num_lora, heads, qk_nope, v_head_dim, rank",
    )
    bench.set_gems(flaggems_sglang.step_a_q)
    bench.run()
