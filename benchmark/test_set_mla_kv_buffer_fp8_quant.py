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

"""Benchmark for kvcache/set_mla_kv_buffer_fp8_quant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.utils.rng_utils import create_generator

from .op_benchmark import OpBenchmark

_NOPE, _ROPE = 512, 64
# (tokens,) — the slot count scales with tokens; loc values are distinct.
SHAPES = [(t,) for t in (16, 128, 1024, 4096, 16384)]
MORE_SHAPES = [(333,), (2048,), (8192,)]


def _input_fn(shape, cur_dtype, device):
    (tokens,) = shape
    slots = max(tokens * 2, 64)
    g = create_generator(device, seed=0)

    def _t(*dims):
        return torch.randn(
            *dims, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    # Allocated in the source dtype and viewed as fp8 in the kernel, so the
    # last dim is half as wide in elements.
    kv_buffer = _t(slots, (_NOPE + _ROPE) // 2).contiguous()
    loc = torch.randperm(slots, generator=g, device=device)[:tokens].to(
        torch.int64
    )
    yield dict(
        kv_buffer=kv_buffer,
        loc=loc,
        cache_k_nope=_t(tokens, _NOPE).contiguous(),
        cache_k_rope=_t(tokens, _ROPE).contiguous(),
        fp8_dtype=torch.float8_e4m3fn,
    )


@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_perf_set_mla_kv_buffer_fp8_quant():
    gems_op = flaggems_sglang.get_op("set_mla_kv_buffer_fp8_quant")
    if gems_op is None:
        pytest.skip("kvcache/set_mla_kv_buffer_fp8_quant not implemented yet")
    bench = OpBenchmark(
        op_name="set_mla_kv_buffer_fp8_quant",
        torch_op=get_reference("set_mla_kv_buffer_fp8_quant"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens",
    )
    bench.set_gems(gems_op)
    bench.run()
