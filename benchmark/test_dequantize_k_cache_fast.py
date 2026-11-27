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

"""Benchmark for dsa/dequantize_k_cache_fast."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark
from flaggems_sglang.utils.rng_utils import create_generator

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0
_DIM_NOPE, _DIM_ROPE = 512, 64
# (num_tokens,) — group_size fixed at 128.
SHAPES = [(t,) for t in (1, 128, 2048, 8192)]
MORE_SHAPES = [(37,), (256,), (4096,)]


def _build_quant_k_cache(num_tokens, device, group_size=128, seed=0):
    """Pack [512 nope fp8 | 4 fp32 scales | 64 bf16 rope] into one fp8 row."""
    g = create_generator(device, seed=seed)
    k_nope = (
        2 * torch.randn(num_tokens, _DIM_NOPE, generator=g, device=device)
    ).to(torch.bfloat16)
    k_rope = torch.randn(num_tokens, _DIM_ROPE, generator=g, device=device).to(
        torch.bfloat16
    )
    num_tiles = _DIM_NOPE // group_size

    buf = torch.empty(
        num_tokens,
        _DIM_NOPE + num_tiles * 4 + _DIM_ROPE * 2,
        dtype=_FP8_DTYPE,
        device=device,
    )
    nope_q = buf[:, :_DIM_NOPE]
    nope_s = buf[:, _DIM_NOPE : _DIM_NOPE + num_tiles * 4].view(torch.float32)
    rope = buf[:, _DIM_NOPE + num_tiles * 4 :].view(torch.bfloat16)
    rope[:] = k_rope

    for t in range(num_tiles):
        lo = t * group_size
        hi = lo + group_size
        tile = k_nope[:, lo:hi].float()
        scale = (tile.abs().amax(dim=-1) / _FP8_MAX).clamp(min=1e-12)
        nope_s[:, t] = scale
        nope_q[:, lo:hi] = (tile / scale.unsqueeze(-1)).to(_FP8_DTYPE)

    return buf


def _input_fn(shape, cur_dtype, device):
    (num_tokens,) = shape
    yield dict(
        quant_k_cache=_build_quant_k_cache(num_tokens, device),
        group_size=128,
    )


@pytest.mark.dequantize_k_cache_fast
def test_perf_dequantize_k_cache_fast():
    gems_op = flaggems_sglang.get_op("dequantize_k_cache_fast")
    if gems_op is None:
        pytest.skip("dsa/dequantize_k_cache_fast not implemented yet")
    bench = OpBenchmark(
        op_name="dequantize_k_cache_fast",
        torch_op=get_reference("dequantize_k_cache_fast"),
        input_fn=_input_fn,
        dtypes=[torch.float8_e4m3fn],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens",
    )
    bench.set_gems(gems_op)
    bench.run()
