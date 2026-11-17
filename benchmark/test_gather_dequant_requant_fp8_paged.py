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

"""Benchmark for dsa/gather_dequant_requant_fp8_paged."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_DIM = 656
_GROUP = 128
_TOTAL_CACHED = 65536
_EXTRA_ROWS = 128
# (num_tokens,) — 65536 cached rows, 128 landing-pad rows.
SHAPES = [(t,) for t in (16, 128, 1024, 4096, 16384)]
MORE_SHAPES = [(7,), (333,), (2048,)]


def _input_fn(shape, cur_dtype, device):
    (num_tokens,) = shape
    g = torch.Generator(device=device).manual_seed(0)
    # Packed 656-byte rows: fp8 nope, fp32 group scales, bf16 rope.
    cache = torch.empty(
        (_TOTAL_CACHED, 1, _DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    flat = cache.view(_TOTAL_CACHED, _DIM)
    flat[:, :512] = torch.randn(
        _TOTAL_CACHED,
        512,
        dtype=torch.float32,
        device=device,
        generator=g,
    ).to(torch.float8_e4m3fn)
    scales = 0.05 * (
        1.0
        + torch.rand(
            _TOTAL_CACHED,
            4,
            dtype=torch.float32,
            device=device,
            generator=g,
        )
    )
    flat[:, 512:528] = scales.view(torch.float8_e4m3fn)
    rope = torch.randn(
        _TOTAL_CACHED,
        64,
        dtype=torch.float32,
        device=device,
        generator=g,
    ).to(torch.bfloat16)
    flat[:, 528:] = rope.view(torch.float8_e4m3fn)

    page_table = torch.randint(
        0,
        _TOTAL_CACHED,
        (num_tokens,),
        device=device,
        generator=g,
        dtype=torch.int32,
    )
    # A dirty persistent destination: every byte must be overwritten.
    out = torch.full(
        (num_tokens + _EXTRA_ROWS, 1, 576),
        3.0,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    yield dict(
        quant_k_cache=cache,
        page_table_1_flattened=page_table,
        group_size=_GROUP,
        extra_rows=_EXTRA_ROWS,
        out=out,
    )


@pytest.mark.gather_dequant_requant_fp8_paged
def test_perf_gather_dequant_requant_fp8_paged():
    gems_op = flaggems_sglang.get_op("gather_dequant_requant_fp8_paged")
    if gems_op is None:
        pytest.skip("dsa/gather_dequant_requant_fp8_paged not implemented yet")
    bench = OpBenchmark(
        op_name="gather_dequant_requant_fp8_paged",
        torch_op=get_reference("gather_dequant_requant_fp8_paged"),
        input_fn=_input_fn,
        dtypes=[torch.float8_e4m3fn],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens",
    )
    bench.set_gems(gems_op)
    bench.run()
