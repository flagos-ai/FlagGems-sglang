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

"""Benchmark for rope/ernie45_rope_fused."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/rope/ernie45_rope_fused.
# mrope_section is [section_h, section_w, section_t] with
# section_h == section_w and the three summing to rotary_dim // 2
# (Ernie4.5 layout), so it is derived from rotary_dim rather than listed.
SHAPES = [(t, 8, 2, 128, 128) for t in (1, 128, 2048, 8192)]
MORE_SHAPES = [(1, 4, 1, 64, 64), (37, 8, 2, 128, 128), (129, 16, 2, 128, 64)]

_MAX_POS = 4096


def _input_fn(shape, cur_dtype, device):
    num_tokens, n_qh, n_kh, head_size, rotary_dim = shape
    g = torch.Generator(device=device).manual_seed(0)
    # section_h == section_w == rotary_dim // 8, section_t takes the rest of
    # rotary_dim // 2.
    section_h = rotary_dim // 8
    mrope_section = [
        section_h,
        section_h,
        rotary_dim // 2 - 2 * section_h,
    ]
    q = torch.randn(
        num_tokens,
        n_qh * head_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    k = torch.randn(
        num_tokens,
        n_kh * head_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    cos_sin_cache = torch.randn(
        _MAX_POS,
        rotary_dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    positions = torch.randint(
        0,
        _MAX_POS,
        (3, num_tokens),
        generator=g,
        device=device,
        dtype=torch.int64,
    )
    yield q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim


@pytest.mark.ernie45_rope_fused
def test_perf_ernie45_rope_fused():
    bench = OpBenchmark(
        op_name="ernie45_rope_fused",
        torch_op=get_reference("ernie45_rope_fused"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, n_qh, n_kh, head_size, rotary_dim",
    )
    bench.set_gems(flaggems_sglang.ernie45_rope_fused)
    bench.run()
