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

"""Benchmark for attention/sglang_fl_mrotary_embedding."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (num_tokens, head_size, rotary_dim, num_q_heads, num_k_heads). The first two
# rows are the contract's own geometry; the rest widen the token axis and
# shrink the head to separate the per-token cost from the per-head cost.
SHAPES = [
    (256, 256, 64, 32, 8),
    (2048, 256, 64, 32, 8),
    (128, 128, 128, 16, 8),
    (4096, 256, 64, 32, 8),
]

MORE_SHAPES = [
    (1, 256, 64, 32, 8),
    (64, 64, 64, 8, 8),
]

_MAX_POS = 4096


def _sections(rotary_dim):
    """Three axis quotas that sum to rotary_dim // 2, contract-shaped."""
    half = rotary_dim // 2
    base = [half // 3] * 3
    for i in range(half % 3):
        base[i] += 1
    return base


def _cos_sin_cache(rotary_dim, dtype, device, base=100000.0):
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, device=device).float()
            / float(rotary_dim)
        )
    )
    positions = torch.arange(_MAX_POS, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def _input_fn(shape, cur_dtype, device):
    num_tokens, head_size, rotary_dim, num_qh, num_kh = shape
    g = torch.Generator(device=device).manual_seed(0)

    # The op mutates q and k in place, so every case gets fresh operands
    # rather than a buffer already rotated by the previous iteration.
    q = torch.randn(
        num_tokens,
        num_qh * head_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    k = torch.randn(
        num_tokens,
        num_kh * head_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)

    cache = _cos_sin_cache(rotary_dim, cur_dtype, device)
    positions = torch.randint(
        0,
        _MAX_POS,
        (3, num_tokens),
        generator=g,
        device=device,
        dtype=torch.int64,
    )

    yield (
        q,
        k,
        cache,
        positions,
        _sections(rotary_dim),
        head_size,
        rotary_dim,
        False,
        False,
        True,
        None,
    )


def _case_fn(shape, cur_dtype):
    num_tokens, head_size, rotary_dim, num_qh, num_kh = shape
    yield BenchmarkCasePlan(
        shape={
            "q": (num_tokens, num_qh * head_size),
            "k": (num_tokens, num_kh * head_size),
            "cos_sin_cache": (_MAX_POS, rotary_dim),
            "positions": (3, num_tokens),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "num_tokens": num_tokens,
            "head_size": head_size,
            "rotary_dim": rotary_dim,
            "num_q_heads": num_qh,
            "num_k_heads": num_kh,
            "sections": _sections(rotary_dim),
        },
        builder_args=(shape,),
    )


@pytest.mark.sglang_fl_mrotary_embedding
def test_perf_sglang_fl_mrotary_embedding():
    bench = OpBenchmark(
        op_name="sglang_fl_mrotary_embedding",
        torch_op=get_reference("sglang_fl_mrotary_embedding"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16, torch.float16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc=(
            "num_tokens, head_size, rotary_dim, num_q_heads, num_k_heads"
        ),
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
