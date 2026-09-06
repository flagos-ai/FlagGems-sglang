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

"""Benchmark for mrope_fused operator.

Uses the same shape grid as kernel-comp-baseline/problems/rope/mrope_fused
so speedup numbers are directly comparable. Benchmarks both the optimized
Triton kernel and the pure-torch reference to compute speedup.
"""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.ops.mrope_fused import mrope_fused
from flaggems_sglang.reference import get_reference

mrope_fused_ref = get_reference("mrope_fused")

# Bench shapes: (num_tokens, n_qh, n_kh, head_size, rotary_dim, mrope_section, max_pos)
BENCH_CASES = [
    (1, 8, 2, 128, 128, [16, 24, 24], 4096),
    (128, 8, 2, 128, 128, [16, 24, 24], 4096),
    (2048, 8, 2, 128, 128, [16, 24, 24], 4096),
    (8192, 8, 2, 128, 128, [16, 24, 24], 4096),
]

BENCH_IDS = [
    f"T{c[0]}_qh{c[1]}_kh{c[2]}_hd{c[3]}_rd{c[4]}" for c in BENCH_CASES
]


def _make_inputs(case, device):
    num_tokens, n_qh, n_kh, head_size, rotary_dim, mrope_section, max_pos = (
        case
    )
    dtype = torch.bfloat16
    q = torch.randn(num_tokens, n_qh * head_size, device=device, dtype=dtype)
    k = torch.randn(num_tokens, n_kh * head_size, device=device, dtype=dtype)
    cos_sin_cache = torch.randn(
        max_pos, rotary_dim, device=device, dtype=dtype
    )
    positions = torch.randint(
        0, max_pos, (3, num_tokens), device=device, dtype=torch.int64
    )
    return q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.mrope_fused
def test_mrope_fused_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    device = flaggems_sglang.device
    case = BENCH_CASES[case_idx]
    q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim = (
        _make_inputs(case, device)
    )

    def run_triton():
        return mrope_fused(
            q,
            k,
            cos_sin_cache,
            positions,
            mrope_section,
            head_size,
            rotary_dim,
        )

    def run_ref():
        return mrope_fused_ref(
            q,
            k,
            cos_sin_cache,
            positions,
            mrope_section,
            head_size,
            rotary_dim,
        )

    ref_us = do_bench_us(run_ref)
    triton_us = do_bench_us(run_triton)
    record_case("rope/mrope_fused", BENCH_IDS[case_idx], ref_us, triton_us)
