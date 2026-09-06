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

"""Correctness test for mrope_fused using a pure-torch reference.

The reference implementation is self-contained (no SGLang dependency) and
follows the spec from kernel-comp-baseline/problems/rope/mrope_fused.
"""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference.mrope_fused import reference as mrope_fused

from . import conftest as cfg

mrope_fused_ref = get_reference("mrope_fused")

# ---------------------------------------------------------------------------
# Test cases (from kernel-comp-baseline/problems/rope/mrope_fused/cases.py)
# ---------------------------------------------------------------------------

CORRECTNESS_CASES = [
    # (num_tokens, n_qh, n_kh, head_size, rotary_dim, mrope_section, max_pos)
    (1, 4, 1, 64, 64, [8, 12, 12], 4096),
    (37, 8, 2, 128, 128, [16, 24, 24], 4096),
    (129, 16, 2, 128, 64, [8, 12, 12], 4096),
]

BENCH_CASES = [
    (t, 8, 2, 128, 128, [16, 24, 24], 4096) for t in (1, 128, 2048, 8192)
]

ALL_CASES = CORRECTNESS_CASES + BENCH_CASES

CASE_IDS = [f"T{c[0]}_qh{c[1]}_kh{c[2]}_hd{c[3]}_rd{c[4]}" for c in ALL_CASES]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASES, ids=CASE_IDS)
@pytest.mark.mrope_fused
def test_mrope_fused(case):
    num_tokens, n_qh, n_kh, head_size, rotary_dim, mrope_section, max_pos = (
        case
    )
    device = cfg.device
    dtype = torch.bfloat16
    seed = 42

    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        num_tokens, n_qh * head_size, generator=g, device=device, dtype=dtype
    )
    k = torch.randn(
        num_tokens, n_kh * head_size, generator=g, device=device, dtype=dtype
    )
    cos_sin_cache = torch.randn(
        max_pos, rotary_dim, generator=g, device=device, dtype=dtype
    )
    positions = torch.randint(
        0,
        max_pos,
        (3, num_tokens),
        generator=g,
        device=device,
        dtype=torch.int64,
    )

    # Reference (pure torch)
    q_ref, k_ref = mrope_fused_ref(
        q,
        k,
        cos_sin_cache,
        positions,
        mrope_section,
        head_size,
        rotary_dim,
    )

    # Operator under test
    q_out, k_out = mrope_fused(
        q,
        k,
        cos_sin_cache,
        positions,
        mrope_section,
        head_size,
        rotary_dim,
    )

    atol, rtol = 1.5e-2, 1.5e-2
    torch.testing.assert_close(q_out, q_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_out, k_ref, atol=atol, rtol=rtol)
