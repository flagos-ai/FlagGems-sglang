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

"""Correctness test for attention/decode_grouped_attention."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("decode_grouped_attention")


# ---------------------------------------------------------------------------
# Tolerance helper (from kernel-comp-baseline/harness/correctness.py)
# ---------------------------------------------------------------------------

_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)


def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = dict(
        _TOLERANCES.get(
            dtype if dtype is not None else expected.dtype, _DEFAULT_TOLERANCE
        )
    )
    tol.update(overrides)
    equal_nan = tol.pop("equal_nan", True)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        (
            expected.to(torch.float32)
            if expected.dtype.is_floating_point
            else expected
        ),
        equal_nan=equal_nan,
        **tol,
    )


# ---------------------------------------------------------------------------
# Cases (from kernel-comp-baseline/problems/attention/decode_grouped_attention/cases.py)
# ---------------------------------------------------------------------------


def _case(B, H_Q, H_KV, D, seq_len, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    total_tokens = B * seq_len
    sm_scale = 1.0 / (D**0.5)

    q = torch.randn(
        B,
        H_Q,
        D,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    k_buffer = torch.randn(
        total_tokens,
        H_KV,
        D,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    v_buffer = torch.randn(
        total_tokens,
        H_KV,
        D,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)

    b_seq_len = torch.full((B,), seq_len, device=flaggems_sglang.device)
    kv_indptr = torch.zeros(
        (B + 1,), dtype=torch.int32, device=flaggems_sglang.device
    )
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
    kv_indices = torch.arange(
        total_tokens, device=flaggems_sglang.device, dtype=torch.int32
    )

    return dict(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        sm_scale=sm_scale,
        check=_check,
    )


def _check(actual, expected):
    assert_close(actual.to(torch.float32), expected, atol=3e-2, rtol=1e-2)


# H_Q:H_KV ratios chosen well above the MHA/small-GQA range already
# exercised by attention/decode_attention, to actually route through the
# grouped kernel (kv_group_num > 1) with a realistic MLA-style head count.
CORRECTNESS_CASES = [
    _case(2, 32, 1, 64, 10),
    _case(2, 128, 1, 64, 10),
    _case(2, 32, 4, 80, 10),
    _case(2, 128, 8, 512, 128),
]

BENCH_CASES = [
    (lambda B=B, seq_len=seq_len: _case(B, 128, 1, 128, seq_len))
    for B, seq_len in (
        (1, 2048),
        (8, 2048),
        (64, 512),
        (512, 128),
        (4096, 128),
    )
]
CORRECTNESS_CASES = CORRECTNESS_CASES + [c() for c in BENCH_CASES]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.decode_grouped_attention
def test_decode_grouped_attention(case_idx):
    case = CORRECTNESS_CASES[case_idx]
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    # Reference
    expected = reference(**kwargs)

    # Operator under test
    try:
        from flaggems_sglang.ops.decode_grouped_attention import (
            decode_grouped_attention,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/decode_grouped_attention ops module not found")
        return

    try:
        actual = decode_grouped_attention(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/decode_grouped_attention not yet implemented")
        return

    # Compare
    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if a is None and e is None:
                continue
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    # Restore check for reuse
    if check is not None:
        case["check"] = check
