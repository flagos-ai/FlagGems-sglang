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

"""Correctness test for mamba/chunk_state_varlen."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("chunk_state_varlen")


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
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        (
            expected.to(torch.float32)
            if expected.dtype.is_floating_point
            else expected
        ),
        **tol,
    )


# ---------------------------------------------------------------------------
# Cases (from kernel-comp-baseline/problems/mamba/chunk_state_varlen/cases.py)
# ---------------------------------------------------------------------------


from flaggems_sglang.reference.chunk_cumsum import (
    reference as chunk_cumsum_reference,
)


def _check(actual, expected):
    assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def _case(
    seq_lens,
    chunk_size,
    nheads,
    ngroups,
    headdim,
    dstate,
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)

    # Tight-packed cu_seqlens, with `seq_lens` chosen (by the caller) so that
    # every running cumulative sum which ends a chunk lands exactly on a
    # chunk_size multiple -- i.e. no sequence straddles a chunk boundary.
    # That keeps every sequence within a single physical chunk, matching
    # this problem's documented scope (no initial-state / cross-chunk path).
    cu_list = [0]
    for length in seq_lens:
        cu_list.append(cu_list[-1] + length)
    total_seqlen = cu_list[-1]
    assert total_seqlen % chunk_size == 0
    nchunks = total_seqlen // chunk_size
    cu_seqlens = torch.tensor(
        cu_list, dtype=torch.int32, device=flaggems_sglang.device
    )

    x = torch.randn(
        total_seqlen,
        nheads,
        headdim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    b = torch.randn(
        total_seqlen,
        ngroups,
        dstate,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    raw_dt = torch.rand(
        1,
        total_seqlen,
        nheads,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    )
    a = (
        -torch.rand(
            nheads,
            generator=g,
            device=flaggems_sglang.device,
            dtype=torch.float32,
        )
        - 0.1
    )
    dt_out, dA_cumsum = chunk_cumsum_reference(raw_dt, a, chunk_size)
    dt_out = dt_out.squeeze(0)
    dA_cumsum = dA_cumsum.squeeze(0)

    chunk_states = torch.randn(
        nchunks,
        nheads,
        headdim,
        dstate,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)

    return dict(
        B=b,
        x=x,
        dt=dt_out,
        dA_cumsum=dA_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_states=chunk_states,
        check=_check,
    )


CORRECTNESS_CASES = [
    _case([3, 5], 8, 4, 2, 16, 8),
    _case([10, 6, 4, 12], 16, 8, 2, 32, 16),
    _case([20, 12, 5, 27, 32], 32, 16, 4, 64, 32),
]

BENCH_CASES = [
    _case([256] * 8, 256, 32, 8, 64, 128),
    _case([256] * 32, 256, 64, 8, 64, 128),
]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.chunk_state_varlen
def test_chunk_state_varlen(case_idx):
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
        from flaggems_sglang.ops.chunk_state_varlen import chunk_state_varlen
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mamba/chunk_state_varlen ops module not found")
        return

    try:
        actual = chunk_state_varlen(**kwargs)
    except NotImplementedError:
        pytest.skip("mamba/chunk_state_varlen not yet implemented")
        return

    # Compare
    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    # Restore check for reuse
    if check is not None:
        case["check"] = check
