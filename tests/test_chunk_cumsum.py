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

"""Correctness test for mamba/chunk_cumsum."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("chunk_cumsum")


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
# Cases (from kernel-comp-baseline/problems/mamba/chunk_cumsum/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    a_dt, a_dA = actual
    e_dt, e_dA = expected
    assert_close(a_dt, e_dt, dtype=torch.float32)
    assert_close(a_dA, e_dA, dtype=torch.float32)


def _case(
    batch,
    nchunks,
    chunk_size,
    nheads,
    dt_bias=False,
    dt_softplus=False,
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    seqlen = nchunks * chunk_size
    dt = torch.randn(
        batch,
        seqlen,
        nheads,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    a = (
        -torch.rand(
            nheads,
            generator=g,
            device=flaggems_sglang.device,
            dtype=torch.float32,
        )
        - 0.1
    )
    bias = None
    if dt_bias:
        bias = torch.randn(
            nheads,
            generator=g,
            device=flaggems_sglang.device,
            dtype=torch.float32,
        )
    return dict(
        dt=dt,
        A=a,
        chunk_size=chunk_size,
        dt_bias=bias,
        dt_softplus=dt_softplus,
        check=_check,
    )


CORRECTNESS_CASES = [
    _case(1, 1, 8, 4),
    _case(2, 3, 16, 8, dt_bias=True, dt_softplus=True),
    _case(3, 2, 32, 16, dt_bias=True),
]

BENCH_CASES = [
    _case(8, 16, 256, 32, dt_bias=True, dt_softplus=True),
    _case(32, 4, 256, 64, dt_bias=True, dt_softplus=True),
]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.chunk_cumsum
def test_chunk_cumsum(case_idx):
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
        from flaggems_sglang.ops.chunk_cumsum import chunk_cumsum
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mamba/chunk_cumsum ops module not found")
        return

    try:
        actual = chunk_cumsum(**kwargs)
    except NotImplementedError:
        pytest.skip("mamba/chunk_cumsum not yet implemented")
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
