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

"""Correctness test for mamba/state_passing."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference.chunk_cumsum import (
    reference as chunk_cumsum_reference,
)

reference = get_reference("state_passing")


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
# Cases (from kernel-comp-baseline/problems/mamba/state_passing)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    a_out, a_final = actual
    e_out, e_final = expected
    assert_close(a_out, e_out)
    assert_close(a_final, e_final, dtype=torch.float32)


def _case(
    batch,
    nchunks,
    chunk_size,
    nheads,
    dim,
    has_init=False,
    dtype=torch.bfloat16,
    seed=0,
):
    device = flaggems_sglang.device
    g = torch.Generator(device=device).manual_seed(seed)
    states = torch.randn(
        batch,
        nchunks,
        nheads,
        dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(dtype)
    raw_dt = torch.rand(
        batch,
        nchunks * chunk_size,
        nheads,
        generator=g,
        device=device,
        dtype=torch.float32,
    )
    a = (
        -torch.rand(nheads, generator=g, device=device, dtype=torch.float32)
        - 0.1
    )
    # dA_cumsum comes from the chunk_cumsum stage that precedes this op.
    _, dA_cumsum = chunk_cumsum_reference(raw_dt, a, chunk_size)
    initial_states = None
    if has_init:
        initial_states = torch.randn(
            batch, nheads, dim, generator=g, device=device, dtype=torch.float32
        ).to(dtype)
    return dict(
        states=states,
        dA_cumsum=dA_cumsum,
        initial_states=initial_states,
        check=_check,
    )


CORRECTNESS_CASES = [
    _case(1, 1, 8, 4, 16),
    _case(2, 3, 16, 8, 32, has_init=True),
    _case(3, 5, 32, 16, 64),
]

BENCH_CASES = [
    _case(8, 16, 256, 32, 8192),
    _case(32, 4, 256, 64, 8192),
]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.state_passing
def test_state_passing(case_idx):
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
        from flaggems_sglang import state_passing
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mamba/state_passing ops module not found")
        return

    try:
        actual = state_passing(**kwargs)
    except NotImplementedError:
        pytest.skip("mamba/state_passing not yet implemented")
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
