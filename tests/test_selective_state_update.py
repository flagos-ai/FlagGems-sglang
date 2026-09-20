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

"""Correctness test for mamba/selective_state_update."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("selective_state_update")


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
# Cases (from kernel-comp-baseline/problems/mamba/selective_state_update)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    a_y, a_state = actual
    e_y, e_state = expected
    assert_close(a_y, e_y)
    assert_close(a_state, e_state)


def _case(
    batch,
    nheads,
    dim,
    dstate,
    ngroups=1,
    has_z=True,
    has_d=True,
    has_dt_bias=True,
    dt_softplus=True,
    dtype=torch.bfloat16,
    seed=0,
):
    device = flaggems_sglang.device
    g = torch.Generator(device=device).manual_seed(seed)
    state = torch.randn(
        batch,
        nheads,
        dim,
        dstate,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(dtype)
    x = torch.randn(
        batch, nheads, dim, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    dt = torch.randn(
        batch, nheads, dim, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    a = (
        -torch.rand(
            nheads,
            dim,
            dstate,
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        - 0.1
    )
    b = torch.randn(
        batch, ngroups, dstate, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    c = torch.randn(
        batch, ngroups, dstate, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    # D / z / dt_bias are all optional per the signature; the kernel guards
    # them with USE_D / USE_Z / USE_DT_BIAS.
    d = None
    if has_d:
        d = torch.randn(
            nheads, dim, generator=g, device=device, dtype=torch.float32
        )
    z = None
    if has_z:
        z = torch.randn(
            batch, nheads, dim, generator=g, device=device, dtype=torch.float32
        ).to(dtype)
    bias = None
    if has_dt_bias:
        bias = torch.randn(
            nheads, dim, generator=g, device=device, dtype=torch.float32
        )

    return dict(
        state=state,
        x=x,
        dt=dt,
        A=a,
        B=b,
        C=c,
        D=d,
        z=z,
        dt_bias=bias,
        dt_softplus=dt_softplus,
        check=_check,
    )


CORRECTNESS_CASES = [
    _case(1, 4, 16, 8),
    _case(5, 8, 64, 16, ngroups=2),
    _case(3, 16, 128, 32, ngroups=4, has_z=False, dt_softplus=False),
    # All three optional tensors absent, exercising the None guards.
    _case(
        2,
        8,
        32,
        16,
        ngroups=2,
        has_z=False,
        has_d=False,
        has_dt_bias=False,
        dt_softplus=False,
    ),
]

BENCH_CASES = [
    _case(64, 32, 128, 128, ngroups=8),
    _case(256, 64, 64, 128, ngroups=8),
]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.selective_state_update
def test_selective_state_update(case_idx):
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
        from flaggems_sglang import selective_state_update
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mamba/selective_state_update ops module not found")
        return

    try:
        actual = selective_state_update(**kwargs)
    except NotImplementedError:
        pytest.skip("mamba/selective_state_update not yet implemented")
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
