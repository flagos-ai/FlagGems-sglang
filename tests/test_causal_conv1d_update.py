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

"""Correctness test for mamba/causal_conv1d_update."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("causal_conv1d_update")


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
# Cases (from kernel-comp-baseline/problems/mamba/causal_conv1d_update/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    a_out, a_state = actual
    e_out, e_state = expected
    assert_close(a_out, e_out)
    assert_close(a_state, e_state)


def _case(
    batch, dim, width=4, seqlen=1, state_len=None, dtype=torch.bfloat16, seed=0
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    state_len = state_len if state_len is not None else width - 1
    x = torch.randn(
        batch,
        dim,
        seqlen,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    if seqlen == 1:
        x = x.squeeze(-1)
    conv_state = torch.randn(
        batch,
        dim,
        state_len,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    weight = torch.randn(
        dim,
        width,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    bias = torch.randn(
        dim, generator=g, device=flaggems_sglang.device, dtype=torch.float32
    ).to(dtype)
    return dict(
        x=x, conv_state=conv_state, weight=weight, bias=bias, check=_check
    )


CORRECTNESS_CASES = [
    _case(2, 16, width=4, seqlen=1),
    _case(5, 64, width=4, seqlen=1),
    _case(3, 32, width=3, seqlen=4),
]

BENCH_CASES = [
    _case(64, 4096, width=4, seqlen=1),
    _case(256, 2048, width=4, seqlen=1),
]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.causal_conv1d_update
def test_causal_conv1d_update(case_idx):
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
        from flaggems_sglang.ops.causal_conv1d_update import (
            causal_conv1d_update,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mamba/causal_conv1d_update ops module not found")
        return

    try:
        actual = causal_conv1d_update(**kwargs)
    except NotImplementedError:
        pytest.skip("mamba/causal_conv1d_update not yet implemented")
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
