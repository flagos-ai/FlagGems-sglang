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

"""Correctness test for mamba/causal_conv1d_fn."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("causal_conv1d_fn")


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
# Cases (from kernel-comp-baseline/problems/mamba/causal_conv1d_fn/cases.py)
# ---------------------------------------------------------------------------


def _case(seq_lens, dim, width=4, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    total = sum(seq_lens)
    x = torch.randn(
        dim,
        total,
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
    query_start_loc = torch.zeros(
        len(seq_lens) + 1, dtype=torch.int32, device=flaggems_sglang.device
    )
    query_start_loc[1:] = torch.cumsum(
        torch.tensor(
            seq_lens, dtype=torch.int32, device=flaggems_sglang.device
        ),
        dim=0,
    )
    return dict(
        x=x,
        weight=weight,
        bias=bias,
        query_start_loc=query_start_loc,
        seq_lens_cpu=list(seq_lens),
    )


CORRECTNESS_CASES = [
    _case([7], 16),
    _case([5, 9, 3], 32, width=4),
    _case([1, 1, 1, 1], 8, width=3),
]

BENCH_CASES = [
    _case([2048] * 8, 4096),
    _case([512] * 32, 2048),
]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.causal_conv1d_fn
def test_causal_conv1d_fn(case_idx):
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
        from flaggems_sglang.ops.causal_conv1d_fn import causal_conv1d_fn
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mamba/causal_conv1d_fn ops module not found")
        return

    try:
        actual = causal_conv1d_fn(**kwargs)
    except NotImplementedError:
        pytest.skip("mamba/causal_conv1d_fn not yet implemented")
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
