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

"""Correctness test for quantization/per_token_group_quant_mla_masked_fp8."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

# ---------------------------------------------------------------------------
# Tolerance helper
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
# Cases
# ---------------------------------------------------------------------------


_GROUP = 128
_EPS = 1e-12


def _check(actual, expected):
    a_q, a_s, a_masked, a_m, a_am = actual
    e_q, e_s, e_masked, e_m, e_am = expected
    assert a_m == e_m and a_am == e_am, "m / aligned_m mismatch"
    assert_close(a_masked, e_masked)
    m = e_m
    # Only [:m] is defined: the baseline allocates x_q/x_s with new_empty and
    # the kernel's grid is (m, b), so the alignment padding is never written.
    assert_close(a_s[:, :m], e_s[:, :m], dtype=torch.float32)
    a_deq = a_q[:, :m].to(torch.float32)
    e_deq = e_q[:, :m].to(torch.float32)
    atol, rtol = 2e-2, 2e-2
    mismatch = (a_deq - e_deq).abs() > (atol + rtol * e_deq.abs())
    assert (
        mismatch.float().mean() < 1e-2
    ), "too many fp8 rounding-boundary mismatches"


def _case(b, m, k, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    x = torch.randn(
        b, m, k, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    return dict(x=x.contiguous(), group_size=_GROUP, eps=_EPS, check=_check)


_CASE_FACTORIES = [
    lambda: _case(1, 7, 512),
    lambda: _case(4, 128, 512),
    lambda: _case(8, 300, 1024),
] + [
    (lambda b, m: lambda: _case(b, m, 512))(b, m)
    for b in (1, 8, 32)
    for m in (16, 128, 1024)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.per_token_group_quant_mla_masked_fp8
def test_per_token_group_quant_mla_masked_fp8(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("per_token_group_quant_mla_masked_fp8")
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
        candidate = get_candidate(
            "per_token_group_quant_mla_masked_fp8", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "quantization/per_token_group_quant_mla_masked_fp8 ops module not found"
        )
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "quantization/per_token_group_quant_mla_masked_fp8 not yet implemented"
        )
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
