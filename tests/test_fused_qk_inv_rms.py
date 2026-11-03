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

"""Correctness test for diffusion/fused_qk_inv_rms."""

import pytest
import torch

from flaggems_sglang.reference import get_reference

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

_EPS = 1e-6


def _case(b, n, heads, dim, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    qkv = torch.randn(
        b,
        n,
        3,
        heads,
        dim,
        dtype=torch.float32,
        device=cfg.device,
        generator=g,
    ).to(torch.bfloat16)
    return dict(qkv=qkv.contiguous(), eps=_EPS, check=_check)


def _check(actual, expected):
    for a, e in zip(actual, expected):
        assert_close(a, e, dtype=torch.float32, atol=1e-3, rtol=1e-3)


_CASE_FACTORIES = [
    lambda: _case(2, 16, 4, 64),
    lambda: _case(1, 1024, 16, 128),
    lambda: _case(4, 77, 8, 64),
] + [
    (lambda n: lambda: _case(1, n, 16, 128))(n) for n in (64, 512, 4096, 16384)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fused_qk_inv_rms
def test_fused_qk_inv_rms(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("fused_qk_inv_rms")
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
        from flaggems_sglang import fused_qk_inv_rms
    except (ImportError, ModuleNotFoundError):
        pytest.skip("diffusion/fused_qk_inv_rms ops module not found")
        return

    try:
        actual = fused_qk_inv_rms(**kwargs)
    except NotImplementedError:
        pytest.skip("diffusion/fused_qk_inv_rms not yet implemented")
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
