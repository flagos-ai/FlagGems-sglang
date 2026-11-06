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

"""Correctness test for moe/silu_and_mul_routed."""

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


def _case(
    m,
    n,
    dtype=torch.bfloat16,
    seed=0,
    weights=False,
    interleaved=True,
    out_dtype=None,
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    gateup = torch.randn(
        m, 2 * n, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    w = (
        torch.rand(m, dtype=torch.float32, device=cfg.device, generator=g).to(
            dtype
        )
        if weights
        else None
    )
    return dict(
        gateup_output=gateup.contiguous(),
        topk_weights=w,
        out_dtype=out_dtype,
        use_interleaved=interleaved,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(7, 128),
    lambda: _case(1024, 2048, weights=True),
    lambda: _case(333, 1024, interleaved=False),
    lambda: _case(256, 512, weights=True, interleaved=False),
    lambda: _case(128, 256, out_dtype=torch.float32),
] + [
    (lambda m, n: lambda: _case(m, n, weights=True))(m, n)
    for m in (1, 8, 64, 512, 4096)
    for n in (1024, 4096)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.silu_and_mul_routed
def test_silu_and_mul_routed(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("silu_and_mul_routed")
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
        from flaggems_sglang import silu_and_mul_routed
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/silu_and_mul_routed ops module not found")
        return

    try:
        actual = silu_and_mul_routed(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/silu_and_mul_routed not yet implemented")
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
