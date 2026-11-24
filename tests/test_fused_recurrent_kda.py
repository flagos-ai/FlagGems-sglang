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

"""Correctness test for fla/fused_recurrent_kda."""

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


def _check(actual, expected):
    a_o, a_s = actual
    e_o, e_s = expected
    assert_close(a_o, e_o, atol=3e-2, rtol=3e-2)
    assert_close(a_s, e_s, dtype=torch.float32, atol=3e-2, rtol=3e-2)


def _case(
    b,
    t,
    h,
    kdim,
    vdim,
    dtype=torch.bfloat16,
    seed=0,
    l2norm=False,
    headwise=False,
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)

    def _t(*dims):
        return torch.randn(
            *dims, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)

    beta = torch.rand(
        b,
        t,
        h,
        vdim if headwise else 1,
        dtype=torch.float32,
        device=cfg.device,
        generator=g,
    )
    if not headwise:
        beta = beta.squeeze(-1)
    # Log-space decay, <= 0.
    gate = -0.05 * torch.rand(
        b, t, h, kdim, dtype=torch.float32, device=cfg.device, generator=g
    )
    return dict(
        q=_t(b, t, h, kdim).contiguous(),
        k=_t(b, t, h, kdim).contiguous(),
        v=_t(b, t, h, vdim).contiguous(),
        g=gate.contiguous(),
        beta=beta.to(dtype).contiguous(),
        scale=kdim**-0.5,
        initial_state=torch.zeros(
            b, h, vdim, kdim, dtype=torch.float32, device=cfg.device
        ),
        use_qk_l2norm_in_kernel=l2norm,
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(1, 16, 2, 64, 64),
    lambda: _case(2, 64, 4, 128, 128),
    lambda: _case(1, 32, 2, 64, 64, l2norm=True),
    lambda: _case(1, 32, 2, 64, 64, headwise=True),
] + [(lambda b: lambda: _case(b, 64, 8, 128, 128))(b) for b in (1, 4, 16, 64)]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fused_recurrent_kda
def test_fused_recurrent_kda(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("fused_recurrent_kda")
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
        candidate = get_candidate("fused_recurrent_kda", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("fla/fused_recurrent_kda ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("fla/fused_recurrent_kda not yet implemented")
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
