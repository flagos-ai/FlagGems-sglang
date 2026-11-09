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

"""Correctness test for fla/chunk_kda_scaled_dot_kkt."""

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


def _case(b, t, h, kdim, chunk_size=64, seed=0, gk_scale=1.0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    q = torch.randn(
        b, t, h, kdim, dtype=torch.float32, device=cfg.device, generator=g
    ).to(torch.bfloat16)
    k = torch.randn(
        b, t, h, kdim, dtype=torch.float32, device=cfg.device, generator=g
    ).to(torch.bfloat16)
    # gk is a within-chunk cumsum of a non-positive gate, so it decreases down
    # the chunk; keep the span small so exp2 of the difference stays bounded.
    gate = -0.05 * torch.rand(
        b, t, h, kdim, dtype=torch.float32, device=cfg.device, generator=g
    )
    gk = (
        gate.reshape(b, -1, chunk_size, h, kdim)
        .cumsum(dim=2)
        .reshape(b, t, h, kdim)
    )
    beta = torch.rand(
        b, t, h, dtype=torch.float32, device=cfg.device, generator=g
    )
    return dict(
        q=q.contiguous(),
        k=k.contiguous(),
        gk=gk.contiguous(),
        beta=beta.contiguous(),
        scale=kdim**-0.5,
        gk_scale=gk_scale,
        chunk_size=chunk_size,
        check=_check,
    )


def _check(actual, expected):
    for a, e in zip(actual, expected):
        assert_close(a, e, dtype=torch.float32, atol=2e-2, rtol=2e-2)


_CASE_FACTORIES = [
    lambda: _case(1, 64, 4, 64),
    lambda: _case(2, 256, 4, 128),
    # RCP_LN2: a natural-log gk converted inside the kernel.
    lambda: _case(1, 128, 8, 64, gk_scale=1.4426950408889634),
] + [(lambda t: lambda: _case(1, t, 8, 128))(t) for t in (64, 256, 1024, 4096)]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.chunk_kda_scaled_dot_kkt
def test_chunk_kda_scaled_dot_kkt(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("chunk_kda_scaled_dot_kkt")
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
        from flaggems_sglang import chunk_kda_scaled_dot_kkt
    except (ImportError, ModuleNotFoundError):
        pytest.skip("fla/chunk_kda_scaled_dot_kkt ops module not found")
        return

    try:
        actual = chunk_kda_scaled_dot_kkt(**kwargs)
    except NotImplementedError:
        pytest.skip("fla/chunk_kda_scaled_dot_kkt not yet implemented")
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
