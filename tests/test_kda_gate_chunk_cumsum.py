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

"""Correctness test for fla/kda_gate_chunk_cumsum."""

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
    b,
    t,
    h,
    s,
    chunk_size=64,
    seed=0,
    bias=False,
    scale=None,
    lower_bound=None,
):
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    g = torch.randn(
        b, t, h, s, dtype=torch.float32, device=cfg.device, generator=gen
    ).to(torch.bfloat16)
    a_log = (
        torch.randn(h, dtype=torch.float32, device=cfg.device, generator=gen)
        * 0.5
    )
    dt_bias = (
        torch.randn(
            h * s, dtype=torch.float32, device=cfg.device, generator=gen
        )
        * 0.1
        if bias
        else None
    )
    return dict(
        g=g.contiguous(),
        A_log=a_log,
        chunk_size=chunk_size,
        scale=scale,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        check=_check,
    )


def _check(actual, expected):
    assert_close(actual, expected, dtype=torch.float32, atol=1e-4, rtol=1e-4)


_CASE_FACTORIES = [
    lambda: _case(1, 64, 4, 128),
    lambda: _case(2, 256, 8, 64, bias=True),
    lambda: _case(1, 333, 4, 128),  # T not a multiple of the chunk
    lambda: _case(1, 128, 4, 128, scale=0.5),
    lambda: _case(1, 128, 4, 128, lower_bound=0.9),
] + [
    (lambda t: lambda: _case(1, t, 16, 128))(t) for t in (64, 512, 2048, 8192)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.kda_gate_chunk_cumsum
def test_kda_gate_chunk_cumsum(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("kda_gate_chunk_cumsum")
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
        from flaggems_sglang import kda_gate_chunk_cumsum
    except (ImportError, ModuleNotFoundError):
        pytest.skip("fla/kda_gate_chunk_cumsum ops module not found")
        return

    try:
        actual = kda_gate_chunk_cumsum(**kwargs)
    except NotImplementedError:
        pytest.skip("fla/kda_gate_chunk_cumsum not yet implemented")
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
