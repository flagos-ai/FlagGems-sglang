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

"""Correctness test for moe/silu_mul_dynamic_tensorwise_quant."""

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
    a_out, a_scale = actual
    e_out, e_scale = expected
    # The scale is a global absmax reduced with atomic_max: order-independent.
    assert_close(a_scale, e_scale, dtype=torch.float32)
    a_deq = a_out.to(torch.float32) * a_scale
    e_deq = e_out.to(torch.float32) * e_scale
    atol, rtol = 2e-2, 2e-2
    mismatch = (a_deq - e_deq).abs() > (atol + rtol * e_deq.abs())
    assert (
        mismatch.float().mean() < 1e-2
    ), "too many fp8 rounding-boundary mismatches"


def _case(expected_tokens, live_tokens, inter, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    inp = torch.randn(
        expected_tokens,
        2 * inter,
        dtype=torch.float32,
        device=cfg.device,
        generator=g,
    ).to(dtype)
    return dict(
        input=inp.contiguous(),
        output=torch.zeros(
            (expected_tokens, inter),
            dtype=torch.float8_e4m3fn,
            device=cfg.device,
        ),
        scale=torch.zeros(1, dtype=torch.float32, device=cfg.device),
        num_tokens_tensor=torch.tensor(
            [live_tokens], dtype=torch.int32, device=cfg.device
        ),
        expected_num_tokens=expected_tokens,
        intermediate_size=inter,
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(128, 128, 512),
    # Live region shorter than the buffer.
    lambda: _case(1024, 700, 1024),
    lambda: _case(256, 256, 2048),
] + [(lambda t: lambda: _case(t, t, 2048))(t) for t in (64, 512, 4096)]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.silu_mul_dynamic_tensorwise_quant
def test_silu_mul_dynamic_tensorwise_quant(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("silu_mul_dynamic_tensorwise_quant")
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
            "silu_mul_dynamic_tensorwise_quant", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "moe/silu_mul_dynamic_tensorwise_quant ops module not found"
        )
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "moe/silu_mul_dynamic_tensorwise_quant not yet implemented"
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
