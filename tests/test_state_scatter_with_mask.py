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

"""Correctness test for mamba/state_scatter_with_mask."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

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
    # NaN at the same position is a match (both sides overflowed the same
    # way); NaN vs a real value still fails.
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
# Cases (from
# kernel-comp-baseline/problems/mamba/state_scatter_with_mask/cases.py)
# ---------------------------------------------------------------------------


def _case(
    layers,
    cache,
    requests,
    draft,
    state_shape,
    invalid=0.0,
    seed=0,
    dtype=torch.float32,
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(seed)
    dst = torch.randn(
        layers,
        cache,
        *state_shape,
        dtype=torch.float32,
        device=cfg.device,
        generator=g,
    ).to(dtype)
    src = torch.randn(
        layers,
        requests,
        draft,
        *state_shape,
        dtype=torch.float32,
        device=cfg.device,
        generator=g,
    ).to(dtype)
    dst_idx = torch.randperm(cache, device="cpu", generator=perm_g)[
        :requests
    ].to(device=cfg.device, dtype=torch.int32)
    step = torch.randint(
        0,
        draft,
        (requests,),
        device=cfg.device,
        generator=g,
        dtype=torch.int32,
    )
    if invalid:
        mask = torch.rand(step.shape, device=cfg.device, generator=g) < invalid
        step = torch.where(mask, torch.full_like(step, -1), step)
    return dict(
        dst=dst.contiguous(),
        src=src.contiguous(),
        dst_indices_raw=dst_idx,
        step_indices_raw=step,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(4, 64, 8, 4, (16, 32)),
    lambda: _case(2, 256, 64, 8, (8, 128), invalid=0.4),
    lambda: _case(1, 32, 32, 2, (64,), dtype=torch.bfloat16),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.state_scatter_with_mask
def test_state_scatter_with_mask(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("state_scatter_with_mask")
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
            "state_scatter_with_mask", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mamba/state_scatter_with_mask ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("mamba/state_scatter_with_mask not yet implemented")
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
