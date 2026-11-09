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

"""Correctness test for utility/normalize_and_patchify."""

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
    batch,
    channels,
    h,
    w,
    patch,
    pad_h=0,
    pad_w=0,
    seed=0,
    dtype=torch.float32,
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    image = torch.rand(
        batch,
        channels,
        h,
        w,
        dtype=torch.float32,
        device=cfg.device,
        generator=g,
    ).to(dtype)
    scale = torch.rand(
        channels, 1, 1, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    bias = torch.randn(
        channels, 1, 1, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    return dict(
        image=image.contiguous(),
        image_scale=scale.contiguous(),
        image_bias=bias.contiguous(),
        patch_size=patch,
        padded_height=h + pad_h,
        padded_width=w + pad_w,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(1, 3, 32, 32, patch=16),
    lambda: _case(4, 3, 224, 224, patch=14),
    # padding to a patch multiple
    lambda: _case(2, 3, 30, 30, patch=16, pad_h=2, pad_w=2),
    lambda: _case(1, 3, 64, 64, patch=16, dtype=torch.bfloat16),
] + [
    (lambda b: lambda: _case(b, 3, 448, 448, patch=14))(b) for b in (1, 4, 16)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.normalize_and_patchify
def test_normalize_and_patchify(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("normalize_and_patchify")
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
        from flaggems_sglang import normalize_and_patchify
    except (ImportError, ModuleNotFoundError):
        pytest.skip("utility/normalize_and_patchify ops module not found")
        return

    try:
        actual = normalize_and_patchify(**kwargs)
    except NotImplementedError:
        pytest.skip("utility/normalize_and_patchify not yet implemented")
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
