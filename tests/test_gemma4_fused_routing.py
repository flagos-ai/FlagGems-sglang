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

"""Correctness test for sampling/gemma4_fused_routing."""

import pytest
import torch

from flaggems_sglang.reference import get_reference

from . import conftest as cfg

_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)


def assert_close(actual, expected, *, dtype=None, **overrides):
    if not expected.dtype.is_floating_point:
        torch.testing.assert_close(actual, expected)
        return
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


def _case(t, e, k, dtype=torch.bfloat16, seed=0, strided=False, ties=False):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    if strided:
        base = torch.randn(
            t,
            3 * e,
            dtype=torch.float32,
            device=cfg.device,
            generator=g,
        ).to(dtype)
        gating = base[:, e : 2 * e]
    else:
        gating = torch.randn(
            t, e, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)
    if ties:
        gating = (gating.float() * 4).round().to(dtype)
    scale = (
        0.5 * torch.rand(e, dtype=torch.float32, device=cfg.device,
                         generator=g)
        + 0.5
    ).to(dtype)
    return dict(
        gating_output=gating,
        per_expert_scale=scale,
        topk=k,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(1, 256, 8),
    lambda: _case(17, 128, 6),
    lambda: _case(128, 128, 8, ties=True),
    lambda: _case(256, 64, 8, ties=True),
    lambda: _case(64, 128, 8, strided=True),
    lambda: _case(33, 8, 8),
    lambda: _case(33, 8, 1),
    lambda: _case(16, 256, 2, dtype=torch.float16),
    lambda: _case(64, 512, 8, dtype=torch.float32),
    lambda: _case(512, 8, 4, ties=True),
] + [
    (lambda t, e, k: lambda: _case(t, e, k))(t, e, k)
    for t, e, k in [
        (2048, 256, 8),
        (8192, 128, 6),
        (4096, 128, 8),
        (1024, 64, 2),
        (1, 64, 4),
    ]
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.gemma4_fused_routing
def test_gemma4_fused_routing(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("gemma4_fused_routing")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        from flaggems_sglang import gemma4_fused_routing
    except (ImportError, ModuleNotFoundError):
        pytest.skip("sampling/gemma4_fused_routing ops module not found")
        return

    try:
        actual = gemma4_fused_routing(**kwargs)
    except NotImplementedError:
        pytest.skip("sampling/gemma4_fused_routing not yet implemented")
        return

    if check is not None:
        if isinstance(expected, (tuple, list)):
            for a, e in zip(actual, expected):
                check(a, e)
        else:
            check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    if check is not None:
        case["check"] = check
