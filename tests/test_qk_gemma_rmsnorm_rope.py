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

"""Correctness test for rope/qk_gemma_rmsnorm_rope."""

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


def _case(
    m, nq, nkv, head_dim, rotary_dim, neox=True, dtype=torch.bfloat16, seed=0
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)

    def _t(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)

    max_seq = 4096
    angle = (
        torch.rand(max_seq, rotary_dim // 2, device=cfg.device, generator=g)
        * 6.28
    )
    cos_sin_cache = torch.cat([angle.cos(), angle.sin()], dim=-1).to(dtype)
    return dict(
        q=_t(m, nq * head_dim).contiguous(),
        k=_t(m, nkv * head_dim).contiguous(),
        q_weight=(0.5 * _t(head_dim).float()).to(dtype),
        k_weight=(0.5 * _t(head_dim).float()).to(dtype),
        positions=torch.randint(
            0,
            max_seq,
            (m,),
            device=cfg.device,
            generator=g,
            dtype=torch.int32,
        ),
        cos_sin_cache=cos_sin_cache,
        eps=_EPS,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        is_neox_style=neox,
        check=_check,
    )


def _check(actual, expected):
    for a, e in zip(actual, expected):
        assert_close(a, e)


_CASE_FACTORIES = [
    lambda: _case(7, 8, 2, 128, 128),
    lambda: _case(1024, 16, 4, 128, 64),
    lambda: _case(333, 4, 1, 64, 64, neox=False),
    lambda: _case(256, 8, 2, 128, 128, neox=False),
    lambda: _case(256, 8, 2, 128, 128, dtype=torch.float16),
] + [
    (lambda m: lambda: _case(m, 32, 8, 128, 128))(m)
    for m in (1, 8, 64, 512, 4096)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.qk_gemma_rmsnorm_rope
def test_qk_gemma_rmsnorm_rope(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("qk_gemma_rmsnorm_rope")
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
        from flaggems_sglang import qk_gemma_rmsnorm_rope
    except (ImportError, ModuleNotFoundError):
        pytest.skip("rope/qk_gemma_rmsnorm_rope ops module not found")
        return

    try:
        actual = qk_gemma_rmsnorm_rope(**kwargs)
    except NotImplementedError:
        pytest.skip("rope/qk_gemma_rmsnorm_rope not yet implemented")
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
