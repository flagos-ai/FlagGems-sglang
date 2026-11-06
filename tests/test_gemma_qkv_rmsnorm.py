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

"""Correctness test for activation_norm/gemma_qkv_rmsnorm."""

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
    m,
    nq,
    nkv,
    head_dim,
    dtype=torch.bfloat16,
    seed=0,
    split_view=False,
    kv=True,
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)

    def _t(cols):
        return torch.randn(
            m, cols, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)

    if split_view:
        # The layout the kernel's stride(0) handling exists for: q/k/v are
        # slices of one packed qkv buffer, so their rows are strided.
        qkv = _t((nq + 2 * nkv) * head_dim)
        q, k, v = qkv.split(
            [nq * head_dim, nkv * head_dim, nkv * head_dim], dim=-1
        )
    else:
        q, k, v = _t(nq * head_dim), _t(nkv * head_dim), _t(nkv * head_dim)
    if not kv:
        k = v = None

    q_weight = torch.randn(
        head_dim, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    k_weight = torch.randn(
        head_dim, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    return dict(
        q=q,
        k=k,
        v=v,
        q_weight=q_weight,
        k_weight=k_weight,
        num_q_heads=nq,
        num_kv_heads=nkv,
        head_dim=head_dim,
        eps=_EPS,
        check=_check,
    )


def _check(actual, expected):
    for a, e in zip(actual, expected):
        if e is None:
            assert a is None
            continue
        assert_close(a, e)


_CASE_FACTORIES = [
    lambda: _case(7, 8, 2, 128),
    lambda: _case(256, 16, 4, 128),
    # M > 256 takes the one-program-per-token branch instead of (M, heads).
    lambda: _case(1024, 8, 1, 256),
    lambda: _case(64, 8, 2, 128, split_view=True),
    lambda: _case(64, 8, 2, 128, kv=False),
] + [
    (lambda m: lambda: _case(m, 32, 8, 128))(m) for m in (1, 8, 64, 512, 4096)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.gemma_qkv_rmsnorm
def test_gemma_qkv_rmsnorm(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("gemma_qkv_rmsnorm")
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
        from flaggems_sglang import gemma_qkv_rmsnorm
    except (ImportError, ModuleNotFoundError):
        pytest.skip("activation_norm/gemma_qkv_rmsnorm ops module not found")
        return

    try:
        actual = gemma_qkv_rmsnorm(**kwargs)
    except NotImplementedError:
        pytest.skip("activation_norm/gemma_qkv_rmsnorm not yet implemented")
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
