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

"""Correctness test for fla/kda_prologue."""

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

_BT = 64


def _case(seqlens, hv, kdim, scale=0.125, seed=0):
    """Sequence lengths are multiples of the kernel's BT=64 chunk.

    The six outputs come from torch.empty_like, so rows a chunk masks off are
    never written; keeping every chunk full means every row is defined.
    """
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    t = sum(seqlens)
    q = torch.randn(
        t, hv, kdim, dtype=torch.float32, device=cfg.device, generator=g
    ).to(torch.bfloat16)
    k = torch.randn(
        t, hv, kdim, dtype=torch.float32, device=cfg.device, generator=g
    ).to(torch.bfloat16)
    # KDA gates are log-space decays (<= 0). Keep them small in magnitude so
    # exp(g_cu - g_last) stays inside bf16 range over a 64-row chunk.
    g_act = (
        -0.02
        * torch.rand(
            t, hv, kdim, dtype=torch.float32, device=cfg.device, generator=g
        )
    ).to(torch.bfloat16)

    cu = torch.tensor(
        [0] + list(torch.tensor(seqlens).cumsum(0)),
        dtype=torch.int32,
        device=cfg.device,
    )
    idx = []
    for s, n in enumerate(seqlens):
        idx += [(s, c) for c in range(n // _BT)]
    chunk_indices = torch.tensor(idx, dtype=torch.int32, device=cfg.device)
    return dict(
        q=q.contiguous(),
        k=k.contiguous(),
        g_act=g_act.contiguous(),
        scale=scale,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
        num_chunks=len(idx),
        check=_check,
    )


def _check(actual, expected):
    for a, e in zip(actual, expected):
        assert_close(a, e)


_CASE_FACTORIES = [
    lambda: _case([64], 4, 128),
    lambda: _case([128, 64, 256], 8, 128),
    lambda: _case([512], 16, 64),
] + [(lambda n: lambda: _case([n], 16, 128))(n) for n in (64, 512, 2048, 8192)]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.kda_prologue
def test_kda_prologue(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("kda_prologue")
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
        from flaggems_sglang import kda_prologue
    except (ImportError, ModuleNotFoundError):
        pytest.skip("fla/kda_prologue ops module not found")
        return

    try:
        actual = kda_prologue(**kwargs)
    except NotImplementedError:
        pytest.skip("fla/kda_prologue not yet implemented")
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
