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

"""Correctness test for moe/ep_gather."""

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


def _case(num_tokens, hidden, topk, num_src, seed=0, drop=0.0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    input_tensor = torch.randn(
        num_src, hidden, dtype=torch.float32, device=cfg.device, generator=g
    ).to(torch.bfloat16)
    ids = torch.randint(
        0,
        32,
        (num_tokens, topk),
        device=cfg.device,
        generator=g,
        dtype=torch.int32,
    )
    if drop:
        mask = torch.rand(ids.shape, device=cfg.device, generator=g) < drop
        ids = torch.where(mask, torch.full_like(ids, -1), ids)
    weight = torch.rand(
        num_tokens, topk, dtype=torch.float32, device=cfg.device, generator=g
    )
    index = torch.randint(
        0,
        num_src,
        (num_tokens, topk),
        device=cfg.device,
        generator=g,
        dtype=torch.int32,
    )
    # Destination-passing: every row is overwritten, so the fill is arbitrary.
    out = torch.full(
        (num_tokens, hidden), 5.0, dtype=torch.float32, device=cfg.device
    ).to(torch.bfloat16)
    return dict(
        input_tensor=input_tensor.contiguous(),
        recv_topk_ids=ids.contiguous(),
        recv_topk_weight=weight.contiguous(),
        input_index=index.contiguous(),
        output_tensor=out.contiguous(),
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(7, 1024, 4, 64),
    lambda: _case(1024, 2048, 8, 4096, drop=0.25),
    lambda: _case(333, 1024, 2, 1024),
] + [
    (lambda t: lambda: _case(t, 4096, 8, max(t * 8, 16)))(t)
    for t in (1, 8, 64, 512, 4096)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.ep_gather
def test_ep_gather(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("ep_gather")
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
        from flaggems_sglang import ep_gather
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/ep_gather ops module not found")
        return

    try:
        actual = ep_gather(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/ep_gather not yet implemented")
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
