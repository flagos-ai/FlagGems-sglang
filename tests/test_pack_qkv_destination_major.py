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

"""Correctness test for diffusion/pack_qkv_destination_major."""

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
    rows,
    global_heads,
    head_size,
    world_size,
    seed=0,
    dtype=torch.bfloat16,
    pass_out=False,
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)

    def _t():
        return torch.randn(
            rows,
            global_heads,
            head_size,
            dtype=torch.float32,
            device=cfg.device,
            generator=g,
        ).to(dtype)

    out = None
    if pass_out:
        out = torch.empty(
            (world_size, rows, global_heads // world_size, 3 * head_size),
            dtype=dtype,
            device=cfg.device,
        )
    return dict(
        q=_t().contiguous(),
        k=_t().contiguous(),
        v=_t().contiguous(),
        world_size=world_size,
        out=out,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(64, 8, 64, world_size=2),
    lambda: _case(1024, 32, 128, world_size=8),
    lambda: _case(77, 16, 72, world_size=4, pass_out=True),
] + [
    (lambda r: lambda: _case(r, 32, 128, world_size=8))(r)
    for r in (64, 1024, 4096, 16384)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.pack_qkv_destination_major
def test_pack_qkv_destination_major(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("pack_qkv_destination_major")
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
        from flaggems_sglang import pack_qkv_destination_major
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "diffusion/pack_qkv_destination_major ops module not found"
        )
        return

    try:
        actual = pack_qkv_destination_major(**kwargs)
    except NotImplementedError:
        pytest.skip("diffusion/pack_qkv_destination_major not yet implemented")
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
