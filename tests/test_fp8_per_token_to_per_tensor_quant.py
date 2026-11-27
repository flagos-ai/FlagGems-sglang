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

"""Correctness test for moe/fp8_per_token_to_per_tensor_quant."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate
from flaggems_sglang.utils.rng_utils import create_generator

from . import conftest as cfg

# torch_gcu aborts the whole process when casting to these dtypes
# on-device (backend CHECK failure); skip every test in this module.
pytestmark = cfg.cast_skip_mark(torch.float8_e4m3fn)


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


_FP8_DTYPE = torch.float8_e4m3fn
_K_SCALE_BLOCK_SIZE = 128


def _check_factory(masked_m):
    def _check(actual, expected):
        for e in range(masked_m.shape[0]):
            n = int(masked_m[e].item())
            if n <= 0:
                continue
            a = actual[e, :n].to(torch.float32)
            b = expected[e, :n].to(torch.float32)
            atol, rtol = 2e-2, 2e-2
            mismatch = (a - b).abs() > (atol + rtol * b.abs())
            assert (
                mismatch.float().mean() < 3e-2
            ), "too many fp8 rounding-boundary mismatches"

    return _check


def _case(expert_num, token_num_padded, k, counts, seed=0):
    g = create_generator(cfg.device, seed=seed)
    num_groups = k // _K_SCALE_BLOCK_SIZE
    x = (
        2
        * torch.randn(
            expert_num,
            token_num_padded,
            k,
            generator=g,
            device=cfg.device,
        )
    ).to(_FP8_DTYPE)
    x_scale = (
        torch.rand(
            expert_num,
            token_num_padded,
            num_groups,
            generator=g,
            device=cfg.device,
            dtype=torch.float32,
        )
        * 0.1
        + 0.01
    )
    masked_m = torch.tensor(counts, dtype=torch.int32, device=cfg.device)
    output_scale = torch.tensor([0.02], dtype=torch.float32, device=cfg.device)
    return dict(
        x=x,
        x_scale=x_scale,
        masked_m=masked_m,
        output_scale=output_scale,
        check=_check_factory(masked_m),
    )


# K must be a multiple of 1024 (the kernel's fixed K_BLOCK_SIZE), not just
# 128 (the scale-group width).
_CASE_FACTORIES = [
    lambda: _case(4, 16, 1024, [0, 3, 16, 9]),
    lambda: _case(8, 128, 1024, [128, 0, 64, 1, 17, 128, 5, 90]),
] + [(lambda e: lambda: _case(e, 256, 4096, [256] * e))(e) for e in (4, 8, 32)]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fp8_per_token_to_per_tensor_quant
def test_fp8_per_token_to_per_tensor_quant(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("fp8_per_token_to_per_tensor_quant")
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
            "fp8_per_token_to_per_tensor_quant", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "moe/fp8_per_token_to_per_tensor_quant ops module not found"
        )
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "moe/fp8_per_token_to_per_tensor_quant not yet implemented"
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
