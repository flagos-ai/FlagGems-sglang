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

"""Correctness test for dsa/dequantize_k_cache_fast."""

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
_FP8_MAX = 448.0
_DIM_NOPE = 512
_DIM_ROPE = 64


def _build_quant_k_cache(num_tokens, group_size=128, seed=0):
    """Pack [512 nope fp8 | 4 fp32 scales | 64 bf16 rope] into one fp8 row."""
    g = create_generator(cfg.device, seed=seed)
    k_nope = (
        2 * torch.randn(num_tokens, _DIM_NOPE, generator=g, device=cfg.device)
    ).to(torch.bfloat16)
    k_rope = torch.randn(
        num_tokens, _DIM_ROPE, generator=g, device=cfg.device
    ).to(torch.bfloat16)
    num_tiles = _DIM_NOPE // group_size

    buf = torch.empty(
        num_tokens,
        _DIM_NOPE + num_tiles * 4 + _DIM_ROPE * 2,
        dtype=_FP8_DTYPE,
        device=cfg.device,
    )
    nope_q = buf[:, :_DIM_NOPE]
    nope_s = buf[:, _DIM_NOPE : _DIM_NOPE + num_tiles * 4].view(torch.float32)
    rope = buf[:, _DIM_NOPE + num_tiles * 4 :].view(torch.bfloat16)
    rope[:] = k_rope

    for t in range(num_tiles):
        lo = t * group_size
        hi = lo + group_size
        tile = k_nope[:, lo:hi].float()
        scale = (tile.abs().amax(dim=-1) / _FP8_MAX).clamp(min=1e-12)
        nope_s[:, t] = scale
        nope_q[:, lo:hi] = (tile / scale.unsqueeze(-1)).to(_FP8_DTYPE)

    return buf


def _case(num_tokens, group_size=128, seed=0):
    return dict(
        quant_k_cache=_build_quant_k_cache(
            num_tokens, group_size=group_size, seed=seed
        )
    )


_CASE_FACTORIES = [
    lambda: _case(1),
    lambda: _case(37),
    lambda: _case(256),
] + [(lambda t: lambda: _case(t))(t) for t in (1, 128, 2048, 8192)]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.dequantize_k_cache_fast
def test_dequantize_k_cache_fast(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("dequantize_k_cache_fast")
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
            "dequantize_k_cache_fast", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("dsa/dequantize_k_cache_fast ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("dsa/dequantize_k_cache_fast not yet implemented")
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
