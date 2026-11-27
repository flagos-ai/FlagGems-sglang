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

"""Correctness test for dsa/triton_sparse_mla_fwd."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg
from flaggems_sglang.utils.rng_utils import create_generator

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


_DIM = 576
_D_V = 512


def _check(actual, expected):
    # The kernel requantizes the softmax probabilities to fp8
    # (p * 448 -> e4m3) before the P@V dot, so it cannot match an fp32
    # reference tightly. e4m3 carries 3 mantissa bits (~6% half-ulp), and
    # averaging over topk keys damps that to the low percent range.
    assert_close(actual, expected, atol=6e-2, rtol=6e-2)


def _case(seq, h, topk, num_pages, invalid_frac=0.0, seed=0):
    g = create_generator(cfg.device, seed=seed)
    fp8 = torch.float8_e4m3fn

    def _q(width):
        return (
            0.5
            * torch.randn(
                seq,
                h,
                width,
                dtype=torch.float32,
                device=cfg.device,
                generator=g,
            )
        ).to(fp8)

    kv = (
        0.5
        * torch.randn(
            num_pages,
            1,
            _DIM,
            dtype=torch.float32,
            device=cfg.device,
            generator=g,
        )
    ).to(fp8)
    idx = torch.randint(
        0,
        num_pages,
        (seq, 1, topk),
        device=cfg.device,
        generator=g,
        dtype=torch.int32,
    )
    if invalid_frac:
        # -1 slots are masked out; a row of all-invalid keys must give zeros.
        drop = (
            torch.rand(idx.shape, device=cfg.device, generator=g)
            < invalid_frac
        )
        idx = torch.where(drop, torch.full_like(idx, -1), idx)
    return dict(
        q_nope=_q(_D_V),
        q_rope=_q(_DIM - _D_V),
        kv=kv,
        indices=idx,
        sm_scale=(_DIM**-0.5),
        d_v=_D_V,
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(4, 16, 64, num_pages=256),
    lambda: _case(64, 16, 256, num_pages=4096),
    lambda: _case(32, 16, 128, num_pages=1024, invalid_frac=0.3),
] + [
    (lambda seq, topk: lambda: _case(seq, 16, topk, num_pages=8192))(seq, topk)
    for seq in (1, 16, 128)
    for topk in (128, 512, 2048)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.triton_sparse_mla_fwd
def test_triton_sparse_mla_fwd(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("triton_sparse_mla_fwd")
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
        candidate = get_candidate("triton_sparse_mla_fwd", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("dsa/triton_sparse_mla_fwd ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("dsa/triton_sparse_mla_fwd not yet implemented")
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
