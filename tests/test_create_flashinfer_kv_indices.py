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

"""Correctness test for kvcache/create_flashinfer_kv_indices."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from . import conftest as cfg

reference = get_reference("create_flashinfer_kv_indices")


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
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        (
            expected.to(torch.float32)
            if expected.dtype.is_floating_point
            else expected
        ),
        **tol,
    )


# ---------------------------------------------------------------------------
# Cases (from kernel-comp-baseline/problems/kvcache/create_flashinfer_kv_indices/cases.py)
# ---------------------------------------------------------------------------

import torch

from . import conftest as cfg


def _case(
    bs, max_len=2048, max_batch=256, max_context=8192, with_start=False, seed=0
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    req_to_token = torch.randint(
        0,
        1 << 20,
        (max_batch, max_context),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    req_pool_indices = (
        torch.randperm(
            max_batch,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )[:bs]
        .to(torch.int32)
        .to(cfg.device)
    )
    lens = torch.randint(
        1,
        max_len + 1,
        (bs,),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=cfg.device)
    kv_indptr[1:] = torch.cumsum(lens, dim=0)
    kv_start = (
        torch.randint(
            0, 128, (bs,), dtype=torch.int32, device=cfg.device, generator=g
        )
        if with_start
        else None
    )
    kv_indices = torch.zeros(
        int(kv_indptr[-1]), dtype=torch.int32, device=cfg.device
    )
    return dict(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        page_kernel_lens=lens,
        kv_indptr=kv_indptr,
        kv_start_idx=kv_start,
        kv_indices=kv_indices,
        check=assert_close,
    )


CORRECTNESS_CASES = [
    _case(1),
    _case(16, seed=1),
    _case(64, max_len=512, seed=2),
    _case(32, max_len=1024, with_start=True, seed=3),
]

BENCH_CASES = [_case(bs, max_len=4096, seed=7) for bs in (1, 8, 32, 128)]

CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.create_flashinfer_kv_indices
def test_create_flashinfer_kv_indices(case_idx):
    case = CORRECTNESS_CASES[case_idx]
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
        from flaggems_sglang import create_flashinfer_kv_indices
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "kvcache/create_flashinfer_kv_indices ops module not found"
        )
        return

    try:
        actual = create_flashinfer_kv_indices(**kwargs)
    except NotImplementedError:
        pytest.skip("kvcache/create_flashinfer_kv_indices not yet implemented")
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
