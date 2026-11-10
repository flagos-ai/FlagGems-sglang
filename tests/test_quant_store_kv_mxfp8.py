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

"""Correctness test for quantization/quant_store_kv_mxfp8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

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
        expected.to(torch.float32) if expected.dtype.is_floating_point else expected,
        equal_nan=True,
        **tol,
    )


_PAGE = 128


def _case(t, nkv, d, num_pages, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    k = torch.randn(
        t, nkv, d, dtype=torch.float32, device=flaggems_sglang.device, generator=g
    ).to(dtype)
    v = torch.randn(
        t, nkv, d, dtype=torch.float32, device=flaggems_sglang.device, generator=g
    ).to(dtype)
    slots = torch.randperm(
        num_pages * _PAGE, generator=g, device=flaggems_sglang.device
    )[:t]
    loc = slots.to(torch.int32)

    sf = d // 32
    k_cache = torch.zeros(
        (num_pages * _PAGE, nkv, d),
        dtype=torch.float8_e4m3fn,
        device=flaggems_sglang.device,
    )
    v_cache = torch.zeros_like(k_cache)
    sfk = torch.zeros(
        (num_pages, nkv, 32, _PAGE // 32, sf),
        dtype=torch.uint8,
        device=flaggems_sglang.device,
    )
    sfv = torch.zeros_like(sfk)
    return dict(
        k=k.contiguous(),
        v=v.contiguous(),
        loc=loc,
        k_cache=k_cache,
        v_cache=v_cache,
        sfk=sfk,
        sfv=sfv,
        page_size=_PAGE,
        check=_check,
    )


def _check(actual, expected):
    a_kc, a_vc, a_sk, a_sv = actual
    e_kc, e_vc, e_sk, e_sv = expected
    assert_close(a_sk, e_sk)
    assert_close(a_sv, e_sv)
    assert_close(a_kc.to(torch.float32), e_kc.to(torch.float32), atol=0, rtol=0)
    assert_close(a_vc.to(torch.float32), e_vc.to(torch.float32), atol=0, rtol=0)


_CORRECTNESS_CASES = [
    lambda: _case(16, 1, 128, num_pages=2),
    lambda: _case(129, 2, 64, num_pages=4),
    lambda: _case(512, 1, 512, num_pages=8),
]
_CORRECTNESS_CASES += [
    (lambda t=t, nkv=nkv: _case(t, nkv, 128, num_pages=max(2, t // _PAGE + 1)))
    for t in (16, 128, 1024, 4096)
    for nkv in (1, 2)
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.quant_store_kv_mxfp8
def test_quant_store_kv_mxfp8(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = case.pop("check", None) if isinstance(case, dict) and "check" in case else None
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("quant_store_kv_mxfp8")(**kwargs)

    try:
        from flaggems_sglang.ops.quant_store_kv_mxfp8 import quant_store_kv_mxfp8
    except (ImportError, ModuleNotFoundError):
        pytest.skip("quantization/quant_store_kv_mxfp8 ops module not found")
        return

    try:
        actual = quant_store_kv_mxfp8(**kwargs)
    except NotImplementedError:
        pytest.skip("quantization/quant_store_kv_mxfp8 not yet implemented")
        return

    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    if check is not None:
        case["check"] = check
