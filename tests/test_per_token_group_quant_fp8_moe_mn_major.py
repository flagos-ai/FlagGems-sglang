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

"""Correctness test for quantization/per_token_group_quant_fp8_moe_mn_major."""

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
        (
            expected.to(torch.float32)
            if expected.dtype.is_floating_point
            else expected
        ),
        equal_nan=True,
        **tol,
    )


_GROUP = 128


def _case(rows_per_expert, k, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    m_total = sum(rows_per_expert)
    A = torch.randn(
        m_total,
        k,
        dtype=torch.float32,
        device=flaggems_sglang.device,
        generator=g,
    ).to(dtype)
    offs = torch.tensor(
        [0] + list(torch.tensor(rows_per_expert).cumsum(0).tolist()),
        dtype=torch.int32,
        device=flaggems_sglang.device,
    )
    problem_sizes = torch.zeros(
        (len(rows_per_expert), 3),
        dtype=torch.int32,
        device=flaggems_sglang.device,
    )
    problem_sizes[:, 0] = torch.tensor(
        rows_per_expert, dtype=torch.int32, device=flaggems_sglang.device
    )
    problem_sizes[:, 1] = k
    problem_sizes[:, 2] = k
    return dict(
        A=A.contiguous(),
        expert_offsets=offs,
        problem_sizes=problem_sizes,
        group_size=_GROUP,
        check=_check,
    )


def _check(actual, expected):
    aq, asc = actual
    eq, esc = expected
    assert_close(asc, esc, dtype=torch.float32)
    a_deq = aq.to(torch.float32)
    e_deq = eq.to(torch.float32)
    atol, rtol = 2e-2, 2e-2
    mismatch = (a_deq - e_deq).abs() > (atol + rtol * e_deq.abs())
    assert (
        mismatch.float().mean() < 1e-2
    ), "too many fp8 rounding-boundary mismatches"


_CORRECTNESS_CASES = [
    lambda: _case([128], 512),
    lambda: _case([64, 192, 0, 256], 1024),
    lambda: _case([333, 7, 512], 2048),
]
_CORRECTNESS_CASES += [
    (lambda e=e, n=n, k=k: _case([n] * e, k))
    for e, n in ((8, 256), (32, 128))
    for k in (2048, 4096, 7168)
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.per_token_group_quant_fp8_moe_mn_major
def test_per_token_group_quant_fp8_moe_mn_major(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("per_token_group_quant_fp8_moe_mn_major")(
        **kwargs
    )

    try:
        from flaggems_sglang.ops.per_token_group_quant_fp8_moe_mn_major import (
            per_token_group_quant_fp8_moe_mn_major,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "quantization/per_token_group_quant_fp8_moe_mn_major ops module not found"
        )
        return

    try:
        actual = per_token_group_quant_fp8_moe_mn_major(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "quantization/per_token_group_quant_fp8_moe_mn_major not yet implemented"
        )
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
