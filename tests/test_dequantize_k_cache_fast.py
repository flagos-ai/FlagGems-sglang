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

"""Correctness test for kv_cache/dequantize_k_cache_fast."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)


def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = dict(
        _TOLERANCES.get(
            dtype if dtype is not None else expected.dtype,
            _DEFAULT_TOLERANCE,
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


def _pack(num_rows, dim_nope, dim_rope, group_size=128, seed=0):
    """Build a packed row buffer with a valid fp8 / scale / rope layout."""
    dev = flaggems_sglang.device
    g = torch.Generator(device=dev).manual_seed(seed)
    num_tiles = dim_nope // group_size
    dim_quant = dim_nope + num_tiles * 4 + dim_rope * 2
    buf = torch.empty((num_rows, dim_quant), dtype=torch.uint8, device=dev)

    q = (
        torch.randn(
            (num_rows, dim_nope),
            generator=g,
            device=dev,
            dtype=torch.float32,
        )
        * 50.0
    ).to(torch.float8_e4m3fn)
    buf[:, :dim_nope] = q.view(torch.uint8)

    scale = (
        torch.rand(
            (num_rows, num_tiles),
            generator=g,
            device=dev,
            dtype=torch.float32,
        )
        * 0.02
        + 0.005
    )
    buf[:, dim_nope : dim_nope + num_tiles * 4] = scale.contiguous().view(
        torch.uint8
    )

    rope = torch.randn(
        (num_rows, dim_rope),
        generator=g,
        device=dev,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    buf[:, dim_nope + num_tiles * 4 :] = rope.view(torch.uint8)

    return buf.view(torch.float8_e4m3fn)


def _case(num_tokens, dim_nope, dim_rope, seed=0, block_size=None):
    packed = _pack(num_tokens, dim_nope, dim_rope, seed=seed)
    if block_size is not None:
        assert num_tokens % block_size == 0
        packed = packed.view(num_tokens // block_size, block_size, -1)
    return dict(
        quant_k_cache=packed,
        dim_nope=dim_nope,
        dim_rope=dim_rope,
    )


_CASE_FACTORIES = [
    lambda: _case(1, 512, 64),
    lambda: _case(37, 512, 64),
    lambda: _case(512, 512, 64),
    lambda: _case(2048, 512, 64),
    lambda: _case(129, 512, 64),
    lambda: _case(8, 256, 0),
    lambda: _case(8, 512, 0),
    lambda: _case(8, 128, 64),
    lambda: _case(8, 1024, 64),
    lambda: _case(8, 1024, 128),
    lambda: _case(64, 512, 64, block_size=8),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.dequantize_k_cache_fast
def test_dequantize_k_cache_fast(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("dequantize_k_cache_fast")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        candidate = get_candidate(
            "dequantize_k_cache_fast", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("kv_cache/dequantize_k_cache_fast ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("kv_cache/dequantize_k_cache_fast not yet implemented")
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
