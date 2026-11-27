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

"""Correctness test for dsa/gather_dequant_requant_fp8_paged."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg
from flaggems_sglang.utils.rng_utils import create_generator

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


_DIM = 656
_GROUP = 128


def _check(actual, expected):
    assert_close(actual.view(torch.uint8), expected.view(torch.uint8))


def _case(total_cached, num_tokens, extra_rows, seed=0):
    g = create_generator(cfg.device, seed=seed)
    # Build the packed 656-byte rows: fp8 nope, fp32 group scales, bf16 rope.
    cache = torch.empty(
        (total_cached, 1, _DIM),
        dtype=torch.float8_e4m3fn,
        device=cfg.device,
    )
    flat = cache.view(total_cached, _DIM)
    flat[:, :512] = torch.randn(
        total_cached,
        512,
        dtype=torch.float32,
        device=cfg.device,
        generator=g,
    ).to(torch.float8_e4m3fn)
    scales = 0.05 * (
        1.0
        + torch.rand(
            total_cached,
            4,
            dtype=torch.float32,
            device=cfg.device,
            generator=g,
        )
    )
    flat[:, 512:528] = scales.view(torch.float8_e4m3fn)
    rope = torch.randn(
        total_cached,
        64,
        dtype=torch.float32,
        device=cfg.device,
        generator=g,
    ).to(torch.bfloat16)
    flat[:, 528:] = rope.view(torch.float8_e4m3fn)

    page_table = torch.randint(
        0,
        total_cached,
        (num_tokens,),
        device=cfg.device,
        generator=g,
        dtype=torch.int32,
    )
    # A dirty persistent destination: the kernel must overwrite every byte,
    # including the landing-pad rows.
    out = torch.full(
        (num_tokens + extra_rows, 1, 576),
        3.0,
        dtype=torch.float32,
        device=cfg.device,
    ).to(torch.float8_e4m3fn)
    return dict(
        quant_k_cache=cache,
        page_table_1_flattened=page_table,
        group_size=_GROUP,
        extra_rows=extra_rows,
        out=out,
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(1024, 7, 1),
    lambda: _case(4096, 1024, 128),
    lambda: _case(2048, 333, 0),
] + [
    (lambda t: lambda: _case(65536, t, 128))(t)
    for t in (16, 128, 1024, 4096, 16384)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.gather_dequant_requant_fp8_paged
def test_gather_dequant_requant_fp8_paged(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("gather_dequant_requant_fp8_paged")
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
            "gather_dequant_requant_fp8_paged", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "dsa/gather_dequant_requant_fp8_paged ops module not found"
        )
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("dsa/gather_dequant_requant_fp8_paged not yet implemented")
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
