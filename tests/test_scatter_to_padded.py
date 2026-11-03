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

"""Correctness test for diffusion/scatter_to_padded."""

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


def _case(batch, seqlen, heads, head_dim, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    # Ragged lengths: each sequence keeps a prefix, the rest is padding.
    lens = torch.randint(
        1,
        seqlen + 1,
        (batch,),
        device=cfg.device,
        generator=g,
        dtype=torch.int64,
    )
    total = int(lens.sum())
    inv = torch.full(
        (batch * seqlen,), -1, dtype=torch.int32, device=cfg.device
    )
    cursor = 0
    for b in range(batch):
        n = int(lens[b])
        inv[b * seqlen : b * seqlen + n] = torch.arange(
            cursor, cursor + n, dtype=torch.int32, device=cfg.device
        )
        cursor += n
    out_unpad = torch.randn(
        total,
        heads,
        head_dim,
        dtype=torch.float32,
        device=cfg.device,
        generator=g,
    ).to(dtype)
    return dict(
        out_unpad=out_unpad.contiguous(),
        inv_indices=inv,
        batch_size=batch,
        seqlen=seqlen,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(2, 16, 4, 64),
    lambda: _case(8, 512, 16, 128),
    lambda: _case(1, 77, 8, 72),
] + [(lambda b: lambda: _case(b, 1024, 16, 128))(b) for b in (1, 4, 16, 64)]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.scatter_to_padded
def test_scatter_to_padded(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("scatter_to_padded")
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
        from flaggems_sglang import scatter_to_padded
    except (ImportError, ModuleNotFoundError):
        pytest.skip("diffusion/scatter_to_padded ops module not found")
        return

    try:
        actual = scatter_to_padded(**kwargs)
    except NotImplementedError:
        pytest.skip("diffusion/scatter_to_padded not yet implemented")
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
