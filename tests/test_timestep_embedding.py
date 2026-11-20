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

"""Correctness test for embedding/timestep_embedding."""

import pytest
import torch

from flaggems_sglang.reference import get_reference

from . import conftest as cfg

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


def _case(
    n,
    dim,
    flip=True,
    shift=0.0,
    scale=1.0,
    max_period=10000,
    int_t=False,
    seed=0,
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    if int_t:
        t = torch.randint(
            0, 1000, (n,), dtype=torch.int64, device=cfg.device,
            generator=g,
        )
    else:
        t = (
            1000.0
            * torch.rand(n, device=cfg.device, generator=g)
        )
    return dict(
        t=t,
        dim=dim,
        flip_sin_to_cos=flip,
        downscale_freq_shift=shift,
        scale=scale,
        max_period=max_period,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(8, 256),
    lambda: _case(8, 256, flip=False),
    lambda: _case(4, 512, shift=1.0),
    lambda: _case(4, 257),
    lambda: _case(16, 16, scale=1000.0),
    lambda: _case(8, 256, int_t=True),
    lambda: _case(1, 32, max_period=100),
    lambda: _case(1024, 512, flip=False, scale=100.0),
    lambda: _case(33, 96),
    lambda: _case(2048, 1280),
    lambda: _case(128, 1280, shift=1.0, scale=10.0),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.timestep_embedding
def test_timestep_embedding(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("timestep_embedding")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        from flaggems_sglang import timestep_embedding
    except (ImportError, ModuleNotFoundError):
        pytest.skip("embedding/timestep_embedding ops module not found")
        return

    try:
        actual = timestep_embedding(**kwargs)
    except NotImplementedError:
        pytest.skip("embedding/timestep_embedding not yet implemented")
        return

    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    if check is not None:
        case["check"] = check
