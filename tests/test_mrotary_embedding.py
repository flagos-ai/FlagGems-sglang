# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare candidates with native NVIDIA or portable Torch mRoPE."""

from copy import deepcopy

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference import mrotary_embedding as native
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

# Keep the six documented workloads, plus one correctness-only routing case.
WORKLOADS = native.WORKLOADS + [dict(native.WORKLOADS[4], axis_routing=True)]


def _make_case(case):
    if cfg.TO_CPU:
        pytest.fail("This suite uses the device reference; omit --ref cpu.")
    inputs = native.build_inputs(case, torch.bfloat16, cfg.device)
    if case.get("axis_routing"):
        inputs["positions"].add_(
            torch.arange(3, device=cfg.device, dtype=torch.int64)[:, None] * 7
        )
    reference_inputs = {
        name: (
            torch.empty_strided(
                value.shape,
                value.stride(),
                dtype=value.dtype,
                device=value.device,
            ).copy_(value)
            if isinstance(value, torch.Tensor)
            else deepcopy(value)
        )
        for name, value in inputs.items()
    }
    storage = {}
    for name in ("q", "k"):
        handle = inputs[name].untyped_storage()
        # Retain the allocation, but compare immutable snapshots: resize_ can
        # change a retained Storage object's pointer and size in place.
        storage[name] = (
            handle,
            (
                handle.data_ptr(),
                handle.nbytes(),
                inputs[name].storage_offset(),
            ),
        )

    def check(actual, expected):
        # Both callables return None. Their in-place q/k results are compared.
        assert actual is expected is None
        for name, value in inputs.items():
            wanted = reference_inputs[name]
            if not isinstance(value, torch.Tensor):
                assert value == wanted
                continue
            assert (
                value.shape == wanted.shape
                and value.stride() == wanted.stride()
            )
            assert (
                value.dtype == wanted.dtype and value.device == wanted.device
            )
            if name in ("q", "k"):
                current = value.untyped_storage()
                assert (
                    current.data_ptr(),
                    current.nbytes(),
                    value.storage_offset(),
                ) == storage[name][1]
                torch.testing.assert_close(
                    value, wanted, atol=2e-3, rtol=0.016, equal_nan=False
                )
                assert torch.equal(
                    value.view(case["tokens"], -1, 256)[..., 64:],
                    wanted.view(case["tokens"], -1, 256)[..., 64:],
                )
            else:
                assert torch.equal(value, wanted)

    return dict(inputs, _reference_inputs=reference_inputs, check=check)


_CASE_FACTORIES = [lambda row=row: _make_case(row) for row in WORKLOADS]


@pytest.mark.mrotary_embedding
@pytest.mark.parametrize(
    "case_idx",
    range(len(WORKLOADS)),
    ids=[
        f"{c['model']}-{c['tokens']}-stride{c['position_stride']}"
        + ("-axis-routing" if c.get("axis_routing") else "")
        for c in WORKLOADS
    ],
)
def test_mrotary_embedding(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    check = case.pop("check")
    reference_inputs = case.pop("_reference_inputs", case)
    reference = get_reference("mrotary_embedding")
    expected = reference(**reference_inputs)
    try:
        candidate = get_candidate("mrotary_embedding", reference=reference)
    except ImportError:
        pytest.skip("mrotary_embedding: supply a candidate with --override")
    actual = candidate(**case)
    check(actual, expected)
