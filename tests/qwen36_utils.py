# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Assertions shared only by the new Qwen workload correctness suites."""

from copy import deepcopy

import torch

# Engineering thresholds, not values supplied by the workload documents.
# BF16 relative resolution follows the repository's testing.RESOLUTION (0.016).
BF16_TOLERANCE = dict(atol=2e-3, rtol=0.016)
STATE_TOLERANCE = dict(atol=2e-3, rtol=0.016)
FP32_TOLERANCE = dict(atol=2e-5, rtol=2e-5)
MUTATIONS = {
    "gemma_rmsnorm": ("out",),
    "silu_and_mul_out": ("out",),
    "gemma_fused_add_rmsnorm": ("input", "residual"),
    "chunk_gated_delta_rule_fwd": ("initial_state",),
    "fused_recurrent_gated_delta_rule_packed_decode": ("out", "initial_state"),
    "select_experts": (),
    "_fused_moe_kernel_sequence": ("hidden_states",),
}


class InputSnapshot(dict):
    """Value copies plus held allocations and immutable pre-call geometry."""

    def __init__(self):
        super().__init__()
        self.allocations = {}


def clone_inputs(inputs, device=None):
    copied = InputSnapshot()
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            copied.allocations[name] = (
                storage,
                (storage.data_ptr(), storage.nbytes(), value.storage_offset()),
            )
            backing = torch.empty(
                storage.nbytes() // value.element_size(),
                dtype=value.dtype,
                device=value.device if device is None else device,
            )
            copied[name] = backing.as_strided(
                value.shape, value.stride(), value.storage_offset()
            ).copy_(value)
        else:
            copied[name] = deepcopy(value)
    return copied


def compare_tensor(actual, expected, device, *, tolerance=None, exact=False):
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == torch.device(device)
    if exact or not actual.is_floating_point():
        tolerance = dict(atol=0, rtol=0)
    elif tolerance is None:
        tolerance = (
            BF16_TOLERANCE if actual.dtype == torch.bfloat16 else FP32_TOLERANCE
        )
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), **tolerance, equal_nan=False
    )


def compare_output(actual, expected, device, operator):
    if isinstance(expected, torch.Tensor):
        compare_tensor(actual, expected, device)
    elif expected is None:
        assert actual is None
    else:
        assert isinstance(actual, (tuple, list))
        assert len(actual) == len(expected)
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            if isinstance(wanted, torch.Tensor):
                tolerance = None
                if (
                    operator == "fused_recurrent_gated_delta_rule_packed_decode"
                    and index == 1
                ):
                    tolerance = STATE_TOLERANCE
                compare_tensor(got, wanted, device, tolerance=tolerance)
            else:
                assert got is wanted is None


def validate_result(
    operator, actual, expected, inputs, expected_inputs, before
):
    device = next(
        value.device
        for value in inputs.values()
        if isinstance(value, torch.Tensor)
    )
    if operator == "select_experts":
        assert isinstance(actual, tuple) and len(actual) == 3
        assert actual.format == 1
        weights, ids, logits = actual
        assert (
            weights.shape == expected[0].shape
            and weights.dtype == torch.float32
        )
        assert ids.shape == expected[1].shape and ids.dtype == torch.int32
        assert weights.device == ids.device == device
        assert logits is inputs["router_logits"]
        order, expected_order = ids.argsort(-1), expected[1].argsort(-1)
        compare_tensor(
            ids.gather(-1, order),
            expected[1].gather(-1, expected_order),
            device,
            exact=True,
        )
        compare_tensor(
            weights.gather(-1, order),
            expected[0].gather(-1, expected_order),
            device,
        )
        torch.testing.assert_close(
            weights.sum(-1).cpu(),
            torch.ones(weights.shape[0]),
            **FP32_TOLERANCE,
        )
    else:
        compare_output(actual, expected, device, operator)
    if operator in ("gemma_rmsnorm", "silu_and_mul_out"):
        if inputs["out"] is not None:
            assert actual is inputs["out"]
        else:
            assert actual is not inputs["input"] and actual.is_contiguous()
            assert (
                actual.untyped_storage().data_ptr()
                != inputs["input"].untyped_storage().data_ptr()
            )
    if operator == "fused_recurrent_gated_delta_rule_packed_decode":
        assert actual[0] is inputs["out"]
        assert actual[1] is inputs["initial_state"]
    if operator == "_fused_moe_kernel_sequence":
        assert actual is inputs["hidden_states"]
    mutations = MUTATIONS[operator]
    for name, original in before.items():
        value = inputs[name]
        if isinstance(original, torch.Tensor):
            storage = value.untyped_storage()
            assert (
                storage.data_ptr(),
                storage.nbytes(),
                value.storage_offset(),
            ) == before.allocations[name][1]
        if not isinstance(original, torch.Tensor):
            assert value == original
            continue
        assert (
            value.shape == original.shape
            and value.stride() == original.stride()
        )
        assert value.dtype == original.dtype and value.device == original.device
        if name not in mutations:
            compare_tensor(value, original, device, exact=True)
        else:
            compare_tensor(
                value, expected_inputs[name], device, tolerance=STATE_TOLERANCE
            )
    if "initial_state" in mutations:
        indices = inputs.get(
            "initial_state_indices", inputs.get("ssm_state_indices")
        )
        unchanged = torch.ones(
            inputs["initial_state"].shape[0], dtype=torch.bool, device=device
        )
        selected = indices[indices >= 0].long()
        unchanged[selected] = False
        assert torch.equal(
            inputs["initial_state"][unchanged],
            before["initial_state"][unchanged],
        )
