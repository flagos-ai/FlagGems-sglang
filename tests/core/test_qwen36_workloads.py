# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Counterexamples through the real new correctness test bodies."""

import importlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference.qwen36 import StandardTopKOutput
from flaggems_sglang.testing.qwen36_workloads import OPERATORS, workloads


@pytest.mark.parametrize("operator", OPERATORS)
def test_definition_matches_the_exercised_reference_signature(operator):
    root = Path(__file__).resolve().parents[2]
    data = json.loads((root / "definitions" / f"{operator}.json").read_text())
    signature = inspect.signature(get_reference(operator))
    assert data["name"] == operator and data["api_version"] == "v6.0"
    assert [item["name"] for item in data["parameters"]] == list(
        signature.parameters
    )
    for item, parameter in zip(
        data["parameters"], signature.parameters.values()
    ):
        assert item["kind"] == parameter.kind.name.lower()
        required = parameter.default is inspect.Parameter.empty
        assert item["required"] is required
        if not required:
            assert item["default"] == parameter.default
    effects = data["effects"]
    assert set(effects["mutates"]) <= set(signature.parameters)
    assert set(effects["returns_alias_of"]) <= set(data["outputs"])


def _invoke_body(operator, monkeypatch, mutate=None):
    module = importlib.import_module(f"tests.test_{operator}")
    row = dict(workloads(operator)[-1], pool=3)
    reference = get_reference(operator)
    calls = []

    def candidate(**kwargs):
        result = reference(**kwargs)
        calls.append(1)
        return result if mutate is None else mutate(result, kwargs)

    monkeypatch.setattr(module.cfg, "device", "cpu")
    monkeypatch.setattr(module.cfg, "TO_CPU", False)
    monkeypatch.setattr(module, "get_candidate", lambda *a, **kw: candidate)
    getattr(module, f"test_{operator}")(row)
    assert calls == [1]


@pytest.mark.parametrize(
    "operator",
    [
        "gemma_rmsnorm",
        "gemma_fused_add_rmsnorm",
        "chunk_gated_delta_rule_fwd",
        "fused_recurrent_gated_delta_rule_packed_decode",
        "select_experts",
    ],
)
def test_honest_reference_executes_the_original_assertions(
    operator, monkeypatch
):
    _invoke_body(operator, monkeypatch)


def test_gemma_cannot_replace_output_with_zeros(monkeypatch):
    with pytest.raises(AssertionError):
        _invoke_body(
            "gemma_rmsnorm", monkeypatch, lambda out, _: torch.zeros_like(out)
        )


def test_gemma_readonly_weight_write_is_detected(monkeypatch):
    def mutation(result, inputs):
        inputs["weight"].add_(0.125)
        return result

    with pytest.raises(AssertionError):
        _invoke_body("gemma_rmsnorm", monkeypatch, mutation)


def test_packed_decode_cannot_return_nonalias_copies(monkeypatch):
    def mutation(result, inputs):
        return result[0].clone(), result[1].clone()

    with pytest.raises(AssertionError):
        _invoke_body(
            "fused_recurrent_gated_delta_rule_packed_decode",
            monkeypatch,
            mutation,
        )


def test_unselected_state_writes_are_exact_even_below_numeric_tolerance(
    monkeypatch,
):
    def mutation(result, inputs):
        inputs["initial_state"][0, 0, 0, 0] += 1e-4
        return result

    with pytest.raises(AssertionError):
        _invoke_body(
            "fused_recurrent_gated_delta_rule_packed_decode",
            monkeypatch,
            mutation,
        )


def test_chunk_intermediate_output_cannot_be_discarded(monkeypatch):
    def mutation(result, inputs):
        changed = list(result)
        changed[2] = torch.zeros_like(changed[2])
        return tuple(changed)

    with pytest.raises(AssertionError):
        _invoke_body("chunk_gated_delta_rule_fwd", monkeypatch, mutation)


def test_chunk_none_slots_are_part_of_the_contract(monkeypatch):
    def mutation(result, inputs):
        changed = list(result)
        changed[3] = torch.zeros(1)
        return tuple(changed)

    with pytest.raises(AssertionError):
        _invoke_body("chunk_gated_delta_rule_fwd", monkeypatch, mutation)


def test_topk_allows_consistent_expert_permutations(monkeypatch):
    def mutation(result, inputs):
        return StandardTopKOutput(
            result[0].flip(-1), result[1].flip(-1), result[2]
        )

    _invoke_body("select_experts", monkeypatch, mutation)


def test_topk_rejects_disassociated_ids_and_weights(monkeypatch):
    def mutation(result, inputs):
        return StandardTopKOutput(result[0], result[1].flip(-1), result[2])

    with pytest.raises(AssertionError):
        _invoke_body("select_experts", monkeypatch, mutation)


def test_topk_preserves_router_logits_alias(monkeypatch):
    def mutation(result, inputs):
        return StandardTopKOutput(result[0], result[1], result[2].clone())

    with pytest.raises(AssertionError):
        _invoke_body("select_experts", monkeypatch, mutation)


def test_silu_out_native_abi_checks_supplied_output_alias(monkeypatch):
    _invoke_body("silu_and_mul_out", monkeypatch)
    with pytest.raises(AssertionError):
        _invoke_body(
            "silu_and_mul_out",
            monkeypatch,
            lambda result, inputs: result.clone(),
        )


def test_gemma_reallocation_cannot_update_the_storage_snapshot(monkeypatch):
    def mutate(result, inputs):
        value = inputs["input"]
        shape, stride, offset = (
            value.shape,
            value.stride(),
            value.storage_offset(),
        )
        value.resize_(
            value.untyped_storage().nbytes() // value.element_size() * 2
        )
        value.as_strided_(shape, stride, offset)
        return result

    with pytest.raises(AssertionError):
        _invoke_body("gemma_rmsnorm", monkeypatch, mutate)
