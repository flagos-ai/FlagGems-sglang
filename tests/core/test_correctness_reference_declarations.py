# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Check explicit selection through real operator test bodies."""

import importlib
from pathlib import Path

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference_candidate import reference_test_mode

ROOT = Path(__file__).resolve().parents[2]
OPERATORS = sorted(p.stem[5:] for p in (ROOT / "tests").glob("test_*.py"))
DICT_OPERATORS = [
    op
    for op in OPERATORS
    if op not in {"mrope_fused", "softcap_inplace_logits"}
]


@pytest.mark.parametrize("operator", DICT_OPERATORS)
@pytest.mark.parametrize("enabled", [False, True])
def test_real_test_bodies_select_explicit_references_without_copying(
    operator, enabled, monkeypatch
):
    module = importlib.import_module(f"tests.test_{operator}")

    class Input(list):
        def __deepcopy__(self, memo):
            raise AssertionError("test input must not be implicitly copied")

    value = Input([1])
    calls, received = [], []

    def operation(kind, left, right):
        assert left is right is value
        received.append(left)
        calls.append(kind)
        return tuple(left)

    def reference(**inputs):
        return operation("reference", **inputs)

    def candidate(**inputs):
        return operation("candidate", **inputs)

    def compare(actual, expected):
        calls.append("compare")
        assert actual == expected == (1,)

    def resolve(name):
        assert name == operator
        return reference

    original_get_candidate = module.get_candidate

    def select(name, *, reference):
        assert name == operator
        assert reference is resolve(operator)
        calls.append("select")
        return original_get_candidate(name, reference=reference)

    monkeypatch.setattr(module, "get_reference", resolve)
    monkeypatch.setattr(module, "get_candidate", select)
    monkeypatch.setattr(
        module,
        "_CASE_FACTORIES",
        [lambda: dict(left=value, right=value, check=compare)],
    )
    monkeypatch.setattr(flaggems_sglang, operator, candidate, raising=False)
    with reference_test_mode(enabled, operator) as state:
        getattr(module, f"test_{operator}")(0)
    assert calls == [
        "reference",
        "select",
        "reference" if enabled else "candidate",
        "compare",
    ]
    assert received[0] is received[1] is value
    assert value == [1]
    assert state.candidate_calls == state.reference_calls == int(enabled)


def test_default_test_body_retains_reference_mutation_visibility(monkeypatch):
    module = importlib.import_module("tests.test_add3")
    original, order = [1], []

    def reference(payload):
        assert payload is original
        order.append("reference")
        payload.append(2)
        return tuple(payload)

    def candidate(payload):
        assert payload is original
        assert payload == [1, 2]
        order.append("candidate")
        return tuple(payload)

    def compare(actual, expected):
        order.append("compare")
        assert actual == expected == (1, 2)

    monkeypatch.setattr(module, "get_reference", lambda name: reference)
    monkeypatch.setattr(
        module,
        "_CASE_FACTORIES",
        [lambda: dict(payload=original, check=compare)],
    )
    monkeypatch.setattr(flaggems_sglang, "add3", candidate)
    module.test_add3(0)
    assert order == ["reference", "candidate", "compare"]
    assert original == [1, 2]


@pytest.mark.parametrize("enabled", [False, True])
def test_inplace_test_retains_its_one_explicit_input_copy(
    enabled, monkeypatch
):
    module = importlib.import_module("tests.test_softcap_inplace_logits")
    original = torch.tensor([[2.0, -5.0]])
    seen, received, copied = [], [], []
    original_clone = torch.Tensor.clone

    def clone(tensor, *args, **kwargs):
        if tensor is original:
            copied.append(tensor)
        return original_clone(tensor, *args, **kwargs)

    def reference(full_logits, final_logit_softcapping):
        seen.append("reference")
        received.append(full_logits)
        return full_logits / final_logit_softcapping

    def candidate(full_logits, final_logit_softcapping):
        seen.append("candidate")
        received.append(full_logits)
        return full_logits.div_(final_logit_softcapping)

    monkeypatch.setattr(module, "get_reference", lambda name: reference)
    monkeypatch.setattr(
        module,
        "_CASE_FACTORIES",
        [lambda: dict(full_logits=original, final_logit_softcapping=2.0)],
    )
    monkeypatch.setattr(flaggems_sglang, "softcap_inplace_logits", candidate)
    monkeypatch.setattr(torch.Tensor, "clone", clone)
    with reference_test_mode(enabled) as state:
        module.test_softcap_inplace_logits(0)
    torch.testing.assert_close(original, torch.tensor([[2.0, -5.0]]))
    assert seen == ["reference", "reference" if enabled else "candidate"]
    assert received[0] is original
    assert received[1] is not original
    assert len(copied) == 1 and copied[0] is original
    assert state.candidate_calls == int(enabled)


@pytest.mark.parametrize("enabled", [False, True])
def test_mrope_keeps_positional_inputs_and_both_output_checks(
    enabled, monkeypatch
):
    module = importlib.import_module("tests.test_mrope_fused")
    monkeypatch.setattr(module.cfg, "device", "cpu")
    original_reference = module.mrope_fused_ref
    calls, received = [], []

    def reference(*inputs):
        calls.append("reference")
        received.append(inputs)
        return original_reference(*inputs)

    def candidate(*inputs):
        calls.append("candidate")
        received.append(inputs)
        return original_reference(*inputs)

    monkeypatch.setattr(module, "mrope_fused_ref", reference)
    monkeypatch.setattr(flaggems_sglang, "mrope_fused", candidate)
    with reference_test_mode(enabled) as state:
        module.test_mrope_fused(module.CORRECTNESS_CASES[0])
    assert calls == ["reference", "reference" if enabled else "candidate"]
    assert all(a is b for a, b in zip(received[0], received[1]))
    assert state.candidate_calls == int(enabled)

    def incorrect(*inputs):
        q, k = original_reference(*inputs)
        return q, k + 100

    monkeypatch.setattr(flaggems_sglang, "mrope_fused", incorrect)
    with pytest.raises(AssertionError):
        module.test_mrope_fused(module.CORRECTNESS_CASES[0])
