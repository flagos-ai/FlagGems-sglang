# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone operator adapters shared by golden export and native collection."""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_CACHE = {}


def load(relative, label):
    if label not in _CACHE:
        spec = importlib.util.spec_from_file_location(label, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[label] = module
        spec.loader.exec_module(module)
        _CACHE[label] = module
    return _CACHE[label]


def cases_module():
    return load(
        "src/flaggems_sglang/testing/qwen36_workloads.py", "_qwen36_cases"
    )


def oracle(operator):
    module = load("src/flaggems_sglang/reference/qwen36.py", "_qwen36_torch")
    return getattr(module, operator)


def native_module():
    return load("verification/qwen36_native.py", "_qwen36_native")


def canonical_output(operator, result):
    if operator == "select_experts":
        # Expert order is unspecified; preserve ID/weight pairing for comparison.
        weights, ids, logits = result
        order = ids.argsort(-1)
        return weights.gather(-1, order), ids.gather(-1, order), logits
    return result


def reference(operator, native=False):
    if native:
        native_impl = native_module()
        fn = native_impl.get_native(operator)
    else:
        # Use the same vendor dispatch as correctness and benchmark tests.
        from flaggems_sglang.reference import get_reference

        fn = get_reference(operator)

    def invoke(**inputs):
        if native:
            inputs = native_impl.prepare_inputs(operator, inputs)
        return canonical_output(operator, fn(**inputs))

    return invoke


def cases(operator, comprehensive=False, device="cuda"):
    module = cases_module()
    return [
        module.build_inputs(operator, row, torch.bfloat16, device)
        for row in module.workloads(operator, comprehensive=comprehensive)
    ]
