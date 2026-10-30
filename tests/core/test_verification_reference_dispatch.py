# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Golden replay uses the same vendor selection as correctness/benchmark."""

import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def helper():
    spec = importlib.util.spec_from_file_location(
        "_test_qwen36_common", ROOT / "verification/qwen36_common.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_dispatch(monkeypatch, get_reference):
    package = ModuleType("flaggems_sglang")
    package.__path__ = []
    package.vendor_name = "test_vendor"
    reference = ModuleType("flaggems_sglang.reference")
    reference.get_reference = get_reference
    monkeypatch.setitem(sys.modules, "flaggems_sglang", package)
    monkeypatch.setitem(sys.modules, "flaggems_sglang.reference", reference)


def test_qwen_replay_uses_selected_vendor_reference(monkeypatch):
    selected = []

    def get_reference(operator):
        selected.append(operator)
        return lambda **inputs: ("vendor-result", inputs)

    install_dispatch(monkeypatch, get_reference)
    result = helper().reference("silu_and_mul_out")(input=7)
    assert selected == ["silu_and_mul_out"]
    assert result == ("vendor-result", {"input": 7})


def test_native_export_keeps_explicit_native_selection(monkeypatch):
    install_dispatch(
        monkeypatch,
        lambda _: (_ for _ in ()).throw(AssertionError("vendor called")),
    )
    module = helper()
    native = SimpleNamespace(
        get_native=lambda _: lambda **inputs: inputs["input"] + 1,
        prepare_inputs=lambda _, inputs: inputs,
    )
    monkeypatch.setattr(module, "native_module", lambda: native)
    assert module.reference("silu_and_mul_out", native=True)(input=7) == 8


def test_mrope_replay_uses_existing_vendor_dispatch(monkeypatch):
    selected = []
    reference = lambda **inputs: inputs

    def get_reference(operator):
        selected.append(operator)
        return reference

    install_dispatch(monkeypatch, get_reference)
    module = runpy.run_path(
        str(
            ROOT
            / "verification/problems/rope/mrotary_embedding/reference_torch.py"
        )
    )
    assert module["reference"] is reference
    assert selected == ["mrotary_embedding"]


def test_mrope_nvidia_replay_keeps_portable_torch_oracle(monkeypatch):
    install_dispatch(
        monkeypatch,
        lambda _: (_ for _ in ()).throw(AssertionError("native selected")),
    )
    sys.modules["flaggems_sglang"].vendor_name = "nvidia"
    reference = lambda **inputs: inputs
    loader = ModuleType("mrotary")
    loader.load = lambda name: SimpleNamespace(reference=reference)
    monkeypatch.setitem(sys.modules, "mrotary", loader)
    module = runpy.run_path(
        str(
            ROOT
            / "verification/problems/rope/mrotary_embedding/reference_torch.py"
        )
    )
    assert module["reference"] is reference
