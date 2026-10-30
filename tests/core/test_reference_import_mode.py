# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit reference-only imports preserve the ordinary registration path."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "mode,expected_calls", [(None, 1), ("0", 1), ("1", 0)]
)
def test_registration_is_skipped_only_when_explicitly_enabled(
    monkeypatch, mode, expected_calls
):
    calls = []

    class Registrar:
        def apply(self, namespace):
            calls.append(1)

        def names(self):
            return []

        def get(self, name):
            return None

    package = ModuleType("flaggems_sglang")
    package.__path__ = []
    package.runtime = SimpleNamespace(
        device=SimpleNamespace(name="cuda", vendor_name="kunlunxin")
    )
    package.testing = SimpleNamespace()
    registrar = ModuleType("flaggems_sglang.runtime.op_registrar")
    registrar.OpRegistrar = Registrar
    monkeypatch.setitem(sys.modules, "flaggems_sglang", package)
    monkeypatch.setitem(
        sys.modules, "flaggems_sglang.runtime.op_registrar", registrar
    )
    if mode is None:
        monkeypatch.delenv("FLAGGEMS_SGLANG_REFERENCE_ONLY", raising=False)
    else:
        monkeypatch.setenv("FLAGGEMS_SGLANG_REFERENCE_ONLY", mode)
    module = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "src/flaggems_sglang/__init__.py"
        )
    )
    assert len(calls) == expected_calls
    assert module["all_registered_ops"]() == []
    assert module["get_op"]("missing") is None
    assert module["vendor_name"] == "kunlunxin"
