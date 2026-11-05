"""Environment vendor selection is validated before automatic detection."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def vendor_from_env():
    # Exercise the production method without initializing Torch/Triton or a device.
    source = (
        Path(__file__).resolve().parents[2]
        / "src/flaggems_sglang/runtime/backend/device.py"
    )
    tree = ast.parse(source.read_text())
    detector = next(
        node for node in tree.body if isinstance(node, ast.ClassDef)
    )
    method = next(
        node
        for node in detector.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_get_vendor_from_env"
    )
    namespace = {"os": os}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )
    return lambda: namespace["_get_vendor_from_env"](
        SimpleNamespace(vendor_list=("nvidia", "ascend"))
    )


def test_explicit_vendor_is_normalized(monkeypatch, vendor_from_env):
    monkeypatch.setenv("GEMS_VENDOR", "ASCEND")
    assert vendor_from_env() == "ascend"


def test_unset_vendor_uses_existing_detection(monkeypatch, vendor_from_env):
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    assert vendor_from_env() is None


def test_invalid_vendor_is_an_error(monkeypatch, vendor_from_env):
    monkeypatch.setenv("GEMS_VENDOR", "typo")
    with pytest.raises(ValueError, match="Unknown GEMS_VENDOR"):
        vendor_from_env()
