"""Environment vendor selection is validated before automatic detection."""

import ast
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def clean_vendor_env(monkeypatch):
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    monkeypatch.delenv("DNN_VENDOR", raising=False)


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
    namespace = {"os": os, "warnings": warnings}
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
    monkeypatch.setenv("DNN_VENDOR", "ascend")
    with pytest.raises(ValueError, match="Unknown GEMS_VENDOR"):
        vendor_from_env()


@pytest.fixture
def runtime_modules(monkeypatch):
    """Load the real backend and detector with simulated vendor/device modules."""
    import importlib.util
    import sys
    import types
    from enum import Enum, auto

    root = Path(__file__).resolve().parents[2] / "src/flaggems_sglang/runtime"
    package_name = "_sglang_vendor_retry"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package_name, package)

    common = types.ModuleType(package_name + ".common")
    vendor_class = next(
        node
        for node in ast.parse((root / "common.py").read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "vendors"
    )
    namespace = {"Enum": Enum, "auto": auto}
    exec(
        compile(
            ast.Module(body=[vendor_class], type_ignores=[]),
            "common.py",
            "exec",
        ),
        namespace,
    )
    common.vendors = namespace["vendors"]
    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setitem(
        sys.modules,
        package_name + ".error",
        types.ModuleType(package_name + ".error"),
    )

    torch = types.ModuleType("torch")

    def fail():
        raise RuntimeError("simulated device query failure")

    torch.cuda = SimpleNamespace(device_count=fail)
    torch.npu = SimpleNamespace(device_count=lambda: 2)
    monkeypatch.setitem(sys.modules, "torch", torch)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    backend = load(package_name + ".backend", root / "backend/__init__.py")
    package.backend = backend
    for vendor, device in (("nvidia", "cuda"), ("ascend", "npu")):
        module = types.ModuleType(backend.__name__ + "._" + vendor)
        module.vendor_info = SimpleNamespace(
            vendor_name=vendor,
            device_name=device,
            dispatch_key=None,
        )
        monkeypatch.setitem(sys.modules, module.__name__, module)
    detector = load(backend.__name__ + ".device", root / "backend/device.py")
    return backend, detector.DeviceDetector, torch


@pytest.mark.parametrize(
    "failure_stage", ["vendor_info", "device_object", "device_count"]
)
@pytest.mark.parametrize("variable", ["GEMS_VENDOR", "DNN_VENDOR"])
def test_failed_initialization_can_switch_vendor(
    monkeypatch, runtime_modules, failure_stage, variable
):
    backend, detector, torch = runtime_modules
    monkeypatch.setenv(variable, "nvidia")
    with monkeypatch.context() as attempt:
        if failure_stage != "device_count":
            method_name = {
                "vendor_info": "get_vendor_info",
                "device_object": "gen_torch_device_object",
            }[failure_stage]
            original = getattr(backend, method_name)

            def fail_after_cache(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("simulated cached initialization failure")

            attempt.setattr(backend, method_name, fail_after_cache)
        warning = (
            pytest.warns(FutureWarning, match="DNN_VENDOR.*GEMS_VENDOR")
            if variable == "DNN_VENDOR"
            else nullcontext()
        )
        with warning, pytest.raises(RuntimeError, match="simulated"):
            detector()

    assert backend.vendor_module is None
    assert backend.device_name is None
    assert backend.torch_device_object is None
    assert detector._instance.__dict__ == {}

    monkeypatch.setenv(variable, "ascend")
    warning = (
        pytest.warns(FutureWarning, match="DNN_VENDOR.*GEMS_VENDOR")
        if variable == "DNN_VENDOR"
        else nullcontext()
    )
    with warning:
        result = detector()
    assert result.vendor_name == "ascend"
    assert result.name == "npu"
    assert result.device_count == 2
    assert backend.vendor_module.vendor_info.vendor_name == "ascend"
    assert backend.device_name == "npu"
    assert backend.torch_device_object is torch.npu
    assert detector() is result


def test_legacy_vendor_warns_and_is_normalized(monkeypatch, vendor_from_env):
    monkeypatch.setenv("DNN_VENDOR", "ASCEND")
    with pytest.warns(FutureWarning, match="DNN_VENDOR.*GEMS_VENDOR"):
        assert vendor_from_env() == "ascend"


def test_new_vendor_takes_precedence(monkeypatch, vendor_from_env):
    monkeypatch.setenv("GEMS_VENDOR", "nvidia")
    monkeypatch.setenv("DNN_VENDOR", "ascend")
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        assert vendor_from_env() == "nvidia"
    assert not observed


def test_invalid_legacy_vendor_is_an_error(monkeypatch, vendor_from_env):
    monkeypatch.setenv("DNN_VENDOR", "typo")
    with pytest.warns(FutureWarning, match="GEMS_VENDOR"):
        with pytest.raises(ValueError, match="Unknown DNN_VENDOR"):
            vendor_from_env()
