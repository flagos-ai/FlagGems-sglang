# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Legacy baseline vendor labels retain concrete model matching."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def load_scaling():
    path = Path(__file__).resolve().parents[2] / "benchmark/attri_util.py"
    spec = importlib.util.spec_from_file_location("_test_scaling_alias", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.lookup_scaling_factor


@pytest.mark.parametrize("stored", ["t-head", "thead"])
@pytest.mark.parametrize("runtime", ["t-head", "thead"])
def test_thead_aliases_preserve_concrete_chip_selection(stored, runtime):
    data = {
        "_scaling_factors": {
            "chips": [
                {
                    "vendor": stored,
                    "chip": "PPU-ZW810E",
                    "measured": {"bandwidth_gbps": 0.75},
                },
                {
                    "vendor": stored,
                    "chip": "m890",
                    "measured": {"bandwidth_gbps": 0.25},
                },
            ]
        }
    }
    lookup = load_scaling()
    assert lookup(data, runtime, "mem", torch.bfloat16, "PPU-ZW810E") == 0.75
    assert (
        lookup(data, runtime, "mem", torch.bfloat16, "unknown-model") is None
    )
    assert lookup(data, "metax", "mem", torch.bfloat16, "PPU-ZW810E") is None


def test_explicit_chip_disambiguates_generic_cuda_sdk(monkeypatch):
    monkeypatch.setenv("FLAGGEMS_SGLANG_CHIP", "P800")
    lookup = load_scaling()
    data = {
        "_scaling_factors": {
            "chips": [
                {
                    "vendor": "kunlunxin",
                    "chip": "P900",
                    "measured": {"bandwidth_gbps": 0.8},
                },
                {
                    "vendor": "kunlunxin",
                    "chip": "P800",
                    "measured": {"bandwidth_gbps": 0.4},
                },
            ]
        }
    }
    assert lookup(data, "kunlunxin", "mem", torch.bfloat16) == 0.4


def test_unconfigured_chip_keeps_actual_device_detection(monkeypatch):
    monkeypatch.delenv("FLAGGEMS_SGLANG_CHIP", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "Observed P800")
    lookup = load_scaling()
    assert lookup.__globals__["_SELF_CHIP"] == "Observed P800"


def test_missing_p800_resource_never_uses_p900_factor(monkeypatch):
    monkeypatch.setenv("FLAGGEMS_SGLANG_CHIP", "P800")
    lookup = load_scaling()
    data = {
        "_scaling_factors": {
            "chips": [
                {
                    "vendor": "kunlunxin",
                    "chip": "P900",
                    "measured": {"bandwidth_gbps": 0.8},
                },
                {
                    "vendor": "kunlunxin",
                    "chip": "P800",
                    "measured": {"bandwidth_gbps": None},
                },
            ]
        }
    }
    assert lookup(data, "kunlunxin", "mem", torch.bfloat16) is None
