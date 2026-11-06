# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The release compatibility payload must run without the tuning package."""

import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
COMPAT = ROOT / "src/flaggems_sglang_plugin/compat/dsv41"
PATCH = (
    ROOT
    / "src/flaggems_sglang_backport/patches/sglang-v0.5.18-dsv41-kernels.patch"
)


def test_baseline_has_no_tuned_or_common_integration_dependency():
    modules = {
        "engram_gate",
        "engram_gather",
        "fp4_pack",
        "grouped_bf16_gemv",
        "grouped_projection",
        "hc_combine_norm",
        "mhc_post_split_h",
        "mhc_post_tilelang",
        "native_fp4",
        "native_mhc",
        "rmsnorm_fp32",
        "rope_fake_quant_fp4",
    }
    for path in (COMPAT / (name + ".py") for name in modules):
        source = path.read_text()
        ast.parse(source)
        assert "flag_gems" not in source, path
        assert "flaggems_sglang.runtime" not in source, path
        assert "SGLANG_" not in source, path
    entry = (ROOT / "src/flaggems_sglang_backport/__init__.py").read_text()
    assert not any(
        isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        for n in ast.parse(entry).body
    )


def test_patch_uses_native_entries_and_keeps_framework_boundary():
    patch = PATCH.read_text()
    assert "compat.dsv41.native_fp4 import" in patch
    assert "compat.dsv41.native_mhc import" in patch
    assert "flag_gems" not in patch
    assert "flaggems_sglang.runtime" not in patch
    paths = re.findall(r"^diff --git a/(.*?) b/", patch, flags=re.M)
    assert len(paths) == len(set(paths)) == 63
    assert all(
        "/kernels/" in path or "/models/deepseek_common/amd/" in path
        for path in paths
    )


def test_every_backport_import_has_an_implementation():
    added = "\n".join(
        line[1:]
        for line in PATCH.read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    for module in re.findall(
        r"from flaggems_sglang_plugin.compat.dsv41.(\w+) import", added
    ):
        assert (COMPAT / (module + ".py")).is_file(), module
