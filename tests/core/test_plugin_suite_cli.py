# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Real CLI regression for all eight documented plugin interfaces on CUDA."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
COUNTS = {
    "gemma_rmsnorm_plugin": (13, 12),
    "gemma_fused_add_rmsnorm_plugin": (5, 4),
    "silu_and_mul_out_plugin": (5, 4),
    "select_experts_plugin": (3, 2),
    "_fused_moe_kernel_sequence_plugin": (3, 2),
    "chunk_gated_delta_rule_fwd_plugin": (6, 2),
    "fused_recurrent_gated_delta_rule_packed_decode_plugin": (3, 2),
    "mrotary_embedding_plugin": (7, 6),
}
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA-compatible test device",
)


def cli(tmp_path, name, args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "src"),
            str(ROOT / "verification"),
            env.get("PYTHONPATH", ""),
        ]
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )
    (tmp_path / (name + ".log")).write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-12000:]
    return result.stdout


@pytest.mark.parametrize(
    "mode",
    [
        "correctness-list",
        "benchmark-list",
        "correctness-reference",
        "benchmark-reference",
    ],
)
def test_plugin_cli_modes(mode, tmp_path):
    benchmark = mode.startswith("benchmark")
    paths = [
        ("benchmark" if benchmark else "tests") + "/test_" + op + ".py"
        for op in COUNTS
    ]
    args = paths + ["-q"]
    output = tmp_path / (mode + ".json")
    if mode == "correctness-list":
        args += ["--collect-only"]
    elif mode == "benchmark-list":
        args += ["--list-cases", "--level", "core", "--output", str(output)]
    elif mode == "correctness-reference":
        args += ["--ref-as-candidate"]
    else:
        args += [
            "--ref-as-candidate",
            "--level",
            "core",
            "--metrics",
            "latency",
            "--record",
            "json",
            "--output",
            str(output),
        ]
    log = cli(tmp_path, mode, args)
    if mode == "correctness-list":
        assert "45 tests collected" in log
    elif mode == "correctness-reference":
        assert re.search(r"\b45 passed\b", log) and "skipped" not in log
    elif mode == "benchmark-list":
        report = json.loads(output.read_text())
        groups = report["benchmarks"]
        assert sum(len(group["cases"]) for group in groups) == 34
        assert all(
            any(op in case["case_id"] for op in COUNTS)
            for group in groups
            for case in group["cases"]
        )
    else:
        report = json.loads(output.read_text())
        rows = [
            (operator, row)
            for operator, data in report.items()
            for detail in data.get("details", [])
            for row in detail["result"]
        ]
        assert len(rows) == 34 and {operator for operator, _ in rows} == set(
            COUNTS
        )
        for operator, row in rows:
            assert (
                row["latency"] > 0 and row["candidate_source"] == "reference"
            )
            assert row["error_msg"] is None
            assert row["shape_detail"][
                "baseline_operator"
            ] == operator.removesuffix("_plugin")


@pytest.mark.parametrize("operator", list(COUNTS))
def test_plugin_dummy_reaches_every_correctness_and_benchmark_case(
    operator, tmp_path
):
    log = cli(
        tmp_path,
        operator,
        [
            "tests/test_" + operator + ".py",
            "benchmark/test_" + operator + ".py",
            "--dummy-candidate",
            operator,
            "--level",
            "core",
            "-q",
        ],
    )
    expected = sum(COUNTS[operator])
    assert re.search(
        rf"Dummy candidate: PASSED; {expected} reached;", log
    ), log[-12000:]
