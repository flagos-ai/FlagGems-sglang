# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared benchmark result collector.

Benchmark test files call :func:`record_case` for each benchmarked case;
``benchmark/conftest.py`` reads the accumulated results at session end and
writes one JSON report per operator (format documented in ``docs/guide.md``).

This keeps every per-operator benchmark file a plain pytest module — no
argparse/main/JSON boilerplate duplicated across files.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess

import torch
import triton

import flaggems_sglang

# op_full_name (e.g. "activation_norm/silu_and_mul") -> list of case dicts
_RESULTS: dict[str, list[dict]] = {}


def do_bench_us(fn, warmup=25, rep=100) -> float:
    """Median GPU time in microseconds for one call to ``fn()``.

    Uses ``triton.testing.do_bench`` (CUDA-event timing, warmup, median over
    replays) instead of CPU wall-clock, which is inaccurate for async GPU
    kernels. ``warmup``/``rep`` are in milliseconds. Returns microseconds.
    """
    device = flaggems_sglang.device
    _do_bench = triton.testing.do_bench
    return _do_bench(fn, warmup=warmup, rep=rep, return_mode="median") * 1e3


def record_case(
    op_full_name: str, case_id: str, ref_us: float, triton_us: float
) -> None:
    """Record one benchmarked case for later JSON export.

    Args:
        op_full_name: "<group>/<op_name>", e.g. "activation_norm/silu_and_mul".
        case_id: human-readable case label, e.g. "bs512_d4096".
        ref_us: torch reference latency in microseconds (baseline).
        triton_us: triton kernel latency in microseconds.
    """
    speedup = round(ref_us / triton_us, 3) if triton_us > 0 else None
    _RESULTS.setdefault(op_full_name, []).append(
        {
            "case": case_id,
            "ref_us": round(ref_us, 2),
            "triton_us": round(triton_us, 2),
            "speedup": speedup,
        }
    )


def has_results() -> bool:
    return bool(_RESULTS)


def _machine_info() -> dict:
    info = {
        "node": platform.node(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def _commit_info() -> dict:
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    try:
        commit_id = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return {"id": commit_id, "branch": branch, "dirty": dirty}
    except Exception:
        return {}


def _resolve_output_path(
    rootpath: str, custom: str | None, op_short: str
) -> str:
    """Pick the JSON output path for one operator.

    - ``--bench-output`` ending in ``.json``  -> use it verbatim (single op).
    - ``--bench-output`` as a directory        -> ``<dir>/<op_short>.json``.
    - omitted                                  -> ``<rootpath>/out_benchmark/<op_short>.json``.
    """
    if custom:
        if custom.endswith(".json"):
            return os.path.abspath(custom)
        return os.path.join(os.path.abspath(custom), f"{op_short}.json")
    return os.path.join(str(rootpath), "out_benchmark", f"{op_short}.json")


def write_reports(
    rootpath: str, custom_output: str | None = None
) -> list[str]:
    """Write one JSON report per recorded operator. Returns paths written."""
    if not _RESULTS:
        return []

    device = flaggems_sglang.device
    machine = _machine_info()
    commit = _commit_info()
    written = []

    for op_full_name, cases in _RESULTS.items():
        op_short = op_full_name.split("/")[-1]
        speedups = [c["speedup"] for c in cases if c["speedup"] is not None]
        avg_speedup = (
            round(sum(speedups) / len(speedups), 3) if speedups else None
        )

        report = {
            "machine_info": machine,
            "commit_info": commit,
            "op": op_full_name,
            "device": device,
            "cases": cases,
            "avg_speedup": avg_speedup,
        }

        out_path = _resolve_output_path(rootpath, custom_output, op_short)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        written.append(out_path)

    return written
