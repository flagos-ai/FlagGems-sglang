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

"""Correctness-gated, alternating CUDA Graph operator microbenchmarks.

Run from this repository with PYTHONPATH=src. FP4 uses synthetic pages or retained, SHA-pinned C32 inputs through --fixtures.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import sys
import time
from functools import lru_cache
from pathlib import Path

import flaggems_sglang as gems
import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.dsv41_helpers import (  # noqa: E402
    assert_exact,
    fp4_serving_graph_inputs,
)


def cases(full):
    for b in [6, 12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120]:
        for ratio in [1, 2]:
            yield "fp4", (b, 655627 // ratio, ratio)


@lru_cache(maxsize=2)
def captured_fp4(folder, ratio):
    name = "rows48-ratio1-source" if ratio == 1 else "rows48-ratio2-dense"
    path = Path(folder) / (name + ".pt")
    metadata = json.loads((Path(folder) / (name + ".json")).read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("captured fixture SHA mismatch")
    return torch.load(path, map_location="cpu", weights_only=True)


def derived_captured_fp4(folder, shape):
    rows, width, ratio = shape
    x = captured_fp4(folder, ratio)
    assert x["width"] == width and x["q"].shape[0] == 48
    group_requests = x["req"][::6]
    assert torch.equal(x["req"], group_requests.repeat_interleave(6))
    assert torch.equal(group_requests.sort().values, torch.arange(8))
    # Scheduler request slots need not occur in sorted order. Preserve the
    # six-query groups while compacting their RT rows in that same order.
    ordered_mapping = x["req_to_token"][group_requests]
    groups = rows // 6
    copies = (groups + 7) // 8
    table = x["table"].repeat(copies, 1)
    physical_stride = x["table"].shape[0] * x["page_size"] * ratio
    mapping = torch.cat(
        [ordered_mapping + i * physical_stride for i in range(copies)], 0
    )[:groups]
    indices = torch.arange(rows) % 48
    args = [
        x["q"][indices],
        x["weights"][indices],
        mapping,
        torch.arange(groups).repeat_interleave(6),
        x["lens"][indices],
        table,
    ]
    return (*[v.to("cuda") for v in args], x["page_size"], ratio, width)


def prepare(family, shape, fixtures=None):
    xs = (
        derived_captured_fp4(fixtures, shape)
        if fixtures
        else fp4_serving_graph_inputs(shape[0], shape[2])
    )

    def call(group):
        return gems.fp4_index_logits_req_to_token(
            *xs,
            candidate_block_size=8 if shape[2] == 1 else 0,
            write_logits=shape[2] != 1,
            query_group_size=group,
        )

    return lambda: call(1), lambda: call(6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--full", action="store_true")
    p.add_argument(
        "--fixtures",
        help="Optional SHA-pinned historical C32 FP4 fixtures; other row counts are derived",
    )
    p.add_argument("--family", choices=["all", "fp4"], default="all")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--rep-ms", type=int, default=100)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    meta = {
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "versions": {
            n: importlib.metadata.version(n) for n in ["torch", "triton", "tilelang"]
        },
        "input_provenance": "FP4 provenance recorded separately below",
        "protocol": (
            "25 eager warmups; median do_bench_cudagraph; "
            "alternating implementation order; reversed case order in round 2"
        ),
        "rep_ms": a.rep_ms,
        "rounds": a.rounds,
        "warmup_calls": 25,
        "includes": (
            "complete public wrapper captured, including outputs and SplitK/reduction; "
            "JIT and host replay submission excluded"
        ),
        "excludes": "serving, quantization before GEMM, input generation, profiler, kernel load time",
        "baseline": {
            "fp4": "native per-query scoring/candidate chain",
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurements": [],
    }
    if a.fixtures:
        meta["fp4_input_provenance"] = {
            "capture": "historical C32 target-verify B48",
            "other_rows": (
                "cycle complete six-query groups; duplicate complete KV pages "
                "and relocate physical RT slots for distinct groups"
            ),
            "raw_inputs_published": False,
            "fixture_sha256": {
                f"ratio{ratio}": hashlib.sha256(
                    (
                        Path(a.fixtures)
                        / (
                            "rows48-ratio1-source.pt"
                            if ratio == 1
                            else "rows48-ratio2-dense.pt"
                        )
                    ).read_bytes()
                ).hexdigest()
                for ratio in [1, 2]
            },
        }
    else:
        meta["fp4_input_provenance"] = {
            "capture": False,
            "static_full_token_capacity": 655627,
            "visible_full_tokens": 131080,
            "page_size": 64,
            "per_request_KV_pages": "distinct",
        }
    matrix = [
        (family, shape)
        for family, shape in cases(a.full)
        if a.family == "all" or family == a.family
    ]
    for rnd in range(a.rounds):
        order = matrix if rnd % 2 == 0 else list(reversed(matrix))
        for idx, (family, shape) in enumerate(order):
            baseline, candidate = prepare(family, shape, a.fixtures)
            assert_exact(candidate(), baseline())
            for _ in range(25):
                baseline()
                candidate()
            torch.cuda.synchronize()
            values = {}
            funcs = [("baseline", baseline), ("candidate", candidate)]
            if (idx + rnd) % 2:
                funcs.reverse()
            for name, fn in funcs:
                values[name] = 1000 * triton.testing.do_bench_cudagraph(
                    fn, rep=a.rep_ms, return_mode="median"
                )
            row = dict(
                family=family,
                shape=shape,
                round=rnd + 1,
                baseline_us=values["baseline"],
                candidate_us=values["candidate"],
                speedup=values["baseline"] / values["candidate"],
                exact_native=True,
            )
            if family == "fp4":
                row.update(
                    page_size=64,
                    write_logits=shape[2] != 1,
                    candidate_block_size=8 if shape[2] == 1 else 0,
                    static_capacity_full_tokens=655627,
                    fixture_kind=(
                        ("actual C32" if shape[0] == 48 else "derived from C32")
                        if a.fixtures
                        else "synthetic distinct request pages"
                    ),
                )
            meta["measurements"].append(row)
            out.write_text(json.dumps(meta, indent=2))
            print(json.dumps(row), flush=True)
            del baseline, candidate
    summary = {}
    for family in ["fp4"]:
        grouped = {}
        for row in meta["measurements"]:
            if row["family"] == family:
                grouped.setdefault(tuple(row["shape"]), []).append(row)
        rows = []
        for shape, results in grouped.items():
            before = statistics.mean(r["baseline_us"] for r in results)
            after = statistics.mean(r["candidate_us"] for r in results)
            rows.append(
                dict(
                    shape=shape,
                    baseline_us=before,
                    candidate_us=after,
                    speedup=before / after,
                    round_speedups=[r["speedup"] for r in results],
                )
            )
        ratios = [r["speedup"] for r in rows]
        if not ratios:
            continue
        summary[family] = dict(
            cases=len(rows),
            geometric_mean=math.exp(statistics.mean(map(math.log, ratios))),
            min_speedup=min(ratios),
            max_speedup=max(ratios),
            rows=rows,
        )
    meta["summary"] = summary
    meta["completed_epoch"] = time.time()
    out.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
