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

"""Listing must not resolve a candidate; ordinary binding stays unchanged."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark import conftest as bench_conftest
from benchmark import performance_utils as runner
from benchmark.op_benchmark import OpBenchmark

pytest_plugins = ["pytester"]


def test_listing_binding_avoids_lookup_and_execution_still_requires_export(
    monkeypatch,
):
    config = bench_conftest.BenchConfig()
    monkeypatch.setattr(bench_conftest, "Config", config)
    monkeypatch.setattr(runner, "Config", config)
    benchmark = OpBenchmark(
        op_name="missing_candidate",
        torch_op=lambda x: x,
        input_fn=lambda *args: iter(()),
        shapes=[(1,)],
        shape_desc="n",
    )

    class NoLookup:
        def __getattr__(self, name):
            raise AssertionError("listing looked up the candidate")

    config.list_cases = True
    benchmark.set_gems_from_module(NoLookup())
    assert benchmark.gems_op is None
    config.list_cases = False
    with pytest.raises(AttributeError, match="missing_candidate"):
        benchmark.set_gems_from_module(SimpleNamespace())
    candidate = lambda x: x
    benchmark.set_gems_from_module(
        SimpleNamespace(missing_candidate=candidate)
    )
    assert benchmark.gems_op is candidate
    # Unconfigured callers also retain eager binding.
    monkeypatch.setattr(bench_conftest, "Config", None)
    benchmark.set_gems_from_module(
        SimpleNamespace(missing_candidate=candidate)
    )
    assert benchmark.gems_op is candidate


@pytest.mark.parametrize(
    "operator",
    [
        "draft_topk1",
        "fused_moe_router_cudacore",
        "fused_moe_router_tensorcore",
        "gate_up_lora_b",
        "gelu_and_mul",
        "interleaved_rope",
        "moe_fused_gate",
        "moe_fused_mul_sum",
        "per_token_group_quant_int8",
        "per_token_quant_int8",
    ],
)
def test_existing_skip_guard_is_preserved_outside_listing(
    operator, monkeypatch
):
    module = importlib.import_module(f"benchmark.test_{operator}")
    config = bench_conftest.BenchConfig()
    monkeypatch.setattr(bench_conftest, "Config", config)
    monkeypatch.setattr(runner, "Config", config)
    lookups = []
    constructed = []
    ran = []

    def get_op(name):
        lookups.append(name)
        return None

    def construct(**kwargs):
        constructed.append(kwargs)
        return SimpleNamespace(
            set_gems=lambda op: None, run=lambda: ran.append(True)
        )

    monkeypatch.setattr(module.flaggems_sglang, "get_op", get_op)
    monkeypatch.setattr(module, "OpBenchmark", construct)
    test = getattr(module, f"test_perf_{operator}")
    with pytest.raises(pytest.skip.Exception, match="not implemented yet"):
        test()
    assert lookups == [operator]
    assert not constructed and not ran
    config.list_cases = True
    test()
    assert lookups == [operator]
    assert len(constructed) == 1 and ran == [True]


@pytest.mark.parametrize("listing", [True, False])
def test_real_silu_entry_without_public_candidate(pytester, listing):
    root = Path(__file__).resolve().parents[2]
    pytester.makeini(f"[pytest]\npythonpath = . {root / 'src'}\n")
    directory = pytester.path / "benchmark"
    directory.mkdir()
    for path in (root / "benchmark").iterdir():
        if path.is_file() and (
            not path.name.startswith("test_")
            or path.name == "test_silu_and_mul.py"
        ):
            (directory / path.name).symlink_to(path)
    pytester.makeconftest(
        """
        import pytest
        import torch
        import flaggems_sglang

        @pytest.fixture(autouse=True)
        def missing_candidate_and_no_execution(monkeypatch):
            from benchmark.performance_utils import Benchmark
            monkeypatch.delattr(flaggems_sglang, "silu_and_mul", raising=False)
            def forbidden(*args, **kwargs):
                raise AssertionError("constructed inputs or ran a timer")
            for name in ("Generator", "randn", "rand", "zeros", "ones",
                         "empty", "tensor"):
                monkeypatch.setattr(torch, name, forbidden)
            monkeypatch.setattr(Benchmark, "get_latency", forbidden)
            monkeypatch.setattr(Benchmark, "_measure_input", forbidden)
        """
    )
    output = pytester.path / "catalog.json"
    args = ["benchmark/test_silu_and_mul.py", "--level=core", "-q"]
    if listing:
        args += ["--list-cases", f"--output={output}"]
    result = pytester.runpytest_subprocess(*args, timeout=90)
    if listing:
        result.assert_outcomes(passed=1)
        assert result.ret == 0
        cases = json.loads(output.read_text())["benchmarks"][0]["cases"]
        assert len(cases) == len({c["case_id"] for c in cases}) == 5
    else:
        result.assert_outcomes(failed=1)
        assert result.ret != 0
        result.stdout.fnmatch_lines(
            ["*AttributeError*no attribute 'silu_and_mul'*"]
        )
