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

"""Exercise every migrated provider through the production case runner."""

import importlib
import json
from pathlib import Path

import pytest
import torch

from benchmark import conftest as bench_conftest
from benchmark import performance_utils as runner
from benchmark.cases import BenchmarkCasePlan, single_input_builder
from benchmark.consts import BenchLevel, BenchmarkMetrics
from benchmark.op_benchmark import OpBenchmark

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[2]
OPERATORS = sorted(
    path.stem[5:]
    for path in (ROOT / "benchmark").glob("test_*.py")
    if path.stem != "test_silu_and_mul"
)


def test_single_input_builder_is_lazy_and_preserves_arguments():
    calls = []
    value = (object(), {"nested": [object()]})

    def inputs(shape, dtype, device):
        calls.append((shape, dtype, device))
        yield value

    build = single_input_builder(inputs)
    assert not calls
    plan = BenchmarkCasePlan(shape={"x": (2, 3)}, builder_args=((2, 3),))
    assert build(plan, torch.float32, "cpu") is value
    assert calls == [((2, 3), torch.float32, "cpu")]


@pytest.mark.parametrize("count", [0, 2])
def test_single_input_builder_rejects_wrong_cardinality_and_closes(count):
    closed = []

    def inputs(*args):
        try:
            yield from range(count)
        finally:
            closed.append(True)

    plan = BenchmarkCasePlan(shape={}, builder_args=((2,),))
    with pytest.raises(ValueError, match="no inputs|more than one input"):
        single_input_builder(inputs)(plan, torch.float32, "cpu")
    assert closed == [True]


def test_single_input_builder_preserves_original_errors():
    error = RuntimeError("original input error")

    def inputs(*args):
        raise error
        yield  # pragma: no cover

    with pytest.raises(RuntimeError) as caught:
        single_input_builder(inputs)(
            BenchmarkCasePlan(shape={}, builder_args=((2,),)),
            torch.float32,
            "cpu",
        )
    assert caught.value is error


@pytest.mark.parametrize("batch", [512, 2048])
def test_pool_slice_catalog_reports_the_original_actual_input_shape(batch):
    from benchmark import test_assign_draft_cache_locs_contiguous as module

    shape = (batch, 8, 5)
    (plan,) = module._case_fn(shape, torch.bfloat16)
    inputs = single_input_builder(module._input_fn)(
        plan, torch.bfloat16, "cpu"
    )
    assert plan.params["configuration"] == shape
    assert tuple(inputs[0].shape) == plan.shape["req_pool_indices"]
    assert inputs[0].numel() == module._MAX_BATCH
    # The original generator retains full batch-sized lengths/output slots.
    assert tuple(inputs[2].shape) == plan.shape["seq_lens"] == (batch,)
    assert tuple(inputs[3].shape) == plan.shape["out_cache_loc"]


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("level", [BenchLevel.CORE, BenchLevel.COMPREHENSIVE])
def test_every_provider_lists_original_grid_and_materializes_only_selection(
    operator, level, monkeypatch
):
    module = importlib.import_module(f"benchmark.test_{operator}")
    config = bench_conftest.BenchConfig()
    config.bench_level = level
    monkeypatch.setattr(bench_conftest, "Config", config)
    monkeypatch.setattr(runner, "Config", config)
    built = []
    captured = []

    def inputs(shape, dtype, device, **kwargs):
        value = (shape, dtype, device, kwargs)
        built.append(value)
        yield value

    def capture(**kwargs):
        benchmark = OpBenchmark(**kwargs)
        benchmark.run = lambda: None
        captured.append(benchmark)
        return benchmark

    monkeypatch.setattr(module, "_input_fn", inputs)
    monkeypatch.setattr(module, "OpBenchmark", capture)
    monkeypatch.setattr(
        module.flaggems_sglang, operator, lambda *args: None, raising=False
    )
    monkeypatch.setattr(
        module.flaggems_sglang, "get_op", lambda name: lambda *args: None
    )
    test = getattr(module, f"test_perf_{operator}")
    variants = (
        [False, True] if operator.startswith("chunk_local_cumsum_") else [None]
    )
    for reverse in variants:
        suffix = "" if reverse is None else f"[{reverse}]"
        config.current_nodeid = (
            f"benchmark/test_{operator}.py::test_perf_{operator}{suffix}"
        )
        if reverse is None:
            test()
        else:
            test(reverse)
        benchmark = captured[-1]
        del benchmark.run
        assert benchmark.supports_cases()
        before = len(built)
        state = torch.random.get_rng_state()
        catalog = benchmark.list_cases()
        assert torch.equal(state, torch.random.get_rng_state())
        assert len(built) == before
        expected_shapes = list(module.SHAPES)
        if level == BenchLevel.COMPREHENSIVE:
            expected_shapes = list(
                dict.fromkeys(
                    expected_shapes + getattr(module, "MORE_SHAPES", [])
                )
            )
        expected = [
            (shape, dtype)
            for dtype in benchmark.to_bench_dtypes
            for shape in expected_shapes
        ]
        assert [
            (case.params["configuration"], case.dtype)
            for case in catalog.cases
        ] == expected
        assert all(
            case.builder_args[0].builder_args
            == (case.params["configuration"],)
            for case in catalog.cases
        )
        assert (
            len({case.case_id for case in catalog.cases}) == len(expected) > 0
        )
        if reverse is not None:
            assert all(
                case.params["reverse"] is reverse for case in catalog.cases
            )
        json.dumps(catalog.to_dict())
        measured = []

        def measure(value, case_id=None, dtype=None):
            measured.append((value, case_id, dtype))
            return BenchmarkMetrics(case_id=case_id)

        monkeypatch.setattr(benchmark, "_measure_input", measure)
        monkeypatch.setattr(
            benchmark, "_emit_result", lambda dtype, metrics: metrics
        )
        selected = catalog.cases[len(catalog.cases) // 2]
        benchmark._run_cases([selected.case_id])
        assert len(built) == before + 1
        assert built[-1][:3] == (
            selected.params["configuration"],
            selected.dtype,
            benchmark.device,
        )
        if reverse is not None:
            assert built[-1][3] == {"reverse": reverse}
        assert measured == [(built[-1], selected.case_id, selected.dtype)]
        benchmark._run_cases(["unknown-case"])
        assert len(built) == before + 1
        built.clear()
        measured.clear()
        benchmark.run()
        assert [(value[0], value[1]) for value in built] == expected
        assert [item[1] for item in measured] == [
            case.case_id for case in catalog.cases
        ]


@pytest.mark.parametrize("level", ["core", "comprehensive"])
def test_whole_repository_lists_without_building_or_timing(pytester, level):
    pytester.makeini(f"[pytest]\npythonpath = . {ROOT / 'src'}\n")
    directory = pytester.path / "benchmark"
    directory.mkdir()
    for path in (ROOT / "benchmark").iterdir():
        if path.is_file():
            (directory / path.name).symlink_to(path)
    pytester.makeconftest(
        """
        import pytest
        import torch

        @pytest.fixture(autouse=True)
        def forbid_inputs_and_timing(monkeypatch):
            from benchmark.performance_utils import Benchmark
            def forbidden(*args, **kwargs):
                raise AssertionError("listing constructed inputs or ran a timer")
            for name in ("Generator", "randn", "rand", "randint", "randperm",
                         "zeros", "ones", "empty", "tensor", "arange", "full"):
                monkeypatch.setattr(torch, name, forbidden)
            monkeypatch.setattr(Benchmark, "get_latency", forbidden)
            monkeypatch.setattr(Benchmark, "_measure_input", forbidden)
        """
    )
    output = pytester.path / "all-cases.json"
    result = pytester.runpytest_subprocess(
        "benchmark",
        "--list-cases",
        f"--level={level}",
        f"--output={output}",
        "-q",
        timeout=180,
    )
    assert result.ret == 0
    catalog = json.loads(output.read_text())
    assert {item["op_name"] for item in catalog["benchmarks"]} == set(
        OPERATORS
    ) | {"silu_and_mul"}
    ids = [
        case["case_id"]
        for item in catalog["benchmarks"]
        for case in item["cases"]
    ]
    assert ids and len(ids) == len(set(ids))
    assert all(item["cases"] for item in catalog["benchmarks"])
