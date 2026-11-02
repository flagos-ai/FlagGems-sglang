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

"""Tests for the tensor-free workload listing and exact case replay API.

Adapted from upstream FlagGems #6397. Upstream exercises op-specific
benchmark families (BlasBenchmark / UnaryReductionBenchmark / ...), which
this repo does not have; the equivalent coverage here uses the base
Benchmark case API and GenericBenchmark's case_fn/build_inputs_fn provider,
which is the migration path a real sglang op would take.
"""

import pytest
import torch

from benchmark import consts
from benchmark import performance_utils as base
from benchmark.cases import BenchmarkCasePlan

_NODEID = "benchmark/test_case_api.py::test_case_api"


def _case_id(local_id):
    return f"{_NODEID}::{local_id}"


@pytest.fixture(autouse=True)
def _stable_case_identity(monkeypatch):
    # These tests run under the top-level tests/ conftest, which does not
    # configure the benchmark Config singleton (that only happens for a
    # benchmark/ pytest session). Install a fresh BenchConfig so the case
    # API has somewhere to record identity and selection state.
    from benchmark import conftest as bench_conftest

    cfg = bench_conftest.BenchConfig()
    monkeypatch.setattr(bench_conftest, "Config", cfg)
    monkeypatch.setattr(bench_conftest, "CASE_LISTS", [])
    monkeypatch.setattr(bench_conftest, "LISTED_CASE_IDS", set())
    monkeypatch.setattr(base, "Config", cfg)
    cfg.current_nodeid = _NODEID
    cfg.available_case_ids = set()
    cfg.executed_case_ids = set()
    cfg.list_cases = False
    cfg.case_ids = None
    cfg.query = False
    cfg.reference_only = False
    cfg.preflight_only = False
    cfg.profile_only = False
    cfg.bench_level = consts.BenchLevel.CORE
    cfg.user_desired_dtypes = [torch.float16]
    cfg.user_desired_metrics = None


def _make_benchmark(materialized, shapes=None):
    def case_fn(shape, dtype):
        del dtype
        yield BenchmarkCasePlan(
            shape={"input": shape},
            params={"dim": -1},
            builder_args=(shape,),
        )

    def build_inputs_fn(plan, dtype, device):
        materialized.append((plan, dtype, device))
        return plan.builder_args[0], plan.params["dim"]

    benchmark = base.GenericBenchmark(
        op_name="case_api_test",
        case_fn=case_fn,
        build_inputs_fn=build_inputs_fn,
        torch_op=lambda *args: args,
        dtypes=[torch.float16],
    )
    benchmark.shapes = shapes or [(2, 4), (16, 4)]
    benchmark.to_bench_dtypes = [torch.float16]
    return benchmark


def test_list_cases_does_not_materialize_inputs(monkeypatch):
    materialized = []
    benchmark = _make_benchmark(materialized)
    monkeypatch.setattr(benchmark, "init_user_config", lambda: None)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)

    case_list = benchmark.list_cases()

    assert materialized == []
    assert [case.case_id for case in case_list.cases] == [
        _case_id("core::float16::0"),
        _case_id("core::float16::1"),
    ]
    assert [case.shape for case in case_list.cases] == [
        {"input": (2, 4)},
        {"input": (16, 4)},
    ]
    payload = case_list.to_dict()
    assert payload["schema_version"] == "flaggems.benchmark-case-list/v2"
    assert "nodeid" not in payload
    assert payload["cases"][0] == {
        "case_id": _case_id("core::float16::0"),
        "ordinal": 0,
        "dtype": "torch.float16",
        "shape": {"input": (2, 4)},
        "params": {"dim": -1},
    }


def test_selected_case_materializes_exactly_once(monkeypatch):
    materialized = []
    benchmark = _make_benchmark(materialized)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)
    monkeypatch.setattr(
        benchmark,
        "_measure_input",
        lambda input, case_id=None, dtype=None: consts.BenchmarkMetrics(
            case_id=case_id
        ),
    )
    monkeypatch.setattr(
        benchmark, "_emit_result", lambda dtype, metrics: (dtype, metrics)
    )

    results = benchmark._run_cases([_case_id("core::float16::1")])

    assert len(materialized) == 1
    assert materialized[0][0].builder_args == ((16, 4),)
    assert results[0][1][0].case_id == _case_id("core::float16::1")


def test_case_selection_ignores_ids_owned_by_other_nodes(monkeypatch):
    materialized = []
    benchmark = _make_benchmark(materialized)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)

    results = benchmark._run_cases(
        ["benchmark/test_other.py::test_other::core::float16::0"]
    )

    assert results == []
    assert materialized == []
    assert base.Config.available_case_ids == {
        _case_id("core::float16::0"),
        _case_id("core::float16::1"),
    }
    assert base.Config.executed_case_ids == set()


def test_empty_case_selection_runs_nothing(monkeypatch):
    materialized = []
    benchmark = _make_benchmark(materialized)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)

    results = benchmark._run_cases([])

    assert results == []
    assert materialized == []


def test_full_run_preserves_planned_case_order(monkeypatch):
    materialized = []

    def case_fn(shape, dtype):
        del dtype
        for transposed in (False, True):
            yield BenchmarkCasePlan(
                shape={"input": shape},
                params={"transposed": transposed},
                builder_args=(shape, transposed),
            )

    def build_inputs_fn(plan, dtype, device):
        del dtype, device
        materialized.append(plan.builder_args)
        return plan.builder_args

    benchmark = base.GenericBenchmark(
        op_name="generic_case_order_test",
        case_fn=case_fn,
        build_inputs_fn=build_inputs_fn,
        torch_op=lambda *args: args,
        dtypes=[torch.float16],
    )
    benchmark.shapes = [(4, 8), (16, 32)]
    benchmark.to_bench_dtypes = [torch.float16]
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)
    monkeypatch.setattr(
        benchmark,
        "_measure_input",
        lambda input, case_id=None, dtype=None: consts.BenchmarkMetrics(
            case_id=case_id
        ),
    )
    monkeypatch.setattr(
        benchmark, "_emit_result", lambda dtype, metrics: (dtype, metrics)
    )

    results = benchmark._run_cases(None)

    assert materialized == [
        ((4, 8), False),
        ((4, 8), True),
        ((16, 32), False),
        ((16, 32), True),
    ]
    assert [metric.case_id for metric in results[0][1]] == [
        _case_id("core::float16::0"),
        _case_id("core::float16::1"),
        _case_id("core::float16::2"),
        _case_id("core::float16::3"),
    ]


def test_unported_benchmark_keeps_legacy_default_and_rejects_selection(
    monkeypatch,
):
    benchmark = base.GenericBenchmark(
        op_name="legacy_case_api_test",
        input_fn=lambda shape, dtype, device: iter(()),
        torch_op=lambda *args: args,
        dtypes=[torch.float16],
    )
    monkeypatch.setattr(benchmark, "init_user_config", lambda: None)
    monkeypatch.setattr(benchmark, "_run_legacy", lambda: "legacy")

    assert not benchmark.supports_cases()
    assert benchmark.run() == "legacy"
    with pytest.raises(ValueError, match="does not support --case-id"):
        benchmark.run(case_ids=[])


def test_duplicate_generated_ids_fail_during_listing(monkeypatch):
    benchmark = _make_benchmark([])
    monkeypatch.setattr(benchmark, "init_user_config", lambda: None)
    case = next(benchmark.get_case_iter(torch.float16))
    monkeypatch.setattr(
        benchmark, "get_case_iter", lambda dtype: iter([case, case])
    )
    with pytest.raises(ValueError, match="duplicate case IDs"):
        benchmark.list_cases()


def test_empty_provider_fails_without_materializing(monkeypatch):
    materialized = []
    benchmark = _make_benchmark(materialized)
    monkeypatch.setattr(benchmark, "init_user_config", lambda: None)
    monkeypatch.setattr(benchmark, "case_fn", lambda *args: iter(()))

    with pytest.raises(ValueError, match="no benchmark cases"):
        benchmark.list_cases()
    assert materialized == []


def test_catalog_rejects_repeated_ids_across_benchmark_objects():
    from benchmark import conftest

    first = _make_benchmark([])
    second = _make_benchmark([])
    first_payload = first.list_cases(initialize=False).to_dict()
    second_payload = second.list_cases(initialize=False).to_dict()
    conftest.update_case_list(first_payload)

    with pytest.raises(ValueError, match="duplicate case IDs"):
        conftest.update_case_list(second_payload)
    assert conftest.CASE_LISTS == [first_payload]


@pytest.mark.parametrize(
    "initial_status", [pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED]
)
def test_listing_session_with_no_published_cases_fails(initial_status):
    from types import SimpleNamespace

    from benchmark import conftest

    base.Config.list_cases = True
    session = SimpleNamespace(exitstatus=initial_status)
    conftest.pytest_sessionfinish(session, initial_status)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


@pytest.mark.parametrize(
    "kwargs",
    [
        {"case_fn": lambda *args: iter(())},
        {"build_inputs_fn": lambda *args: ()},
        {
            "input_fn": lambda *args: iter(()),
            "case_fn": lambda *args: iter(()),
            "build_inputs_fn": lambda *args: (),
        },
    ],
)
def test_generic_rejects_incomplete_or_mixed_provider(kwargs):
    with pytest.raises(ValueError):
        base.GenericBenchmark(
            op_name="invalid", torch_op=lambda *args: args, **kwargs
        )


def test_listing_run_never_builds_or_measures(monkeypatch):
    benchmark = _make_benchmark([])
    monkeypatch.setattr(benchmark, "init_user_config", lambda: None)
    monkeypatch.setattr(base.Config, "list_cases", True)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)
    monkeypatch.setattr(
        benchmark,
        "build_inputs",
        lambda *args: pytest.fail("input constructed"),
    )
    monkeypatch.setattr(
        benchmark,
        "_measure_input",
        lambda *args, **kwargs: pytest.fail("timing invoked"),
    )

    result = benchmark.run()
    assert len(result.cases) == 2
