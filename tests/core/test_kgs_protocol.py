"""Per-case invocation evidence, including failures and bypasses."""

from types import SimpleNamespace

import pytest

from flaggems_sglang.dynamic_registry import DynamicOpOverride
from tests import conftest


def test_call_count_snapshot_cannot_mutate_registry():
    registry = DynamicOpOverride()
    registry._call_counts["flaggems_sglang.op"] = 2
    snapshot = registry.call_counts()
    snapshot["flaggems_sglang.op"] = 99
    assert registry.call_counts()["flaggems_sglang.op"] == 2


def test_each_correctness_case_records_only_its_own_calls():
    registry = DynamicOpOverride()
    registry._call_counts["flaggems_sglang.op"] = 5
    config = SimpleNamespace(_override_registry=registry)
    for name, increments in [("called", 2), ("bypassed", 0)]:
        conftest.TEST_RESULTS[name] = {}
        hook = conftest.pytest_runtest_call(
            SimpleNamespace(config=config, nodeid=name)
        )
        next(hook)
        registry._call_counts["flaggems_sglang.op"] += increments
        try:
            next(hook)
        except StopIteration:
            pass
        assert conftest.TEST_RESULTS[name]["candidate_calls"] == {
            "flaggems_sglang.op": increments
        }


@pytest.mark.parametrize("invoke", [True, False])
def test_benchmark_override_evidence_requires_actual_invocation(
    monkeypatch, invoke
):
    from contextlib import nullcontext

    from benchmark import performance_utils as base

    registry = DynamicOpOverride()
    registry._call_counts["flaggems_sglang.op"] = 0
    monkeypatch.setattr(
        base, "Config", SimpleNamespace(override_registry=registry)
    )
    monkeypatch.setattr(base, "reference_mode_enabled", lambda: False)
    monkeypatch.setattr(base, "reference_execution", lambda: None)
    bench = base.Benchmark.__new__(base.Benchmark)
    bench.op_name = "op"
    bench.to_bench_metrics = ["latency"]
    bench.unpack_to_args_kwargs = lambda value: ((), {})
    bench.record_shapes = lambda: []

    def candidate():
        registry._call_counts["flaggems_sglang.op"] += 1

    bench._candidate_call = lambda: (candidate, nullcontext(), True)

    def latency(op, *args, **kwargs):
        if invoke:
            op()
        return 0.01

    bench.get_latency = latency
    if invoke:
        assert bench._measure_input(None).candidate_source == "override"
    else:
        with pytest.raises(pytest.fail.Exception, match="did not invoke"):
            bench._measure_input(None)
