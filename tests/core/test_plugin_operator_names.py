# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Plugin ABI names stay separate from immutable native/data identities."""

import ast
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load(relative, label):
    spec = importlib.util.spec_from_file_location(label, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[label] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cases():
    return load(
        "src/flaggems_sglang/testing/operator_workloads.py", "_plugin_cases"
    )


def test_eight_named_triads_and_immutable_dataset_aliases(cases):
    datasets = json.loads(
        (ROOT / "verification/plugin_datasets.json").read_text()
    )
    assert len(cases.PLUGIN_OPERATORS) == 8
    assert set(cases.PLUGIN_OPERATORS) == set(datasets)
    for plugin, native in cases.PLUGIN_TO_NATIVE.items():
        assert plugin == native + "_plugin"
        for folder, prefix, ext in [
            ("tests", "test_", ".py"),
            ("benchmark", "test_", ".py"),
            ("definitions", "", ".json"),
            ("src/flaggems_sglang/reference", "", ".py"),
        ]:
            assert (ROOT / folder / (prefix + plugin + ext)).is_file()
        definition = json.loads(
            (ROOT / "definitions" / (plugin + ".json")).read_text()
        )
        assert definition["name"] == plugin
        record = datasets[plugin]
        assert (
            record["baseline_operator"] == record["native_operator"] == native
        )
        group = "rope" if native == "mrotary_embedding" else "operators"
        current = (
            ROOT
            / "verification/problems"
            / group
            / (native.lstrip("_") + "_plugin")
        )
        old = ROOT / "verification/legacy" / record["problem"]
        assert old.resolve() == current.resolve() and old.is_dir()
        if group == "operators":
            neutral = (
                ROOT / "verification/legacy/operators" / native.lstrip("_")
            )
            assert neutral.resolve() == current.resolve()
    assert cases.native_operator("other_plugin") == "other_plugin"
    with pytest.raises(ValueError):
        cases.plugin_operator("other_plugin")


def test_native_and_plugin_workload_keys_are_identical(cases):
    total = 0
    for plugin in cases.OPERATORS:
        native = cases.native_operator(plugin)
        assert cases.workloads(plugin, True) == cases.workloads(native, True)
        for row in cases.workloads(plugin):
            original = cases.build_inputs(native, row, torch.bfloat16, "meta")
            renamed = cases.build_inputs(plugin, row, torch.bfloat16, "meta")
            assert cases.baseline_key(plugin, renamed) == cases.baseline_key(
                native, original
            )
            assert cases.parameters(plugin, row) == cases.parameters(
                native, row
            )
            assert cases.describe(plugin, row) == cases.describe(native, row)
            total += 1
    assert total == 28


def test_all_34_keys_match_published_unmodified_nv_records(cases):
    directory = os.environ.get("OPERATOR_BASELINE_DIR")
    if directory is None:
        pytest.skip(
            "set OPERATOR_BASELINE_DIR for paired static-data integration"
        )
    artifact = json.loads(
        (
            Path(directory) / "op_sglang_native_h800_perf_baseline.json"
        ).read_text()
    )
    for plugin in cases.OPERATORS:
        native = cases.native_operator(plugin)
        keys = {
            cases.baseline_key(
                plugin, cases.build_inputs(plugin, row, torch.bfloat16, "meta")
            )
            for row in cases.workloads(plugin)
        }
        assert keys == set(artifact[native]["shapes"])
        assert plugin not in artifact
    rope = load(
        "src/flaggems_sglang/reference/mrotary_embedding_cases.py",
        "_plugin_mrope_cases",
    )
    artifact = json.loads(
        (
            Path(directory) / "op_sglang_mrotary_h800_perf_baseline.json"
        ).read_text()
    )
    keys = {
        rope.baseline_key(rope.build_inputs(row, torch.bfloat16, "meta"))
        for row in rope.WORKLOADS
    }
    assert len(keys) == 6 and keys == set(
        artifact["mrotary_embedding"]["shapes"]
    )


def test_explicit_baseline_group_keeps_exact_key_lookup(capsys):
    attr = load("benchmark/attri_util.py", "_plugin_attributes")
    source = ROOT / "benchmark/performance_utils.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Benchmark"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_compare_against_base"
    )
    namespace = dict(
        lookup_base_record_ex=attr.lookup_base_record_ex,
        lookup_scaling_factor=lambda *args: 2,
        vendor_name="fixture",
        Config=SimpleNamespace(
            base_data={
                "native": {
                    "shapes": {"exact": {"torch.bfloat16": {"latency_ms": 8}}}
                }
            }
        ),
    )
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )
    bench = SimpleNamespace(op_name="native_plugin", baseline_op_name="native")
    metric = SimpleNamespace(
        shape_detail={"baseline_key": "exact"},
        latency=2,
        compared_speedup=None,
    )
    namespace["_compare_against_base"](bench, metric, torch.bfloat16)
    assert metric.compared_speedup == 2
    metric.shape_detail["baseline_key"] = "different"
    metric.compared_speedup = None
    namespace["_compare_against_base"](bench, metric, torch.bfloat16)
    assert metric.compared_speedup is None and metric.base_miss_reason


def test_native_exports_resolve_original_entrypoints(monkeypatch, cases):
    selected, sources = [], []
    common = ModuleType("operator_common")
    common.reference = lambda operator, native: selected.append(
        (operator, native)
    )
    common.native_module = lambda: SimpleNamespace(
        source_info=lambda op: sources.append(op)
    )
    monkeypatch.setitem(sys.modules, "operator_common", common)
    for plugin in cases.OPERATORS:
        path = (
            ROOT
            / "verification/problems/operators"
            / plugin.lstrip("_")
            / "native.py"
        )
        runpy.run_path(str(path))
    assert selected == [(op, True) for op in cases.OPERATORS]
    assert sources == list(cases.NATIVE_OPERATORS)


def test_replay_maps_results_without_rewriting_dataset_or_config(cases):
    replay = load("verification/check_plugins.py", "_plugin_replay")
    config = SimpleNamespace(
        ks3_prefix="unrelated", problems_dir="elsewhere", cache_dir="cache"
    )
    called = []

    def checker(local, *, problems, device, isolate):
        called.append((local.ks3_prefix, problems))
        assert local.problems_dir == str(ROOT / "verification/legacy")
        assert device == "cuda" and isolate is True
        return dict(
            results={
                p: {"status": "PASS", "cases": [{"status": "PASS"}]}
                for p in problems
            },
            summary=dict(total_pass=len(problems), total_fail=0, total_skip=0),
        )

    result = replay.check_plugins(
        cases.PLUGIN_OPERATORS, device="cuda", config=config, checker=checker
    )
    assert len(called) == 2 and set(result["results"]) == set(
        cases.PLUGIN_OPERATORS
    )
    assert result["summary"] == dict(total_pass=8, total_fail=0, total_skip=0)
    assert (
        config.ks3_prefix == "unrelated" and config.problems_dir == "elsewhere"
    )
    for plugin, row in result["results"].items():
        assert row["dataset_problem"] == replay.DATASETS[plugin]["problem"]
        assert row["dataset_prefix"] == replay.DATASETS[plugin]["prefix"]
    with pytest.raises(RuntimeError, match="every selected problem"):
        replay.check_plugins(
            [cases.PLUGIN_OPERATORS[0]],
            config=config,
            checker=lambda *a, **kw: {"results": {}},
        )
    with pytest.raises(ValueError, match="Unknown"):
        replay.check_plugins(
            ["unrelated_plugin"], config=config, checker=checker
        )


def test_readwrite_plugin_names_keep_reset_enabled():
    reset = load(
        "src/flaggems_sglang/testing/reset_timing.py", "_plugin_reset"
    )
    for native, fields in reset.SPEC["read_write"].items():
        values = {name: torch.full((2, 3), 4.0) for name in fields}
        for index in fields.values():
            if index is not None:
                values[index] = torch.tensor([1], dtype=torch.int64)
        factory = reset.reset_factory(native + "_plugin")
        assert callable(factory)
        plan = factory(**values)
        for destination, original in plan.copies:
            destination.zero_()
        plan()
        assert all(
            torch.equal(destination, original)
            for destination, original in plan.copies
        )
    assert reset.reset_factory("gemma_rmsnorm_plugin") is None
    assert reset.reset_factory("other_plugin") is None
