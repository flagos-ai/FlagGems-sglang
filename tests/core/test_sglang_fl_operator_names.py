# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public baseline names share the plugin ABI; native tensor identities stay fixed."""

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
        "src/flaggems_sglang/testing/operator_workloads.py", "_sglang_fl_cases"
    )


def test_eight_named_triads_and_immutable_dataset_aliases(cases):
    datasets = json.loads(
        (ROOT / "verification/sglang_fl_datasets.json").read_text()
    )
    assert len(cases.SGLANG_FL_OPERATORS) == 8
    assert set(cases.SGLANG_FL_OPERATORS) == set(datasets)
    for plugin, native in cases.SGLANG_FL_TO_NATIVE.items():
        assert plugin == "sglang_fl_" + native.lstrip("_")
        for folder, prefix, ext in [
            ("tests", "test_", ".py"),
            ("benchmark", "test_", ".py"),
            ("definitions", "", ".json"),
            ("src/flaggems_sglang/reference", "", ".py"),
        ]:
            assert (ROOT / folder / (prefix + plugin + ext)).is_file()
        benchmark = ast.parse(
            (ROOT / "benchmark" / ("test_" + plugin + ".py")).read_text()
        )
        calls = [
            node
            for node in ast.walk(benchmark)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "OpBenchmark"
        ]
        assert len(calls) == 1
        assert "baseline_op_name" not in {kw.arg for kw in calls[0].keywords}
        definition = json.loads(
            (ROOT / "definitions" / (plugin + ".json")).read_text()
        )
        assert definition["name"] == plugin
        record = datasets[plugin]
        assert (
            record["baseline_operator"] == plugin
            and record["native_operator"] == native
        )
        group = "rope" if native == "mrotary_embedding" else "operators"
        current = (
            ROOT
            / "verification/problems"
            / group
            / ("sglang_fl_" + native.lstrip("_"))
        )
        old = ROOT / "verification/legacy" / record["problem"]
        assert old.resolve() == current.resolve() and old.is_dir()
        if group == "operators":
            neutral = (
                ROOT / "verification/legacy/operators" / native.lstrip("_")
            )
            assert neutral.resolve() == current.resolve()
    assert cases.native_operator("sglang_fl_other") == "sglang_fl_other"
    with pytest.raises(ValueError):
        cases.suite_operator("sglang_fl_other")


def test_native_and_prefixed_workload_keys_are_identical(cases):
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


def test_all_34_keys_match_published_nv_records(cases):
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
    attr = load("benchmark/attri_util.py", "_sglang_fl_baseline_lookup")
    for plugin in cases.OPERATORS:
        keys = {
            cases.baseline_key(
                plugin, cases.build_inputs(plugin, row, torch.bfloat16, "meta")
            )
            for row in cases.workloads(plugin)
        }
        assert keys == set(artifact[plugin]["shapes"])
        assert cases.native_operator(plugin) not in artifact
        for key in keys:
            record, reason = attr.lookup_base_record_ex(
                artifact, plugin, torch.bfloat16, {"baseline_key": key}
            )
            assert reason is None and record["latency_ms"] > 0
    rope = load(
        "src/flaggems_sglang/reference/mrotary_embedding_cases.py",
        "_sglang_fl_mrope_cases",
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
        artifact["sglang_fl_mrotary_embedding"]["shapes"]
    )
    for key in keys:
        record, reason = attr.lookup_base_record_ex(
            artifact,
            "sglang_fl_mrotary_embedding",
            torch.bfloat16,
            {"baseline_key": key},
        )
        assert reason is None and record["latency_ms"] > 0


def test_explicit_baseline_group_keeps_exact_key_lookup(capsys):
    attr = load("benchmark/attri_util.py", "_sglang_fl_attributes")
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
    bench = SimpleNamespace(
        op_name="sglang_fl_native", baseline_op_name="native"
    )
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
    replay = load("verification/check_sglang_fl.py", "_sglang_fl_replay")
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
                p: {
                    "status": "PASS",
                    "cases": [{"status": "PASS"}]
                    * next(
                        d["cases"]
                        for d in replay.DATASETS.values()
                        if d["problem"] == p
                    ),
                }
                for p in problems
            },
            summary=dict(
                total_pass=sum(
                    next(
                        d["cases"]
                        for d in replay.DATASETS.values()
                        if d["problem"] == p
                    )
                    for p in problems
                ),
                total_fail=0,
                total_skip=0,
            ),
        )

    result = replay.check_sglang_fl(
        cases.SGLANG_FL_OPERATORS,
        device="cuda",
        config=config,
        checker=checker,
    )
    assert len(called) == 2 and set(result["results"]) == set(
        cases.SGLANG_FL_OPERATORS
    )
    assert result["summary"] == dict(total_pass=35, total_fail=0, total_skip=0)
    assert (
        config.ks3_prefix == "unrelated" and config.problems_dir == "elsewhere"
    )
    for plugin, row in result["results"].items():
        assert row["dataset_problem"] == replay.DATASETS[plugin]["problem"]
        assert row["dataset_prefix"] == replay.DATASETS[plugin]["prefix"]
    with pytest.raises(RuntimeError, match="every selected problem"):
        replay.check_sglang_fl(
            [cases.SGLANG_FL_OPERATORS[0]],
            config=config,
            checker=lambda *a, **kw: {"results": {}},
        )
    with pytest.raises(RuntimeError, match="fixed dataset cases"):
        replay.check_sglang_fl(
            [cases.SGLANG_FL_OPERATORS[0]],
            config=config,
            checker=lambda *a, **kw: {
                "results": {
                    replay.DATASETS[cases.SGLANG_FL_OPERATORS[0]]["problem"]: {
                        "cases": []
                    }
                }
            },
        )
    with pytest.raises(ValueError, match="Unknown"):
        replay.check_sglang_fl(
            ["sglang_fl_unrelated"], config=config, checker=checker
        )


def test_readwrite_prefixed_names_keep_reset_enabled():
    reset = load(
        "src/flaggems_sglang/testing/reset_timing.py", "_sglang_fl_reset"
    )
    for native, fields in reset.SPEC["read_write"].items():
        values = {name: torch.full((2, 3), 4.0) for name in fields}
        for index in fields.values():
            if index is not None:
                values[index] = torch.tensor([1], dtype=torch.int64)
        factory = reset.reset_factory("sglang_fl_" + native.lstrip("_"))
        assert callable(factory)
        plan = factory(**values)
        for destination, original in plan.copies:
            destination.zero_()
        plan()
        assert all(
            torch.equal(destination, original)
            for destination, original in plan.copies
        )
    assert reset.reset_factory("sglang_fl_gemma_rmsnorm") is None
    assert reset.reset_factory("sglang_fl_other") is None


def test_isolated_workers_inherit_dataset_and_current_sources(monkeypatch):
    import subprocess

    replay = load(
        "verification/check_sglang_fl.py", "_sglang_fl_worker_environment"
    )
    monkeypatch.setenv("OP_VERIFY_KS3_PREFIX", "unrelated-prefix")
    monkeypatch.setenv("OP_VERIFY_PROBLEMS_DIR", "old-checkout")
    monkeypatch.setenv("OP_VERIFY_CACHE_DIR", "old-cache")
    monkeypatch.setenv("PYTHONPATH", "vendor-sdk")
    config = SimpleNamespace(
        ks3_prefix="unrelated-prefix",
        problems_dir="old-checkout",
        cache_dir="cache",
    )
    operator = "sglang_fl_gemma_fused_add_rmsnorm"
    dataset = replay.DATASETS[operator]

    def checker(local, **kwargs):
        paths = [str(ROOT / "src"), str(ROOT / "verification")]
        code = (
            "import os;"
            f"assert os.environ['OP_VERIFY_KS3_PREFIX']=={dataset['prefix']!r};"
            f"assert os.environ['OP_VERIFY_PROBLEMS_DIR']=={str(ROOT / 'verification/legacy')!r};"
            f"assert os.environ['OP_VERIFY_CACHE_DIR']=={str(Path('cache') / dataset['cache_group'])!r};"
            f"assert os.environ['PYTHONPATH'].split(os.pathsep)[:2]=={paths!r}"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
        raise RuntimeError("fixture worker failure")

    with pytest.raises(RuntimeError, match="fixture worker failure"):
        replay.check_sglang_fl([operator], config=config, checker=checker)
    assert os.environ["OP_VERIFY_KS3_PREFIX"] == "unrelated-prefix"
    assert os.environ["OP_VERIFY_PROBLEMS_DIR"] == "old-checkout"
    assert os.environ["OP_VERIFY_CACHE_DIR"] == "old-cache"
    assert os.environ["PYTHONPATH"] == "vendor-sdk"
