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

"""Exercise the real correctness/benchmark conftests in fresh processes."""

import json
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    "source", ["override", "binding", "public", "reference"]
)
def test_candidate_resolution_precedence(monkeypatch, source):
    from types import SimpleNamespace

    import flaggems_sglang
    from benchmark import performance_utils as base
    from flaggems_sglang.dynamic_registry import DynamicOpOverride

    name = "_injection_resolution_probe"
    registry = DynamicOpOverride()
    monkeypatch.setattr(
        base, "Config", SimpleNamespace(override_registry=registry)
    )
    reference = lambda value: value + 1
    public = lambda value: value + 2
    binding = lambda value: value + 3
    override = lambda value: value + 4
    bench = base.Benchmark(name, torch_op=reference)
    monkeypatch.delattr(flaggems_sglang, name, raising=False)
    if source != "reference":
        monkeypatch.setattr(flaggems_sglang, name, public, raising=False)
    if source in {"override", "binding"}:
        bench.set_gems(binding)
    if source == "override":
        registry.override(name, override)
    try:
        candidate, dispatch, injected = bench._candidate_call()
        with dispatch:
            actual = candidate(0)
        assert (
            actual
            == {"override": 4, "binding": 3, "public": 1, "reference": 1}[
                source
            ]
        )
        assert injected == (source == "override")
    finally:
        registry.restore_all()


def test_no_override_preserves_falsey_binding_fallback(monkeypatch):
    from types import SimpleNamespace

    from benchmark import performance_utils as base
    from flaggems_sglang.dynamic_registry import DynamicOpOverride

    class FalseyCandidate:
        def __bool__(self):
            return False

        def __call__(self, value):
            return value + 4

    registry = DynamicOpOverride()
    monkeypatch.setattr(
        base, "Config", SimpleNamespace(override_registry=registry)
    )
    bench = base.Benchmark("_falsey_probe", torch_op=lambda value: value + 1)
    binding = FalseyCandidate()
    bench.set_gems(binding)
    candidate, _, injected = bench._candidate_call()
    assert candidate(0) == 1
    assert not injected

    registry.override(bench.op_name, binding)
    try:
        candidate, _, injected = bench._candidate_call()
        assert candidate is binding
        assert candidate(0) == 4
        assert injected
    finally:
        registry.restore_all()


@pytest.fixture
def injection_project(pytester):
    root = Path(__file__).resolve().parents[2]
    pytester.makeini(f"[pytest]\npythonpath = . {root / 'src'}\n")
    # Reuse the actual hooks and benchmark implementation, not a miniature
    # conftest or a fake registry with a different API from production.
    for name in ("tests", "benchmark"):
        directory = pytester.path / name
        directory.mkdir()
        for path in (root / name).iterdir():
            if not path.is_file() or path.name.startswith("test_"):
                continue
            if name == "tests" and path.name not in {
                "__init__.py",
                "conftest.py",
            }:
                continue
            (directory / path.name).symlink_to(path)

    observed = pytester.path / "calls.jsonl"
    candidate = pytester.makepyfile(
        candidate=f"""
        import json
        from pathlib import Path

        def run(x):
            with Path({str(observed)!r}).open("a") as output:
                output.write(json.dumps(list(x.shape)) + "\\n")
            return x + 7
        """
    )
    return candidate, observed


@pytest.mark.parametrize(
    "mode", ["legacy", "cases", "preflight", "profile", "mixed"]
)
def test_real_pytest_hooks_inject_candidate(pytester, injection_project, mode):
    candidate, observed = injection_project
    providers = """
        input_fn=lambda shape, dtype, device: [(torch.zeros(shape),)],
    """
    if mode in {"cases", "preflight", "profile"}:
        providers = """
        case_fn=lambda shape, dtype: [BenchmarkCasePlan(
            shape={"x": shape}, builder_args=(shape,)
        )],
        build_inputs_fn=lambda plan, dtype, device: (
            torch.zeros(plan.builder_args[0]),
        ),
        """
    benchmark = pytester.path / "benchmark" / "test_injection.py"
    benchmark.write_text(
        "import pytest\n"
        "import torch\n"
        "from benchmark.op_benchmark import OpBenchmark\n"
        "from benchmark.cases import BenchmarkCasePlan\n"
        "\n"
        "def stale_candidate(x):\n"
        "    raise AssertionError('stale candidate was used')\n"
        "\n"
        "@pytest.mark.injection_probe\n"
        "def test_injection():\n"
        "    bench = OpBenchmark(\n"
        "        op_name='injection_probe', torch_op=stale_candidate,\n"
        "        shapes=[(4,)], shape_desc='n', dtypes=[torch.float32],\n"
        + providers
        + "    )\n"
        "    bench.set_gems(stale_candidate)\n"
        "    bench.run()\n"
    )
    paths = ["benchmark/test_injection.py"]
    if mode == "mixed":
        (pytester.path / "tests" / "test_injection.py").write_text(
            "import torch\n"
            "import flaggems_sglang\n"
            "\n"
            "def test_injection():\n"
            "    actual = flaggems_sglang.injection_probe(torch.zeros(4))\n"
            "    torch.testing.assert_close(actual, torch.full((4,), 7.0))\n"
        )
        paths.insert(0, "tests/test_injection.py")

    options = [
        "--override",
        f"injection_probe:{candidate}:run",
        "--level=core",
        "--metrics=latency",
        "--mode=operator",
        "--warmup=0",
        "--iter=1",
        "-q",
    ]
    if mode == "preflight":
        options.append("--preflight-only")
    elif mode == "profile":
        case_id = (
            "benchmark/test_injection.py::test_injection::core::float32::0"
        )
        options += [
            "--profile-only",
            f"--case-id={case_id}",
            "--profile-warmup=0",
            "--profile-iterations=1",
        ]
    result = pytester.runpytest_subprocess(*paths, *options, timeout=90)

    result.assert_outcomes(passed=2 if mode == "mixed" else 1)
    assert result.ret == 0
    calls = [json.loads(line) for line in observed.read_text().splitlines()]
    expected = [[4], [4]] if mode == "mixed" else [[4]]
    assert calls == expected


def test_benchmark_override_config_and_lazy_missing_export(
    pytester, injection_project
):
    candidate, observed = injection_project
    config = pytester.path / "overrides.json"
    config.write_text(
        json.dumps(
            {
                "overrides": {
                    "injection_probe": {
                        "file": str(candidate),
                        "function": "run",
                    }
                }
            }
        )
    )
    (pytester.path / "benchmark" / "test_lazy.py").write_text(
        "import pytest\n"
        "import torch\n"
        "from benchmark.op_benchmark import OpBenchmark\n"
        "\n"
        "@pytest.mark.injection_probe\n"
        "def test_lazy():\n"
        "    bench = OpBenchmark(\n"
        "        op_name='injection_probe', torch_op=lambda x: x,\n"
        "        shapes=[(4,)], shape_desc='n', dtypes=[torch.float32],\n"
        "        input_fn=lambda shape, dtype, device: [(torch.zeros(shape),)],\n"
        "    )\n"
        "    bench.run()\n"
    )
    result = pytester.runpytest_subprocess(
        "benchmark/test_lazy.py",
        f"--override-config={config}",
        "--level=core",
        "--metrics=latency",
        "--mode=operator",
        "--warmup=0",
        "--iter=1",
        "-q",
        timeout=90,
    )
    result.assert_outcomes(passed=1)
    assert result.ret == 0
    assert observed.read_text().splitlines() == ["[4]"]
