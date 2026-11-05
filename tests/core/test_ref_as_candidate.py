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

"""Explicit reference contracts through shared helpers and real pytest hooks."""

import importlib.util
import json
import math
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def contract():
    # The public contract module is intentionally usable without a device
    # runtime. CLI integration below loads the normal package and conftests.
    source = ROOT / "src/flaggems_sglang/reference_candidate.py"
    spec = importlib.util.spec_from_file_location(
        "_sglang_reference_contract_test", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _forbidden(*args, **kwargs):
    raise AssertionError("the original candidate was called")


def test_default_candidate_is_the_current_public_export(contract, monkeypatch):
    class FalseyCandidate:
        def __bool__(self):
            return False

        def __call__(self, value):
            return value + 1

    public = ModuleType("flaggems_sglang")
    original, injected = _forbidden, FalseyCandidate()
    public.reference_probe = original
    monkeypatch.setitem(sys.modules, "flaggems_sglang", public)
    assert contract.get_candidate("reference_probe", reference=sum) is original
    public.reference_probe = injected
    assert (
        contract.get_candidate("reference_probe", reference=None) is injected
    )
    assert contract.select_candidate(injected, sum) is injected
    with contract.reference_test_mode(False):
        assert contract.get_candidate("reference_probe") is injected
    monkeypatch.delattr(public, "reference_probe")
    with pytest.raises(ImportError):
        contract.get_candidate("reference_probe", reference=sum)


def test_reference_candidate_does_not_lookup_the_public_export(
    contract, monkeypatch
):
    public = ModuleType("flaggems_sglang")
    public.__getattr__ = _forbidden
    monkeypatch.setitem(sys.modules, "flaggems_sglang", public)
    value = [1]

    def reference(values, *, increment):
        assert values is value
        values.append(increment)
        return values

    with contract.reference_test_mode(True, "explicit-node") as state:
        candidate = contract.get_candidate(
            "absent_export", reference=reference
        )
        assert candidate(value, increment=2) is value
    assert value == [1, 2]
    assert (state.reference_calls, state.candidate_calls) == (1, 1)


@pytest.mark.parametrize("reference", [None, 42, "sum"])
def test_missing_reference_is_explicit_before_any_candidate_lookup(
    contract, monkeypatch, reference
):
    public = ModuleType("flaggems_sglang")
    public.__getattr__ = _forbidden
    monkeypatch.setitem(sys.modules, "flaggems_sglang", public)
    with contract.reference_test_mode(True, "missing-node") as state:
        with pytest.raises(
            contract.ReferenceContractError, match="No predefined reference"
        ):
            contract.get_candidate("absent_export", reference=reference)
    assert state.candidate_calls == 0
    assert len(state.contract_errors) == 1


def test_omitted_reference_and_nested_mode_restoration(contract, monkeypatch):
    public = ModuleType("flaggems_sglang")
    public.reference_probe = _forbidden
    monkeypatch.setitem(sys.modules, "flaggems_sglang", public)
    with contract.reference_test_mode(True):
        with pytest.raises(
            contract.ReferenceContractError, match="No predefined reference"
        ):
            contract.get_candidate("reference_probe")
        with contract.reference_test_mode(False):
            assert contract.get_candidate("reference_probe") is _forbidden
        assert (
            contract.get_candidate("reference_probe", reference=sum)([1]) == 1
        )
    assert contract.get_candidate("reference_probe") is _forbidden


@pytest.fixture
def reference_project(pytester):
    pytester.makeini(
        f"[pytest]\npythonpath = . {ROOT / 'src'}\n"
        "markers = reference_probe: explicit reference contract checks\n"
    )
    for name in ("tests", "benchmark"):
        directory = pytester.path / name
        directory.mkdir()
        for path in (ROOT / name).iterdir():
            if not path.is_file() or path.name.startswith("test_"):
                continue
            if name == "tests" and path.name not in {
                "__init__.py",
                "conftest.py",
            }:
                continue
            (directory / path.name).symlink_to(path)
    return ROOT


def _accuracy(pytester, source=None):
    path = "tests/test_reference_probe.py"
    if source is None:
        source = """
            import pytest
            from flaggems_sglang.reference_candidate import get_candidate

            def reference(a, b):
                return a + b

            @pytest.mark.reference_probe
            @pytest.mark.parametrize('value', [1, 2])
            def test_candidate(value):
                expected = reference(value, 2)
                candidate = get_candidate(
                    'reference_probe', reference=reference,
                )
                actual = candidate(value, 2)
                assert actual == expected
        """
    (pytester.path / path).write_text(textwrap.dedent(source))
    return path


def _benchmark(
    pytester, *, reference="reference", extra="", outer=False, backward=False
):
    path = "benchmark/test_reference_probe.py"
    parametrization = (
        "@pytest.mark.parametrize('offset', [0, 10], ids=['first', 'second'])\n"
        if outer
        else ""
    )
    argument = "offset" if outer else ""
    offset = "offset" if outer else "0"
    source = f"""
        import json
        from pathlib import Path
        import pytest
        import torch
        import flaggems_sglang
        from benchmark.cases import BenchmarkCasePlan
        from benchmark.op_benchmark import OpBenchmark

        def reference(value):
            with Path('reference-calls.jsonl').open('a') as output:
                output.write(json.dumps(value) + '\\n')
            return value

        def forbidden(*args, **kwargs):
            raise AssertionError('the bound candidate was called')

        @pytest.mark.reference_probe
        {parametrization.rstrip()}
        def test_candidate({argument}):
            {extra or 'pass'}
            bench = OpBenchmark(
                op_name='reference_probe', torch_op={reference},
                shapes=[(1,), (2,)], shape_desc='n',
                dtypes=[torch.float32],
                is_backward={backward!r},
                case_fn=lambda shape, dtype: [BenchmarkCasePlan(
                    shape={{'value': shape}}, params={{'offset': {offset}}},
                    builder_args=(shape[0] + {offset},),
                )],
                build_inputs_fn=lambda plan, dtype, device: (
                    plan.builder_args[0],
                ),
            )
            bench.set_gems_from_module(flaggems_sglang)
            bench.run()
    """
    (pytester.path / path).write_text(textwrap.dedent(source))
    return path


def _run(pytester, *paths, enabled=True, options=(), report="result.json"):
    destination = pytester.path / report
    command = [*paths, "-q", "--record=json", f"--output={destination}"]
    if enabled:
        command.append("--ref-as-candidate")
    command.extend(options)
    result = pytester.runpytest_subprocess(*command, timeout=90)
    data = (
        json.loads(destination.read_text()) if destination.exists() else None
    )
    return result, data


def _rows(data):
    return [
        row
        for detail in data["reference_probe"]["details"]
        for row in detail["result"]
    ]


def _bench_options(*extra):
    return (
        "--level=core",
        "--mode=operator",
        "--warmup=0",
        "--iter=1",
        *extra,
    )


def test_real_accuracy_hooks_record_each_reference_candidate(
    pytester, reference_project
):
    path = _accuracy(pytester)
    result, data = _run(pytester, path, options=("-m", "reference_probe"))
    result.assert_outcomes(passed=2)
    assert result.ret == 0
    assert len(data) == 2
    assert all(row["candidate_source"] == "reference" for row in data.values())
    assert all(row["reference_calls"] == 1 for row in data.values())
    assert all(row["reference_candidate_calls"] == 1 for row in data.values())


def test_normal_execution_and_external_overrides_remain_available(
    pytester, reference_project
):
    path = _accuracy(pytester)
    result, ordinary = _run(
        pytester, path, enabled=False, report="ordinary.json"
    )
    result.assert_outcomes(failed=2)
    assert all("candidate_source" not in row for row in ordinary.values())
    candidate = pytester.makepyfile(candidate="def run(a, b): return a + b\n")
    result, data = _run(
        pytester,
        path,
        enabled=False,
        options=("--override", f"reference_probe:{candidate}:run"),
        report="override.json",
    )
    result.assert_outcomes(passed=2)
    assert result.ret == 0

    dummy = pytester.path / "dummy.json"
    result = pytester.runpytest_subprocess(
        path,
        "--dummy-candidate=reference_probe",
        f"--dummy-report={dummy}",
        "-q",
        timeout=90,
    )
    assert result.ret == 0
    records = json.loads(dummy.read_text())
    assert records["status"] == "PASSED"
    assert len(records["records"]) == 2
    assert all(
        row["status"] == "REACHED"
        and row["calls"] == 1
        and row["unexpected_calls"] == 0
        for row in records["records"]
    )
    assert all(
        row.get("candidate_source") != "reference" for row in data.values()
    )
    config = pytester.path / "override-config.json"
    config.write_text(
        json.dumps(
            {
                "overrides": {
                    "reference_probe": {
                        "file": str(candidate),
                        "function": "run",
                    }
                }
            }
        )
    )
    result, _ = _run(
        pytester,
        path,
        enabled=False,
        options=(f"--override-config={config}",),
        report="configured.json",
    )
    result.assert_outcomes(passed=2)
    assert result.ret == 0


def test_missing_declaration_fails_only_when_enabled_and_selected(
    pytester, reference_project
):
    path = _accuracy(pytester)
    legacy = "tests/test_legacy_reference.py"
    (pytester.path / legacy).write_text("def test_legacy(): pass\n")
    result, _ = _run(pytester, legacy, enabled=False, report="ordinary.json")
    result.assert_outcomes(passed=1)
    result, _ = _run(pytester, path, legacy, report="missing.json")
    assert result.ret != 0
    output = result.stdout.str() + "\n" + result.stderr.str()
    assert "test_legacy_reference.py::test_legacy" in output
    assert "predefined reference" in output.lower()
    result, _ = _run(
        pytester,
        path,
        legacy,
        options=("-m", "reference_probe"),
        report="selected.json",
    )
    result.assert_outcomes(passed=2, deselected=1)
    assert result.ret == 0


@pytest.mark.parametrize(
    "argument", ["", ", reference=None", ", reference=42"]
)
def test_direct_test_requires_a_callable_reference(
    pytester, reference_project, argument
):
    path = _accuracy(
        pytester,
        f"""
        from flaggems_sglang.reference_candidate import get_candidate
        def test_candidate():
            candidate = get_candidate('reference_probe'{argument})
            candidate([1])
        """,
    )
    result, data = _run(pytester, path)
    assert result.ret != 0
    assert data[path + "::test_candidate"]["result"] == "failed"
    assert "predefined reference" in result.stdout.str().lower()


@pytest.mark.parametrize(
    "body",
    [
        "pass",
        "pytest.skip('no declaration')",
        "pytest.xfail('no declaration')",
        "get_candidate('reference_probe', reference=sum)",
    ],
)
def test_another_node_cannot_cover_a_missing_or_unused_candidate(
    pytester, reference_project, body
):
    path = _accuracy(
        pytester,
        f"""
        import pytest
        from flaggems_sglang.reference_candidate import get_candidate
        def test_good():
            assert get_candidate('reference_probe', reference=sum)([1]) == 1
        def test_bad():
            {body}
        """,
    )
    result, data = _run(pytester, path)
    assert result.ret != 0
    assert data[path + "::test_good"]["result"] == "passed"
    bad = data[path + "::test_bad"]
    assert bad["result"] == "failed"
    assert bad["reference_candidate_calls"] == 0
    assert "reference" in result.stdout.str().lower()


def test_fixture_invocation_cannot_replace_the_test_body_call(
    pytester, reference_project
):
    path = _accuracy(
        pytester,
        """
        import pytest
        from flaggems_sglang.reference_candidate import get_candidate
        @pytest.fixture
        def prepared():
            assert get_candidate('reference_probe', reference=sum)([1]) == 1
        def test_candidate(prepared):
            pass
        """,
    )
    result, data = _run(pytester, path)
    assert result.ret != 0
    assert data[path + "::test_candidate"]["reference_candidate_calls"] == 0


def test_direct_test_keeps_its_original_comparison(
    pytester, reference_project
):
    path = _accuracy(
        pytester,
        """
        from flaggems_sglang.reference_candidate import get_candidate
        def test_candidate():
            reference = sum
            expected = reference([1, 2])
            actual = get_candidate('reference_probe', reference=reference)([1, 2])
            assert actual == expected + 1, 'the original comparison executed'
        """,
    )
    result, data = _run(pytester, path)
    assert result.ret != 0
    assert data[path + "::test_candidate"]["reference_candidate_calls"] == 1
    assert "the original comparison executed" in result.stdout.str()


@pytest.mark.parametrize(
    "conversion",
    ["skip", "xfail", "fixture_skip", "teardown_xfail", "marked_xfail"],
)
def test_contract_errors_cannot_be_hidden_by_skip_or_xfail(
    pytester, reference_project, conversion
):
    source = """
        import pytest
        from flaggems_sglang.reference_candidate import (
            ReferenceContractError, get_candidate,
        )

        def test_good():
            assert get_candidate('reference_probe', reference=sum)([1]) == 1
    """
    if conversion == "fixture_skip":
        source += """
        @pytest.fixture
        def bad_case():
            try:
                get_candidate('reference_probe', reference=None)
            except ReferenceContractError:
                pytest.skip('fixture hid its missing reference')

        def test_bad(bad_case):
            pass
        """
    elif conversion == "teardown_xfail":
        source += """
        @pytest.fixture
        def bad_case():
            yield
            try:
                get_candidate('reference_probe', reference=None)
            except ReferenceContractError:
                pytest.xfail('teardown hid its missing reference')

        def test_bad(bad_case):
            assert get_candidate('reference_probe', reference=sum)([1]) == 1
        """
    elif conversion == "marked_xfail":
        source += """
        @pytest.mark.xfail(reason='a missing reference must remain an error')
        def test_bad():
            get_candidate('reference_probe', reference=None)
        """
    else:
        source += f"""
        def test_bad():
            assert get_candidate('reference_probe', reference=sum)([1]) == 1
            try:
                get_candidate('reference_probe', reference=None)
            except ReferenceContractError:
                pytest.{conversion}('test hid its missing reference')
        """
    path = _accuracy(pytester, source)
    result, data = _run(pytester, path)
    assert result.ret != 0
    assert data[path + "::test_good"]["result"] == "passed"
    assert data[path + "::test_bad"]["result"] == "failed"
    output = result.stdout.str().lower()
    assert "referencecontracterror" in output or "reference contract" in output


@pytest.mark.parametrize("selection", ["empty", "all_skip"])
def test_empty_or_all_skipped_reference_validation_cannot_pass(
    pytester, reference_project, selection
):
    source = "value = 1\n"
    if selection == "all_skip":
        source = """
            import pytest
            from flaggems_sglang.reference_candidate import get_candidate
            def test_candidate():
                get_candidate('reference_probe', reference=sum)
                pytest.skip('capability absent')
        """
    path = _accuracy(pytester, source)
    result, _ = _run(pytester, path)
    assert result.ret != 0


def test_declared_capability_skip_is_preserved_alongside_real_calls(
    pytester, reference_project
):
    path = _accuracy(
        pytester,
        """
        import pytest
        from flaggems_sglang.reference_candidate import get_candidate
        @pytest.mark.parametrize('available', [True, False])
        def test_candidate(available):
            candidate = get_candidate('reference_probe', reference=sum)
            if not available:
                pytest.skip('explicit reference exists but input unsupported')
            assert candidate([1]) == 1
        """,
    )
    result, data = _run(pytester, path)
    result.assert_outcomes(passed=1, skipped=1)
    assert result.ret == 0
    assert sum(row["reference_candidate_calls"] for row in data.values()) == 1


def test_benchmark_times_the_explicit_reference_for_every_case(
    pytester, reference_project
):
    path = _benchmark(pytester)
    result, data = _run(pytester, path, options=_bench_options())
    result.assert_outcomes(passed=1)
    assert result.ret == 0
    rows = _rows(data)
    assert len(rows) == 2
    assert len({row["case_id"] for row in rows}) == 2
    assert all(row["candidate_source"] == "reference" for row in rows)
    assert all(
        isinstance(row[name], (int, float))
        and math.isfinite(row[name])
        and row[name] > 0
        for row in rows
        for name in ("latency", "latency_base")
    )
    assert (
        pytester.path / "reference-calls.jsonl"
    ).read_text().splitlines() == [
        "1",
        "1",
        "2",
        "2",
    ]


def test_exact_replay_preserves_outer_parameters(pytester, reference_project):
    path = _benchmark(pytester, outer=True)
    selected = path + "::test_candidate[second]::core::float32::1"
    result, data = _run(
        pytester,
        path,
        options=_bench_options(f"--case-id={selected}"),
    )
    assert result.ret == 0
    assert [row["case_id"] for row in _rows(data)] == [selected]
    assert (
        pytester.path / "reference-calls.jsonl"
    ).read_text().splitlines() == [
        "12",
        "12",
    ]


def test_bare_reference_selection_is_not_evidence_of_benchmark_timing(
    pytester, reference_project
):
    path = "benchmark/test_bare_reference.py"
    (pytester.path / path).write_text(
        textwrap.dedent(
            """
            import pytest
            from flaggems_sglang.reference_candidate import select_candidate

            @pytest.mark.reference_probe
            def test_candidate():
                operation = select_candidate(sum, sum)
                assert operation([1, 2]) == 3
            """
        )
    )
    result, data = _run(pytester, path, options=_bench_options())
    assert result.ret != 0
    assert "timing" in result.stdout.str().lower()
    assert not data.get("reference_probe", {}).get("details")


@pytest.mark.parametrize(
    "problem", ["missing", "metric", "zero_timer", "backward"]
)
def test_benchmark_cannot_claim_reference_timing_without_real_invocation(
    pytester, reference_project, problem
):
    extra = ""
    options = _bench_options()
    if problem == "zero_timer":
        extra = (
            "import benchmark.performance_utils as base; "
            "base.do_bench_latency = lambda *args, **kwargs: 1.0"
        )
        options = ("--level=core", "--mode=kernel", "--warmup=0", "--iter=1")
    elif problem == "metric":
        options = _bench_options("--metrics=latency_base")
    path = _benchmark(
        pytester,
        reference="None" if problem == "missing" else "reference",
        extra=extra,
        backward=problem == "backward",
    )
    result, _ = _run(pytester, path, options=options)
    assert result.ret != 0
    if problem == "missing":
        assert "predefined reference" in result.stdout.str().lower()
    elif problem == "metric":
        assert "candidate latency metric" in result.stdout.str()
    elif problem == "backward":
        output = result.stdout.str().lower()
        assert "backward" in output
        assert "unsupported" in output or "not support" in output
    else:
        assert "candidate" in result.stdout.str().lower()
    if problem != "zero_timer":
        assert not (pytester.path / "reference-calls.jsonl").exists()


@pytest.mark.parametrize(
    "method", ["run", "get_latency", "_measure_input", "_candidate_call"]
)
def test_custom_benchmark_execution_cannot_bypass_reference_contract(
    monkeypatch, method
):
    from benchmark import conftest, performance_utils
    from benchmark.op_benchmark import OpBenchmark
    from flaggems_sglang import reference_candidate as reference

    monkeypatch.setattr(performance_utils, "Config", conftest.BenchConfig())
    custom = type("Custom", (OpBenchmark,), {method: _forbidden})
    with reference.reference_test_mode(True, "custom-node"):
        with pytest.raises(
            (ValueError, reference.ReferenceContractError), match="custom"
        ):
            custom(
                op_name="reference_probe",
                torch_op=sum,
                shapes=[(1,)],
                shape_desc="n",
                input_fn=lambda shape, dtype, device: [(1,)],
            )


@pytest.mark.parametrize(
    "conflict",
    [
        ("--dummy-candidate=reference_probe",),
        ("--override=reference_probe:missing.py:run",),
        ("--override-config=missing.json",),
        ("--reference-only",),
        ("--preflight-only",),
        ("--list-cases",),
        ("--query",),
        ("--parallel=2",),
        ("--collect-only",),
    ],
)
def test_reference_mode_rejects_conflicting_execution_modes(
    pytester, reference_project, conflict
):
    path = _benchmark(pytester)
    result, _ = _run(pytester, path, options=conflict)
    assert result.ret != 0
    assert not (pytester.path / "reference-calls.jsonl").exists()


def test_mixed_session_uses_the_same_mode_for_both_suites(
    pytester, reference_project
):
    accuracy = _accuracy(pytester)
    benchmark = _benchmark(pytester)
    result, data = _run(
        pytester, accuracy, benchmark, options=_bench_options()
    )
    result.assert_outcomes(passed=3)
    assert result.ret == 0
    accuracy_rows = [
        row for key, row in data.items() if key.startswith("tests/")
    ]
    assert len(accuracy_rows) == 2
    assert all(row["reference_candidate_calls"] == 1 for row in accuracy_rows)
    assert all(row["candidate_source"] == "reference" for row in _rows(data))


def test_repeated_pytest_sessions_restore_default_candidate_selection(
    pytester, reference_project
):
    path = _accuracy(pytester)
    script = pytester.makepyfile(
        repeat=f"""
        import sys
        sys.path.insert(0, {str(reference_project / 'src')!r})
        import pytest
        assert pytest.main([{path!r}, '--ref-as-candidate', '-q']) == 0
        assert pytest.main([{path!r}, '-q']) != 0
        """
    )
    result = pytester.run(sys.executable, str(script), timeout=180)
    assert result.ret == 0
