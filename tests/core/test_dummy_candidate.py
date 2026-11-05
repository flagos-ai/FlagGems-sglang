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

"""Check dummy invocation evidence through the real pytest entry points."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture
def dummy_project(pytester):
    root = Path(__file__).resolve().parents[2]
    pytester.makeini(
        f"[pytest]\npythonpath = . {root / 'src'}\n"
        "markers = dummy_probe: selected candidate invocation checks\n"
    )
    # Use the production conftests, registry, input builders and case runner.
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
    return root


def _accuracy(pytester, body):
    path = pytester.path / "tests" / "test_dummy_probe.py"
    path.write_text(textwrap.dedent(body))
    return "tests/test_dummy_probe.py"


def _benchmark(
    pytester,
    *,
    legacy=False,
    build_error=False,
    builder_hits_dummy=None,
    skip_after_run=False,
):
    provider = """
        case_fn=lambda shape, dtype: [BenchmarkCasePlan(
            shape={"x": shape}, builder_args=(shape,)
        )],
        build_inputs_fn=build,
    """
    if legacy:
        provider = "input_fn=lambda shape, dtype, device: [(shape[0],)],"
    builder_probe = ""
    if builder_hits_dummy:
        builder_probe = "    if n == 1:\n        import flaggems_sglang\n"
        if builder_hits_dummy == "unhandled":
            builder_probe += "        flaggems_sglang.dummy_probe(n)\n"
        else:
            builder_probe += (
                "        try:\n"
                "            flaggems_sglang.dummy_probe(n)\n"
                "        except RuntimeError:\n"
            )
            builder_probe += (
                "            pytest.skip('builder converted dummy into skip')\n"
                if builder_hits_dummy == "caught_then_skip"
                else "            pass\n"
            )
    path = pytester.path / "benchmark" / "test_dummy_probe.py"
    path.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "import pytest\n"
        "import torch\n"
        "from benchmark.cases import BenchmarkCasePlan\n"
        "from benchmark.op_benchmark import OpBenchmark\n"
        "\n"
        "def forbidden(*args, **kwargs):\n"
        "    raise AssertionError('dummy must not run a timer or reference')\n"
        "\n"
        "def build(plan, dtype, device):\n"
        "    n = plan.builder_args[0][0]\n"
        "    with Path('built.jsonl').open('a') as output:\n"
        "        output.write(json.dumps(n) + '\\n')\n"
        + builder_probe
        + (
            "    if n == 2: raise ValueError('input construction failed')\n"
            if build_error
            else ""
        )
        + "    return (n,)\n"
        "\n"
        "@pytest.mark.dummy_probe\n"
        "def test_dummy(monkeypatch):\n"
        "    monkeypatch.setattr('benchmark.performance_utils.do_bench_latency', forbidden)\n"
        "    bench = OpBenchmark(\n"
        "        op_name='dummy_probe', torch_op=forbidden,\n"
        "        shapes=[(1,), (2,)], shape_desc='n',\n"
        "        dtypes=[torch.float32],\n"
        + textwrap.indent(textwrap.dedent(provider), "        ")
        + "    )\n"
        "    bench.set_gems(forbidden)\n"
        "    bench.run()\n"
        + (
            "    pytest.skip('after successful runs')\n"
            if skip_after_run
            else ""
        )
    )
    return "benchmark/test_dummy_probe.py"


def _run(pytester, *paths, options=()):
    report = pytester.path / "dummy.json"
    result = pytester.runpytest_subprocess(
        *paths,
        "--dummy-candidate=dummy_probe",
        f"--dummy-report={report}",
        *options,
        "-q",
        timeout=90,
    )
    data = json.loads(report.read_text()) if report.exists() else None
    return result, data


def _passed(result, data):
    assert result.ret == 0
    assert data["schema_version"] == "flaggems.dummy-candidate/v1"
    assert data["status"] == "PASSED"
    assert data["proves"] == "candidate_invocation_only"
    assert data["records"]
    assert all(row["status"] == "REACHED" for row in data["records"])
    assert all(row["calls"] > 0 for row in data["records"])
    assert "Dummy candidate: PASSED;" in result.stdout.str()
    assert "NOT_RUN" not in result.stdout.str()


def test_marker_selection_does_not_substitute_for_per_node_hits(
    pytester, dummy_project
):
    path = _accuracy(
        pytester,
        """
        import pytest
        import flaggems_sglang

        @pytest.mark.dummy_probe
        @pytest.mark.parametrize('value', [0, 1], ids=['hit', 'bypass'])
        def test_selected(value):
            if value == 0:
                flaggems_sglang.dummy_probe(value)

        def test_unselected():
            raise AssertionError('marker selection was ignored')
        """,
    )
    result, data = _run(pytester, path, options=("-m", "dummy_probe"))
    assert result.ret != 0
    assert data["status"] == "FAILED"
    records = {row["nodeid"]: row for row in data["records"]}
    assert len(records) == 2
    assert records[path + "::test_selected[hit]"]["status"] == "REACHED"
    assert records[path + "::test_selected[bypass]"]["status"] == "BYPASSED"


def test_accuracy_records_each_selected_node_without_numerical_verdict(
    pytester, dummy_project
):
    path = _accuracy(
        pytester,
        """
        import pytest
        import flaggems_sglang

        @pytest.mark.parametrize('value', [1, 2])
        def test_candidate(value):
            flaggems_sglang.dummy_probe(value)
            pytest.fail('dummy must stop before numerical comparison')
        """,
    )
    result, data = _run(pytester, path)
    _passed(result, data)
    assert len(data["records"]) == 2
    assert {row["phase"] for row in data["records"]} == {"accuracy"}
    assert len({row["nodeid"] for row in data["records"]}) == 2


@pytest.mark.parametrize(
    "failure", ["reference", "forged", "setup", "teardown"]
)
def test_real_errors_are_never_consumed_as_dummy_success(
    pytester, dummy_project, failure
):
    setup = ""
    call = "flaggems_sglang.dummy_probe(1)"
    if failure == "reference":
        call = "raise ValueError('reference failed before candidate')"
    elif failure == "forged":
        call = "raise RuntimeError('DUMMY_CANDIDATE_REACHED: forged')"
    elif failure == "setup":
        setup = """
@pytest.fixture(autouse=True)
def lifecycle():
    raise ValueError('fixture setup failed')
"""
    else:
        setup = """
@pytest.fixture(autouse=True)
def lifecycle():
    yield
    raise ValueError('fixture teardown failed')
"""
    path = _accuracy(
        pytester,
        "import pytest\nimport flaggems_sglang\n"
        + setup
        + "\ndef test_candidate():\n    "
        + call
        + "\n",
    )
    result, data = _run(pytester, path)
    assert result.ret != 0
    assert data["status"] == "FAILED"
    expected = (
        "PYTEST_LIFECYCLE_ERROR"
        if failure in {"setup", "teardown"}
        else "PRE_CALL_ERROR"
    )
    assert [row["status"] for row in data["records"]] == [expected]


def test_fixture_cannot_hide_a_dummy_call_by_catching_and_skipping(
    pytester, dummy_project
):
    path = _accuracy(
        pytester,
        """
        import pytest
        import flaggems_sglang

        @pytest.fixture(autouse=True)
        def setup_probe(value):
            if value == 0:
                try:
                    flaggems_sglang.dummy_probe(value)
                except RuntimeError:
                    pytest.skip('fixture converted dummy call into skip')

        @pytest.mark.parametrize('value', [0, 1], ids=['first', 'second'])
        def test_candidate(value):
            flaggems_sglang.dummy_probe(value)
        """,
    )
    result, data = _run(pytester, path)
    assert result.ret != 0
    assert data["status"] == "FAILED"
    records = {row["nodeid"]: row for row in data["records"]}
    assert len(records) == 2
    first = records[path + "::test_candidate[first]"]
    assert first["status"] == "PYTEST_LIFECYCLE_ERROR"
    assert first["calls"] == 0
    assert first["before_candidate_calls"] == 1
    second = records[path + "::test_candidate[second]"]
    assert second["status"] == "REACHED"
    assert second["calls"] == 1


@pytest.mark.parametrize("after_hit", ["swallowed", "skip", "other_error"])
def test_a_caught_sentinel_cannot_turn_into_a_successful_probe(
    pytester, dummy_project, after_hit
):
    tail = {
        "swallowed": "pass",
        "skip": "pytest.skip('skip after candidate was called')",
        "other_error": "raise ValueError('unrelated error after hit')",
    }[after_hit]
    path = _accuracy(
        pytester,
        "import pytest\nimport flaggems_sglang\n"
        "\ndef test_candidate():\n"
        "    try:\n"
        "        flaggems_sglang.dummy_probe(1)\n"
        "    except RuntimeError:\n"
        "        " + tail + "\n",
    )
    result, data = _run(pytester, path)
    assert result.ret != 0
    assert data["status"] == "FAILED"
    assert len(data["records"]) == 1
    assert data["records"][0]["calls"] == 1
    assert data["records"][0]["status"] != "REACHED"


@pytest.mark.parametrize("selection", ["all_skip", "empty"])
def test_no_reached_case_is_not_success(pytester, dummy_project, selection):
    path = _accuracy(
        pytester,
        """
        import pytest

        @pytest.mark.dummy_probe
        @pytest.mark.skip(reason='native capability missing')
        def test_unavailable():
            raise AssertionError('skip was ignored')
        """,
    )
    options = ("-m", "missing_marker") if selection == "empty" else ()
    result, data = _run(pytester, path, options=options)
    assert result.ret != 0
    assert data["status"] == (
        "NO_CASES" if selection == "empty" else "ALL_SKIP"
    )
    if selection == "all_skip":
        assert [row["status"] for row in data["records"]] == ["SKIP"]


def test_benchmark_checks_every_case_without_reference_or_timer(
    pytester, dummy_project
):
    path = _benchmark(pytester)
    result, data = _run(
        pytester,
        path,
        options=("--level=core", "--record=json", "--output=result.json"),
    )
    _passed(result, data)
    records = data["records"]
    assert len(records) == 2
    assert {row["phase"] for row in records} == {"benchmark"}
    assert {row["case_id"] for row in records} == {
        path + "::test_dummy::core::float32::0",
        path + "::test_dummy::core::float32::1",
    }
    assert [
        json.loads(line)
        for line in (pytester.path / "built.jsonl").read_text().splitlines()
    ] == [1, 2]
    assert not (pytester.path / "result.json").exists()


def test_benchmark_input_failure_remains_visible_after_an_earlier_hit(
    pytester, dummy_project
):
    path = _benchmark(pytester, build_error=True)
    result, data = _run(pytester, path, options=("--level=core",))
    assert result.ret != 0
    assert data["status"] == "FAILED"
    assert [row["status"] for row in data["records"]] == [
        "REACHED",
        "PRE_CALL_ERROR",
    ]


@pytest.mark.parametrize("skip_after_runs", [False, True])
def test_later_benchmark_in_the_same_item_cannot_erase_a_catalog_error(
    pytester, dummy_project, skip_after_runs
):
    path = "benchmark/test_catalog_error.py"
    source = textwrap.dedent(
        """
            from copy import copy
            import pytest
            import torch
            from benchmark.cases import BenchmarkCasePlan
            from benchmark.op_benchmark import OpBenchmark

            def broken_catalog():
                raise ValueError('the first benchmark catalog failed')

            @pytest.mark.dummy_probe
            def test_two_benchmarks(monkeypatch):
                good = OpBenchmark(
                    op_name='dummy_probe', torch_op=lambda value: value,
                    shapes=[(1,)], shape_desc='n',
                    dtypes=[torch.float32],
                    case_fn=lambda shape, dtype: [BenchmarkCasePlan(
                        shape={'x': shape}, builder_args=(shape[0],)
                    )],
                    build_inputs_fn=lambda plan, dtype, device: (
                        plan.builder_args[0],
                    ),
                )
                bad = copy(good)
                monkeypatch.setattr(bad, '_collect_cases', broken_catalog)
                bad.run()
                good.run()
            """
    )
    if skip_after_runs:
        source += "\n    pytest.skip('after runs')\n"
    (pytester.path / path).write_text(source)
    result, data = _run(pytester, path, options=("--level=core",))
    assert result.ret != 0
    assert data["status"] == "FAILED"
    assert {row["status"] for row in data["records"]} >= {
        "PRE_CALL_ERROR",
        "REACHED",
    }


def test_benchmark_body_cannot_skip_after_reaching_the_candidate(
    pytester, dummy_project
):
    path = _benchmark(pytester, skip_after_run=True)
    result, data = _run(pytester, path, options=("--level=core",))
    assert result.ret != 0
    assert data["status"] == "FAILED"
    assert {row["status"] for row in data["records"]} >= {
        "POST_CALL_ERROR",
        "REACHED",
    }


@pytest.mark.parametrize(
    "handling", ["unhandled", "caught_then_skip", "caught_then_return"]
)
def test_input_builder_calls_do_not_count_as_formal_candidate_invocation(
    pytester, dummy_project, handling
):
    path = _benchmark(pytester, builder_hits_dummy=handling)
    result, data = _run(pytester, path, options=("--level=core",))
    assert result.ret != 0
    assert data["status"] == "FAILED"
    assert len(data["records"]) == 2
    first, second = data["records"]
    assert first["status"] == "PRE_CALL_ERROR"
    assert first["calls"] == 0
    assert first["before_candidate_calls"] == 1
    assert second["status"] == "REACHED"
    assert second["calls"] == 1


def test_benchmark_selection_replays_exactly_one_case_and_rejects_unknown_ids(
    pytester, dummy_project
):
    path = _benchmark(pytester)
    selected = path + "::test_dummy::core::float32::1"
    result, data = _run(
        pytester, path, options=("--level=core", f"--case-id={selected}")
    )
    _passed(result, data)
    assert [row["case_id"] for row in data["records"]] == [selected]
    assert (pytester.path / "built.jsonl").read_text().splitlines() == ["2"]

    result, data = _run(
        pytester, path, options=("--level=core", "--case-id=unknown-case")
    )
    assert result.ret != 0
    assert data["status"] != "PASSED"


def test_case_selection_is_resolved_across_all_outer_pytest_parameters(
    pytester, dummy_project
):
    path = "benchmark/test_outer_dummy.py"
    (pytester.path / path).write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            import pytest
            import torch
            from benchmark.cases import BenchmarkCasePlan
            from benchmark.op_benchmark import OpBenchmark

            def forbidden(*args, **kwargs):
                raise AssertionError('dummy must not execute the reference')

            def build(plan, dtype, device):
                value = plan.builder_args[0]
                with Path('outer-built.txt').open('a') as output:
                    output.write(str(value) + '\\n')
                return (value,)

            @pytest.mark.parametrize('offset', [0, 10], ids=['first', 'second'])
            def test_outer(offset):
                bench = OpBenchmark(
                    op_name='dummy_probe', torch_op=forbidden,
                    shapes=[(1,), (2,)], shape_desc='n',
                    dtypes=[torch.float32],
                    case_fn=lambda shape, dtype: [BenchmarkCasePlan(
                        shape={'x': shape}, params={'offset': offset},
                        builder_args=(shape[0] + offset,)
                    )],
                    build_inputs_fn=build,
                )
                bench.set_gems(forbidden)
                bench.run()
            """
        )
    )
    selected = path + "::test_outer[second]::core::float32::1"
    result, data = _run(
        pytester, path, options=("--level=core", f"--case-id={selected}")
    )
    _passed(result, data)
    assert [row["case_id"] for row in data["records"]] == [selected]
    assert (pytester.path / "outer-built.txt").read_text().splitlines() == [
        "12"
    ]

    # A genuine unknown ID must still fail even when another selected ID hits.
    result, data = _run(
        pytester,
        path,
        options=(
            "--level=core",
            f"--case-id={selected}",
            "--case-id=unknown-case",
        ),
    )
    assert result.ret != 0
    assert data["status"] == "FAILED"
    assert any(row["status"] == "REACHED" for row in data["records"])
    assert any(row["status"] == "NOT_RUN" for row in data["records"])


def test_legacy_benchmark_cannot_claim_per_case_coverage(
    pytester, dummy_project
):
    path = _benchmark(pytester, legacy=True)
    result, data = _run(pytester, path, options=("--level=core",))
    assert result.ret != 0
    assert data["status"] == "FAILED"
    assert "UNSUPPORTED" in {row["status"] for row in data["records"]}


@pytest.mark.parametrize(
    "conflict",
    [
        ("--list-cases",),
        ("--preflight-only",),
        ("--profile-only",),
        ("--reference-only",),
        ("--collect-only",),
        ("--override=dummy_probe:missing.py:run",),
        ("--override-config=missing.json",),
    ],
)
def test_dummy_mode_rejects_conflicting_execution_modes(
    pytester, dummy_project, conflict
):
    path = _benchmark(pytester)
    result, data = _run(pytester, path, options=conflict)
    assert result.ret != 0
    assert data is None or data["status"] != "PASSED"
    assert not (pytester.path / "built.jsonl").exists()


def test_mixed_session_shares_one_probe_and_keeps_both_case_kinds(
    pytester, dummy_project
):
    accuracy = _accuracy(
        pytester,
        """
        import pytest
        import flaggems_sglang

        @pytest.mark.parametrize('value', [1, 2])
        def test_candidate(value):
            flaggems_sglang.dummy_probe(value)
        """,
    )
    benchmark = _benchmark(pytester)
    result, data = _run(
        pytester, accuracy, benchmark, options=("--level=core",)
    )
    _passed(result, data)
    assert len(data["records"]) == 4
    assert len([r for r in data["records"] if r["phase"] == "accuracy"]) == 2
    assert len([r for r in data["records"] if r["phase"] == "benchmark"]) == 2


def test_disabled_probe_preserves_normal_pytest_and_writes_no_probe_report(
    pytester, dummy_project
):
    path = _accuracy(
        pytester,
        """
        import pytest
        import flaggems_sglang

        @pytest.mark.parametrize('value', [1, 2])
        def test_normal(value):
            assert not hasattr(flaggems_sglang, 'dummy_probe')
            assert value + 1 in (2, 3)
        """,
    )
    result = pytester.runpytest_subprocess(path, "-q", timeout=90)
    result.assert_outcomes(passed=2)
    assert result.ret == 0
    assert not (pytester.path / "dummy-candidate-report.json").exists()


def test_repeated_pytest_sessions_restore_candidate_and_disable_probe(
    pytester, dummy_project
):
    probed = _accuracy(
        pytester,
        """
        import flaggems_sglang

        def test_candidate():
            flaggems_sglang.dummy_probe(1)
        """,
    )
    ordinary = pytester.path / "tests" / "test_after_probe.py"
    ordinary.write_text(
        "import flaggems_sglang\n"
        "def test_original_is_restored():\n"
        "    assert not hasattr(flaggems_sglang, 'dummy_probe')\n"
    )
    script = pytester.makepyfile(
        twice=f"""
        import sys
        sys.path.insert(0, {str(dummy_project / 'src')!r})
        import pytest
        import flaggems_sglang

        assert not hasattr(flaggems_sglang, 'dummy_probe')
        assert pytest.main([{probed!r}, '--dummy-candidate=dummy_probe',
                            '--dummy-report=first.json', '-q']) == 0
        assert not hasattr(flaggems_sglang, 'dummy_probe')
        assert pytest.main(['tests/test_after_probe.py', '-q']) == 0
        assert not hasattr(flaggems_sglang, 'dummy_probe')
        """
    )
    result = pytester.run(sys.executable, str(script), timeout=180)
    assert result.ret == 0
    assert (
        json.loads((pytester.path / "first.json").read_text())["status"]
        == "PASSED"
    )
    assert not (pytester.path / "dummy-candidate-report.json").exists()
