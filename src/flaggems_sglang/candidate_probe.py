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

"""Opt-in checks that each selected test/case invokes the injected candidate.

This module owns the dummy, observation and report for both correctness and
benchmark tests. It never supplies tensor outputs or measures a latency.
"""

import json
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest


class DummyReached(RuntimeError):
    """Private signal; exception text is not evidence of a hit."""

    def __init__(self, nonce):
        self.nonce = nonce
        super().__init__(f"DUMMY_CANDIDATE_REACHED:{nonce}")


def configure_dummy_candidate(config, registry):
    """Install one probe through the existing shared override lifecycle."""
    operator = getattr(config.option, "dummy_candidate", None)
    output = getattr(config.option, "dummy_report", None)
    if not operator:
        if output:
            raise pytest.UsageError(
                "--dummy-report requires --dummy-candidate."
            )
        return
    if not operator.isidentifier():
        raise pytest.UsageError("--dummy-candidate requires an operator name.")
    incompatible = (
        "override",
        "override_config",
        "list_overrides",
        "list_cases",
        "preflight_only",
        "profile_only",
        "reference_only",
        "query",
        "collectonly",
        "parallel",
        "numprocesses",
    )
    for name in incompatible:
        if getattr(config.option, name, None):
            flag = (
                "--collect-only"
                if name == "collectonly"
                else "--" + name.replace("_", "-")
            )
            raise pytest.UsageError(
                f"--dummy-candidate cannot be combined with {flag}."
            )
    probe = CandidateProbe(
        config, operator, output or "dummy-candidate-report.json"
    )
    if not registry.override(operator, probe.candidate):
        raise pytest.UsageError(
            f"Could not inject dummy candidate for {operator}."
        )
    registry._dummy_probe = probe
    config._dummy_probe = probe
    config.pluginmanager.register(probe, "flaggems-sglang-dummy-candidate")


class CandidateProbe:
    def __init__(self, config, operator, output):
        self.config = config
        self.operator = operator
        self.output = Path(output)
        self.nonce = uuid.uuid4().hex
        self.records = {}
        self.collection_errors = []
        self.active = None
        self.observed = None
        self.observation_stage = "session"
        self.signals = {}
        self.benchmark_nodes = set()
        self.available_case_ids = set()
        self.requested_case_ids = set(
            getattr(config.option, "case_id", None) or []
        )
        self.status = "NOT_RUN"

    def _record(self, nodeid, phase, case_id=None):
        key = (phase, case_id or nodeid)
        if key not in self.records:
            self.records[key] = {
                "operator": self.operator,
                "phase": phase,
                "nodeid": nodeid,
                "status": "NOT_RUN",
                "calls": 0,
                "before_candidate_calls": 0,
                "unexpected_calls": 0,
            }
            if case_id is not None:
                self.records[key]["case_id"] = case_id
        return key, self.records[key]

    def candidate(self, *args, **kwargs):
        signal = DummyReached(self.nonce)
        if self.active is not None:
            self.records[self.active]["calls"] += 1
            self.signals[self.active] = signal
        else:
            key = self.observed
            if key is None:
                key, _ = self._record("<session>", "session")
            elif key not in self.records:
                # Successful benchmark traversal removes its node placeholder.
                # Later calls in that pytest item still need a node record.
                key, _ = self._record(key[1], key[0])
            record = self.records[key]
            record["unexpected_calls"] += 1
            if not record["calls"]:
                record["before_candidate_calls"] += 1
            if not self._failed(record):
                status = (
                    "PYTEST_LIFECYCLE_ERROR"
                    if self.observation_stage in {"setup", "teardown"}
                    else (
                        "POST_CALL_ERROR"
                        if record["calls"]
                        else "PRE_CALL_ERROR"
                    )
                )
                self._error(record, signal, status)
                record["message"] = (
                    "Dummy candidate invoked during "
                    f"{self.observation_stage}, "
                    "outside the formal candidate call."
                )
        raise signal

    @contextmanager
    def _observe(self, key, stage):
        previous = self.observed, self.observation_stage
        self.observed, self.observation_stage = key, stage
        try:
            yield
        finally:
            self.observed, self.observation_stage = previous

    def _is_signal(self, key, exc):
        return (
            self.records[key]["calls"] > 0
            and exc is self.signals.get(key)
            and isinstance(exc, DummyReached)
            and exc.nonce == self.nonce
        )

    @staticmethod
    def _is_placeholder(record):
        return (
            record["status"] == "NOT_RUN"
            and not record["calls"]
            and "message" not in record
            and "error_type" not in record
        )

    @staticmethod
    def _failed(record):
        if record["status"] == "NOT_RUN":
            return "message" in record or "error_type" in record
        return record["status"] not in {"REACHED", "SKIP"}

    @staticmethod
    def _error(record, exc, status):
        if CandidateProbe._failed(record):
            return
        record.update(
            status=status,
            error_type=type(exc).__name__,
            message=str(exc)[:1000],
        )

    def _skip_node(self, record, exc, reason, failure_status):
        # A pytest-level skip is legitimate only before any invocation. Keep
        # earlier failures intact rather than laundering them into skips.
        if self._failed(record):
            return
        had_call = any(
            entry["nodeid"] == record["nodeid"] and entry["calls"]
            for entry in self.records.values()
        )
        if had_call:
            self._error(record, exc, failure_status)
        elif self._is_placeholder(record):
            record.update(status="SKIP", reason=reason)

    @staticmethod
    def _phase(item):
        path = Path(item.nodeid.split("::", 1)[0])
        return "benchmark" if path.parts[0] == "benchmark" else "accuracy"

    def pytest_collection_finish(self, session):
        for item in session.items:
            self._record(item.nodeid, self._phase(item))

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(str(report.longrepr)[-2000:])

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_setup(self, item):
        key, _ = self._record(item.nodeid, self._phase(item))
        with self._observe(key, "setup"):
            yield

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_call(self, item):
        key, _ = self._record(item.nodeid, self._phase(item))
        if self._phase(item) == "accuracy":
            self.active = key
        with self._observe(key, "call"):
            try:
                yield
            finally:
                self.active = None

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_teardown(self, item):
        key, _ = self._record(item.nodeid, self._phase(item))
        with self._observe(key, "teardown"):
            yield

    @pytest.hookimpl(hookwrapper=True, trylast=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        phase = self._phase(item)
        key, record = self._record(item.nodeid, phase)
        is_benchmark = item.nodeid in self.benchmark_nodes
        if call.when != "call":
            if report.failed:
                self._error(
                    record, call.excinfo.value, "PYTEST_LIFECYCLE_ERROR"
                )
            elif report.skipped:
                self._skip_node(
                    record,
                    call.excinfo.value,
                    str(report.longrepr),
                    "PYTEST_LIFECYCLE_ERROR",
                )
            elif is_benchmark and self._is_placeholder(record):
                self.records.pop(key, None)
            return
        if is_benchmark:
            if report.failed:
                self._error(record, call.excinfo.value, "POST_CALL_ERROR")
            elif report.skipped:
                self._skip_node(
                    record,
                    call.excinfo.value,
                    str(report.longrepr),
                    "POST_CALL_ERROR",
                )
            else:
                if self._is_placeholder(record):
                    self.records.pop(key, None)
                report.dummy_status = "dummy_checked"
            return
        if self._failed(record):
            return
        if call.excinfo and self._is_signal(key, call.excinfo.value):
            record.update(status="REACHED", sentinel_observed=True)
            report.outcome = "passed"
            report.longrepr = None
            report.dummy_status = "dummy_reached"
        elif record["calls"]:
            if report.passed or report.skipped:
                record.update(
                    status="SENTINEL_SWALLOWED", sentinel_observed=False
                )
                report.outcome = "failed"
                report.longrepr = (
                    "Dummy candidate signal was swallowed "
                    "or converted to skip."
                )
            else:
                self._error(record, call.excinfo.value, "POST_CALL_ERROR")
        elif report.skipped:
            self._skip_node(
                record,
                call.excinfo.value,
                str(report.longrepr),
                "POST_CALL_ERROR",
            )
        elif call.excinfo:
            self._error(record, call.excinfo.value, "PRE_CALL_ERROR")
        else:
            record.update(status="BYPASSED")
            report.outcome = "failed"
            report.longrepr = (
                "Selected test did not invoke the dummy candidate."
            )
        self.signals.pop(key, None)

    def pytest_report_teststatus(self, report):
        status = getattr(report, "dummy_status", None)
        if status:
            return status, "D", status.replace("_", " ").upper()

    def run_benchmark(self, benchmark, case_ids):
        """Traverse real inputs one case at a time without entering timers."""
        from benchmark.conftest import Config

        key, _ = self._record(Config.current_nodeid, "benchmark")
        with self._observe(key, "benchmark preparation"):
            return self._run_benchmark(benchmark, case_ids)

    def _run_benchmark(self, benchmark, case_ids):
        from benchmark.conftest import Config

        nodeid = Config.current_nodeid
        self.benchmark_nodes.add(nodeid)
        nodekey, node = self._record(nodeid, "benchmark")
        if (
            benchmark.op_name != self.operator
            or not benchmark.supports_cases()
        ):
            node.update(
                status="UNSUPPORTED",
                message=(
                    "Dummy probing requires the selected operator and a "
                    "stable case provider."
                ),
            )
            return []
        try:
            cases = benchmark._collect_cases()
        except Exception as exc:
            self._error(node, exc, "PRE_CALL_ERROR")
            return []
        # One pytest item may run multiple benchmark objects. A later success
        # must not erase an earlier collection/unsupported-provider failure.
        if self._is_placeholder(node):
            self.records.pop(nodekey, None)
        available = {case.case_id for case in cases}
        Config.available_case_ids.update(available)
        self.available_case_ids.update(available)
        selected = available if case_ids is None else set(case_ids)
        if case_ids is not None:
            self.requested_case_ids.update(case_ids)
        executed = []
        for case in cases:
            if case.case_id not in selected:
                continue
            key = ("benchmark", case.case_id)
            if key in self.records:
                self.records[key].update(
                    status="DUPLICATE_CASE",
                    message="Case ID executed more than once.",
                )
                continue
            key, record = self._record(nodeid, "benchmark", case.case_id)
            previous = self.observed, self.observation_stage
            self.observed, self.observation_stage = (
                key,
                "benchmark case preparation",
            )
            try:
                args, kwargs = benchmark.unpack_to_args_kwargs(
                    benchmark.build_inputs(case)
                )
                if self._failed(record):
                    continue
                candidate, dispatch, overridden = benchmark._candidate_call()
                record["override"] = overridden
                if self._failed(record):
                    continue
                with dispatch:
                    if self._failed(record):
                        continue
                    # Only the runner's actual candidate call can prove a hit.
                    # Public-op calls in builders/resolvers are errors.
                    self.active = key
                    try:
                        candidate(*args, **kwargs)
                    finally:
                        self.active = None
            except pytest.skip.Exception as exc:
                if self._failed(record):
                    pass
                elif record["calls"]:
                    record.update(
                        status="SENTINEL_SWALLOWED", sentinel_observed=False
                    )
                else:
                    record.update(status="SKIP", reason=str(exc))
            except DummyReached as exc:
                if self._is_signal(key, exc) and not self._failed(record):
                    record.update(status="REACHED", sentinel_observed=True)
                else:
                    self._error(
                        record,
                        exc,
                        (
                            "POST_CALL_ERROR"
                            if record["calls"]
                            else "PRE_CALL_ERROR"
                        ),
                    )
            except Exception as exc:
                self._error(
                    record,
                    exc,
                    "POST_CALL_ERROR" if record["calls"] else "PRE_CALL_ERROR",
                )
            else:
                if not self._failed(record):
                    record.update(
                        status=(
                            "SENTINEL_SWALLOWED"
                            if record["calls"]
                            else "BYPASSED"
                        ),
                        sentinel_observed=False,
                    )
            finally:
                self.active = None
                self.observed, self.observation_stage = previous
                # Exceptions retain their call frames, including input tensors.
                self.signals.pop(key, None)
            if record["status"] in {"REACHED", "SKIP"}:
                Config.executed_case_ids.add(case.case_id)
                executed.append(case.case_id)
        return executed

    def _summarize(self, exitstatus):
        # A selection may belong to a later parametrized pytest item. Resolve
        # unknown IDs only after all benchmark items have contributed cases.
        unknown = self.requested_case_ids - self.available_case_ids
        for case_id in sorted(unknown):
            _, record = self._record("<unresolved>", "benchmark", case_id)
            record.update(
                status="NOT_RUN", message="Unknown benchmark case ID."
            )
        records = list(self.records.values())
        reached = sum(record["status"] == "REACHED" for record in records)
        acceptable = all(
            record["status"] in {"REACHED", "SKIP"} for record in records
        )
        if not records:
            self.status = "NO_CASES"
        elif all(record["status"] == "SKIP" for record in records):
            self.status = "ALL_SKIP"
        else:
            self.status = "PASSED" if reached and acceptable else "FAILED"
        if self.collection_errors or exitstatus not in (0, 5):
            self.status = "FAILED"
        return records

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session, exitstatus):
        yield
        # Final report includes exit-status changes from other finish hooks.
        records = self._summarize(session.exitstatus)
        if self.status != "PASSED" and session.exitstatus in (0, 5):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            json.dumps(
                {
                    "schema_version": "flaggems.dummy-candidate/v1",
                    "operator": self.operator,
                    "status": self.status,
                    "proves": "candidate_invocation_only",
                    "records": records,
                    "collection_errors": self.collection_errors,
                },
                indent=2,
            )
        )

    def pytest_terminal_summary(self, terminalreporter, exitstatus):
        # TerminalReporter prints while sessionfinish wrappers are unwinding,
        # before our final report is written. Compute its display explicitly.
        self._summarize(exitstatus)
        reached = sum(
            record["status"] == "REACHED" for record in self.records.values()
        )
        terminalreporter.write_sep(
            "-",
            f"Dummy candidate: {self.status}; "
            f"{reached} reached; invocation only",
        )
