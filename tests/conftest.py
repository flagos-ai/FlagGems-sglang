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

import fcntl
import json
import logging
import os
from datetime import datetime

import pytest
import torch  # noqa: F401  (needed: _UNSUPPORTED_DEVICE_CASTS uses torch dtypes)

# TODO(Qiming): Try remove this line
# import torch  # noqa: F401
import yaml

import flaggems_sglang
from flaggems_sglang.cli_override import (
    add_override_arguments,
    cleanup_overrides,
    configure_overrides,
)

BUILTIN_MARKS = {
    "filterwarnings",
    "parametrize",
    "skip",
    "skipif",
    "timeout",
    "tryfirst",
    "trylast",
    "usefixtures",
    "xfail",
}
REGISTERED_MARKS = []
TEST_RESULTS = {}
RUNTEST_INFO = {}
RECORD_LOG = False
RECORD_JSON = False
TO_CPU = False
QUICK_MODE = False

device = flaggems_sglang.device

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_FILE = "accuracy_result.json"

# Some vendor backends abort the whole process (SIGABRT via a C++ CHECK,
# e.g. torch_gcu "gcu_utils.cpp:74 Check failed") when a tensor is cast
# on-device to a dtype their aten_to copy op does not implement. The abort
# kills pytest before it can record anything, so test modules that cast to
# such dtypes on-device must skip up-front. Capability is per vendor AND
# per dtype: each vendor maps to the set of dtypes it cannot cast to
# on-device. Keep entries factual (verified by an isolated-subprocess
# ``Tensor.to(dtype)`` probe) and extend as other vendors are tested.
#
#   enflame (torch_gcu 2.11, verified 2026-11-26):
#     .to(float8_e4m3fn) / .to(float8_e5m2) -> "output data type is not
#     supported" + SIGABRT; .to(uint64) (from any dtype) -> SIGABRT.
#     fp8 tensor *allocation* and CPU-side .to(fp8) still work, so only
#     the on-device cast is listed.
_UNSUPPORTED_DEVICE_CASTS = {
    "enflame": frozenset(
        {
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.uint64,
        }
    ),
}


def skip_if_unsupported_device_casts(*dtypes):
    """Skip the whole module when a needed on-device cast would abort.

    Call at test-module import time (module level, right after imports)
    with every dtype the module's reference/case builders cast a device
    tensor to, e.g.::

        cfg.skip_if_unsupported_device_casts(torch.float8_e4m3fn)

    The module is skipped only when the current vendor's backend is known
    to abort on at least one of the given dtypes; on other vendors (or
    for other dtypes) it is a no-op, so the same test file stays portable
    across backends.
    """
    from flaggems_sglang import vendor_name as _vendor

    unsupported = _UNSUPPORTED_DEVICE_CASTS.get(_vendor, frozenset())
    bad = sorted({t for t in dtypes if t in unsupported}, key=str)
    if bad:
        names = ", ".join(str(t) for t in bad)
        pytest.skip(
            f"device-side cast to {names} aborts the process on "
            f"{_vendor} (backend CHECK failure); module skipped",
            allow_module_level=True,
        )


def cast_skip_mark(*dtypes):
    """A ``pytestmark`` mark skipping every test needing crash-prone casts.

    Same vendor x dtype table as :func:`skip_if_unsupported_device_casts`,
    but returns a pytest mark so each test is recorded as ``skipped`` in
    the json report (a collection-time ``pytest.skip`` leaves the report
    empty, which the aggregator cannot distinguish from a missing run)::

        pytestmark = cfg.cast_skip_mark(torch.float8_e4m3fn)
    """
    from flaggems_sglang import vendor_name as _vendor

    unsupported = _UNSUPPORTED_DEVICE_CASTS.get(_vendor, frozenset())
    bad = sorted({t for t in dtypes if t in unsupported}, key=str)
    if bad:
        names = ", ".join(str(t) for t in bad)
        return pytest.mark.skip(
            reason=f"device-side cast to {names} aborts the process on "
            f"{_vendor} (backend CHECK failure)"
        )
    # No-op: an empty parametrize mark keeps pytestmark a valid list item.
    return pytest.mark.skipif(False, reason="casts supported on this vendor")


def pytest_addoption(parser):
    parser.addoption(
        "--ref",
        action="store",
        default=device,
        required=False,
        choices=[device, "cpu"],
        help="device to run reference tests on",
    )

    parser.addoption(
        "--quick",
        action="store_true",
        help="run tests on quick mode",
    )

    try:
        parser.addoption(
            "--record",
            action="store",
            default="none",
            required=False,
            choices=["none", "log", "json"],
            help="record test results in log/json files or not",
        )
        parser.addoption(
            "--output",
            help="path to the result file",
        )

    except ValueError:
        # Mixed test+benchmark pytest runs may already register --record in
        # benchmark/conftest.py. Reuse the existing option in that case.
        pass

    try:
        parser.addoption(
            "--collect-marks",
            default=None,
            help="Collect the tests with marker information and write to the specified file",
        )
    except ValueError:
        pass

    # Add dynamic operator override options
    add_override_arguments(parser)


def pytest_configure(config):
    global RECORD_LOG
    global RECORD_JSON
    global REPORT_FILE
    global REGISTERED_MARKS
    global RUNTEST_INFO
    global TO_CPU
    global QUICK_MODE

    REGISTERED_MARKS = {
        marker.split(":")[0].strip() for marker in config.getini("markers")
    }

    RECORD_LOG = config.getoption("--record") == "log"
    RECORD_JSON = config.getoption("--record") == "json"
    TO_CPU = config.getoption("--ref") == "cpu"
    QUICK_MODE = config.getoption("--quick") is True

    if RECORD_JSON:
        report_file = config.getoption("--output")
        if report_file:
            REPORT_FILE = report_file

    if RECORD_LOG:
        RUNTEST_INFO = {}
        cmd_args = [
            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]
        logging.basicConfig(
            filename="result_{}.log".format("_".join(cmd_args)).replace(
                "_-", "-"
            ),
            filemode="w",
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )

    # Apply dynamic operator overrides
    configure_overrides(config)


def pytest_unconfigure(config):
    """Cleanup: restore all overridden operators."""
    cleanup_overrides(config)


def pytest_runtest_teardown(item, nextitem):
    if not RECORD_LOG:
        return

    if hasattr(item, "callspec"):
        all_marks = list(item.iter_markers())
        op_marks = [
            mark.name
            for mark in all_marks
            if mark.name not in BUILTIN_MARKS
            and mark.name not in REGISTERED_MARKS
        ]
        if len(op_marks) > 0:
            params = str(item.callspec.params)
            for op_mark in op_marks:
                if op_mark not in RUNTEST_INFO:
                    RUNTEST_INFO[op_mark] = [params]
                else:
                    RUNTEST_INFO[op_mark].append(params)
        else:
            func_name = item.function.__name__
            logging.warning("There is no mark at {}".format(func_name))


def pytest_sessionfinish(session, exitstatus):
    if RECORD_LOG:
        logging.info(json.dumps(RUNTEST_INFO, indent=2))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    TEST_RESULTS[item.nodeid] = {
        "params": None,
        "result": None,
        "opname": None,
    }
    param_values = {}
    request = item._request
    if hasattr(request, "node") and hasattr(request.node, "callspec"):
        param_values = request.node.callspec.params

    TEST_RESULTS[item.nodeid]["params"] = param_values
    # get all mark
    all_marks = [mark.name for mark in item.iter_markers()]
    # exclude marks，such as parametrize、skipif and so on
    operator_marks = [mark for mark in all_marks if mark not in BUILTIN_MARKS]
    TEST_RESULTS[item.nodeid]["opname"] = operator_marks


def get_reason(report):
    if hasattr(report.longrepr, "reprcrash"):
        return report.longrepr.reprcrash.message
    elif isinstance(report.longrepr, tuple):
        return report.longrepr[2]
    else:
        return str(report.longrepr)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    result = TEST_RESULTS.setdefault(
        report.nodeid, {"params": None, "result": None, "opname": None}
    )
    if hasattr(report, "reference_candidate"):
        result.update(report.reference_candidate)
    if report.when == "setup":
        if report.outcome == "skipped":
            reason = get_reason(report)
            result["result"] = "skipped"
            result["reason"] = reason
    elif report.when == "call":
        result["result"] = report.outcome
        if report.outcome in ["skipped", "failed"]:
            reason = get_reason(report)
            result["reason"] = reason
        else:
            result["reason"] = None


def pytest_terminal_summary(terminalreporter):
    if getattr(terminalreporter.config, "_dummy_probe", None) is not None:
        return
    data = TEST_RESULTS
    with open(REPORT_FILE, "a+") as json_file:
        fcntl.flock(json_file, fcntl.LOCK_EX)
        json_file.seek(0)
        content = json_file.read()
        if content:
            existing_data = json.loads(content)
            existing_data.update(TEST_RESULTS)
            data = existing_data
        json_file.seek(0)
        json_file.truncate()
        json.dump(data, json_file, indent=2, default=str)
        json_file.flush()
        os.fsync(json_file.fileno())


def pytest_collection_modifyitems(session, config, items):
    collect_marks_file = config.getoption("--collect-marks")
    if collect_marks_file:
        report = []
        for item in items:
            data = {}

            # Collect some general information
            if item.cls:
                data["class"] = item.cls.__name__
            data["test_case"] = item.name
            if item.originalname:
                data["function"] = item.originalname
            data["file"] = item.location[0]

            all_marks = list(item.iter_markers())
            op_marks = [
                mark.name
                for mark in all_marks
                if mark.name not in BUILTIN_MARKS
                and mark.name not in REGISTERED_MARKS
            ]

            data["marks"] = op_marks
            report.append(data)

        with open(collect_marks_file, "w") as f:
            yaml.dump(report, f, indent=2)

        # Skip all tests
        items.clear()
