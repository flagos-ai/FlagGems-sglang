# Dummy candidate invocation checks

Use the built-in dummy to check that every selected correctness test and every
selected benchmark case reaches the injected operator. A hit proves invocation
only. It does not validate numerical assertions, output contracts or performance.
Run ordinary correctness and timing separately with a real candidate.

```sh
pytest -q tests/test_silu_and_mul.py -m silu_and_mul \
  --dummy-candidate silu_and_mul --dummy-report dummy-correctness.json
pytest -q benchmark/test_silu_and_mul.py -m silu_and_mul --level core \
  --dummy-candidate silu_and_mul --dummy-report dummy-benchmark.json
```

The flag does not select tests: retain the appropriate pytest path/marker.
Both paths use the existing shared override registry. No per-operator dummy is
needed. Ordinary sessions do not load the probe or change candidate resolution.
The initial benchmark example is `silu_and_mul`; benchmark operators need a stable
case provider before they can use this check. Legacy providers report
`UNSUPPORTED` rather than treating one pytest item as full shape coverage.

Correctness retains the original input construction and reference execution. The
dummy accepts any arguments, records a real call and raises a private run-specific
signal. Pytest displays `DUMMY REACHED` for the expected signal. The benchmark
probe builds each selected case's real inputs and uses the normal candidate
resolver, catches the expected signal per case and continues to later shapes.
It never runs a baseline or candidate timer and writes no normal numerical or
latency report. `--case-id` selects an exact benchmark case as usual.

The dedicated JSON report defaults to `dummy-candidate-report.json` and contains
`schema_version: flaggems.dummy-candidate/v1`, `proves: candidate_invocation_only`
and records keyed by correctness `nodeid` or benchmark `case_id`. Statuses include
`REACHED`, `BYPASSED`, `PRE_CALL_ERROR`, `POST_CALL_ERROR`,
`SENTINEL_SWALLOWED`, `PYTEST_LIFECYCLE_ERROR`, `NOT_RUN`, `SKIP` and
`UNSUPPORTED`. Swallowing the signal or converting it to a skip does not pass.
`calls` counts invocations in the candidate execution context. Benchmark input
builders and candidate resolution cannot supply formal hits; pytest fixture calls
are also unexpected. These calls remain errors even if their signal is caught.
`unexpected_calls` and `before_candidate_calls` retain that evidence separately
from candidate hits.
Empty selection, all skips, missing invocations, collection/lifecycle errors and
unknown case IDs return a nonzero exit code. Partial capability skips remain
explicit records; at least one actual hit is required for success.

Dummy mode cannot be combined with external overrides, listing, preflight,
profiling, reference-only execution, collection-only or parallel execution. It
checks the original selected execution grid and does not replace normal testing.
