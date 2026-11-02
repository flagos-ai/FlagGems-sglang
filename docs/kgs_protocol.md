# KGS pytest execution evidence

The shared KernelGen Server Gems adapter can execute this suite using the named Catalog `flaggems-sglang`. Definitions, correctness tests, benchmarks and native reference modules must come from the same committed checkout.

The suite exposes three kinds of evidence:

- `--reference-only --level core` emits `flaggems.reference/v2`. Each completed case includes a real `latency_ms` measured around one baseline invocation and device synchronization. This readiness latency includes cold-start overhead and is not the benchmark denominator or an optimization score. Input/dispatch/skip/failure semantics remain those of the reference runner.
- Correctness JSON records contain `candidate_calls`, keyed by `flaggems_sglang.<operator>`. Counts are deltas for the test call, not cumulative session totals. Zero calls remain zero, including when an earlier case invoked the candidate.
- Normal override benchmarks set `candidate_source="override"` only after verifying that the timer invoked the selected override. `--ref-as-candidate` continues to report `candidate_source="reference"`; the ordinary default implementation does not claim override evidence.

Profiling uses the existing `--profile-only --case-id <id>` path and `pytest_flaggems_profile_scope`. Pytest owns input construction, warmup and candidate iterations. KGS provides the profiler context in the same pytest process; without that plugin, profile-only performs a replay without producing a profiler report.

Run the CPU protocol tests with `pytest -q tests/core/test_reference_only.py tests/core/test_dynamic_override.py tests/core/test_kgs_protocol.py`. Some `test_ref_as_candidate.py` integration cases synchronize the selected device even for scalar inputs and require a usable accelerator.
