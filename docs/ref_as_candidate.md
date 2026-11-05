# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Reference as candidate

`--ref-as-candidate` checks that a selected correctness test or benchmark can
execute with its explicitly declared reference on the candidate side. It is off
by default. Existing operator tests, external overrides and dummy checks retain
their usual commands and behavior.

```sh
pytest -q tests/test_silu_and_mul.py -m silu_and_mul --ref-as-candidate \
  --record json --output silu-reference-correctness.json
pytest -q benchmark/test_silu_and_mul.py -m silu_and_mul --level core \
  --ref-as-candidate --record json --output silu-reference-timing.json
```

## Declare a correctness reference

All 125 operator correctness tests use ordinary pytest functions and explicitly
supply their existing reference to `get_candidate`. Input generation, the
expected-side reference call, comparisons and test-owned copies remain in the
original test body. For example:

```python
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate


def test_example():
    reference = get_reference("example")
    candidate = get_candidate("example", reference=reference)
    inputs = build_inputs()
    expected = reference(**inputs)
    actual = candidate(**inputs)
    check(actual, expected)
```

Without the flag, `get_candidate` returns the operator's current public export,
including external overrides and dummy injection. With the flag it returns the
supplied reference, using the same selection mechanism as benchmarks. The actual
candidate export is never looked up in this mode, so it may be absent.

The reference and candidate must accept the same arguments. Supply a small
reference adapter explicitly when their original signatures differ. A
missing/non-callable reference reports `No predefined reference`. No same-named
Torch function or implicit reference is invented.

The helper does not take over execution or copy inputs. Keep the test's original
input cloning, CPU/upcast reference inputs, mutation/alias checks, output
device/dtype assertions and tolerances. In-place operations need independent
inputs prepared by the test, as they do in normal correctness mode.

The pytest plugin requires an actual candidate-side reference call in each
passing correctness node. Missing declarations are detected at runtime, rather
than by guessing local variable names during collection; skips/xfails cannot
hide a missing declaration or a caught reference-contract error. A capability
skip after a valid explicit reference selection may remain skipped, but an
all-skip run does not count as validation. Calls made only by fixture setup do
not satisfy the test body's invocation check.

Correctness JSON identifies `candidate_source: reference` and
`reference_candidate_calls` per node. `reference_calls` counts only invocations
observed through the shared helper; a test's direct expected-side reference call
is not included. The earlier `ReferenceCase`/`@reference_test` factory interface
has been removed; migrate its input preparation and assertions back into the
test body and use `get_candidate` for candidate selection.

## Benchmark contract

Standard benchmarks explicitly declare their reference as callable `torch_op`.
This option also uses that callable on the candidate side through the same shared
selection and invocation tracking as correctness. It retains the original input
builders, case IDs, selected level, timer, warmup and metrics. `--case-id` replays
one exact case. A real candidate-side call is required for every timing record;
only then is `candidate_source: reference` recorded.

This initial mode supports forward benchmarks only. `is_backward=True` is
rejected explicitly: its preparation forward pass cannot prove that the timed
backward callable ran. Ordinary backward benchmarks are unchanged without the
flag.

A missing/non-callable reference, a custom execution/timing override or metrics
without candidate latency is an explicit error in this mode. The ordinary
behavior of those benchmarks remains unchanged. `OpBenchmark` skips eager public
candidate lookup in reference mode, so an unavailable candidate export is not
needed. No same-named Torch function or implicit reference is invented.

External override/config, dummy, reference-only, preflight, profiling, listing,
query, collection-only, marker collection and parallel execution cannot be
combined with this mode. Catalog listing is a separate command. Empty selection
and an all-skip run do not count as successful validation.

Reference-vs-reference proves harness executability, not that the reference is
independently correct or that an optimized candidate works. Continue to use
dummy injection to validate the external override route. These timings are
functional diagnostics, not published H800 performance baselines.
