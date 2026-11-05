<!--
Copyright 2026 FlagOS Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# List and replay benchmark cases

The existing `flaggems.benchmark-case-list/v2` API separates a lightweight
case plan from Tensor construction. All 125 operator benchmarks in this
repository provide a planner and builder, including both `reverse` variants
of the two local cumsum benchmarks.

## List cases

```shell
pytest -q benchmark/test_silu_and_mul.py --level core --list-cases --output /tmp/silu-core.json
pytest -q benchmark/test_silu_and_mul.py --list-cases --output /tmp/silu-comprehensive.json
pytest -q benchmark --level core --list-cases --output /tmp/all-core.json
pytest -q benchmark --list-cases --output /tmp/all-comprehensive.json
```

For `silu_and_mul`, core contains the original five `SHAPES`; the default
comprehensive level also contains the original ten `MORE_SHAPES`. Both retain
the original `torch.bfloat16` dtype. Listing calls the case planner but not
the Tensor builder, reference, candidate, or timer. It still imports the normal
package and therefore requires its regular Torch/vendor runtime environment.

Listing does not require a candidate export on the current backend. The
entrypoints defer public candidate lookup via `set_gems_from_module` in
listing mode, and existing missing-implementation skip guards bypass only
that mode. Ordinary execution retains its original binding, errors and skips.

The JSON contains the actual input shape, dtype, dimension parameters and
seed. For example, the third core case describes `hidden_states` with shape
`[64, 8192]`, `bs=64`, `d=4096`, and `seed=0`.

The newly migrated providers retain the original `SHAPES` entry in
`params.configuration`, alongside named dimensions, option values, seeds and
fixed module configuration. Optional Tensor inputs are described as `null`
when absent; nested LoRA routing tensors use dotted names such as
`batch_info.seg_indptr`. Dimensions determined by Tensor values are strings,
for example `["sum(seq_lens)"]`, rather than fabricated numeric sizes.
The builder resolves these dimensions using the unchanged original input
generator, only after selection.

## Replay a selected case

Use the exact ID from the catalog and the same level and source revision:

```shell
pytest -q benchmark/test_silu_and_mul.py --level core --case-id 'benchmark/test_silu_and_mul.py::test_perf_silu_and_mul::core::bfloat16::2' --record json --output /tmp/silu-selected.json
```

The builder materializes only the selected case, and the ordinary benchmark
executes its actual reference/candidate timers. Unknown or unexecuted IDs
fail the session. IDs include the pytest node, level, dtype and ordinal;
outer pytest parametrization therefore gets separate node-scoped IDs.
These IDs are not content hashes. Preserve the source revision, invocation,
and actual outer parameter values with a catalog before using it as an
acceptance artifact; changing source or case order can change its meaning.
Exact replay selects the workload configuration. It does not reseed global
RNGs or save input values. `mrope_fused` retains its original global Torch
RNG, and shuffled LoRA permutations also retain their original global RNG.

Listing rejects empty providers and duplicate IDs within a provider or across
the session's published benchmark lists. A listing session that publishes no
cases also fails. Consumers must check the pytest exit status as well as the
catalog; a failed run may retain partial output for diagnosis.

## Connect another operator

Use the existing `GenericBenchmark`/`OpBenchmark` arguments:

- `case_fn(shape, dtype)` yields `BenchmarkCasePlan` objects containing the
  actual Tensor shapes, non-Tensor configuration in `params`, and private
  `builder_args`. It must not allocate inputs.
- `build_inputs_fn(plan, dtype, device)` constructs exactly one input tuple
  using the original input-generation logic. `builder_args` are private and
  are not serialized into the public catalog.
- Keep the original shapes, their order, dtypes, seeds and reference/candidate
  bindings. Do not mix the planner/builder pair with `input_fn`.

For a legacy generator that yields exactly once per shape, reuse
`single_input_builder(_input_fn)` from `benchmark.cases` as `build_inputs_fn`
and set the planner's `builder_args=(shape,)`. This adapter calls the original
generator only for selected cases and rejects zero or multiple yields. Bind
any outer pytest arguments to both the planner and the generator; record
those arguments in `params` as the cumsum benchmarks do.

The migrations preserve the original input generators, including seeded
float32 generation followed by conversion where used. Default benchmark
invocation still runs the same core or comprehensive workloads; this change
adds case IDs to their timing records without changing the timed operator.
