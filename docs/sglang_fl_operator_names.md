# Documented sglang_fl operator names

The eight interfaces from the workload documents use the `sglang_fl_` prefix to avoid sharing
an API name with independent operators synchronized from master. This is a name
migration: signatures, input values/layouts, mathematical references, mutation
and alias checks, tolerances, timing and workload coverage are unchanged.

| Test / Definition / candidate name | Native implementation |
| --- | --- |
| `sglang_fl_gemma_rmsnorm` | `gemma_rmsnorm` |
| `sglang_fl_gemma_fused_add_rmsnorm` | `gemma_fused_add_rmsnorm` |
| `sglang_fl_silu_and_mul_out` | `silu_and_mul_out` |
| `sglang_fl_select_experts` | `select_experts` |
| `sglang_fl_fused_moe_kernel_sequence` | `_fused_moe_kernel_sequence` |
| `sglang_fl_chunk_gated_delta_rule_fwd` | `chunk_gated_delta_rule_fwd` |
| `sglang_fl_fused_recurrent_gated_delta_rule_packed_decode` | `fused_recurrent_gated_delta_rule_packed_decode` |
| `sglang_fl_mrotary_embedding` | `mrotary_embedding` |

Files are `tests/test_<operator>.py`, `benchmark/test_<operator>.py` and
`definitions/<operator>.json`. The MoE marker is
`sglang_fl_fused_moe_kernel_sequence`; other markers equal the plugin
name. Existing upstream operator implementations/exports keep their names.
A prefixed candidate must be supplied under its documented name; it never implicitly
falls back to a same-named native or upstream candidate with a different ABI.

```sh
pytest tests/test_sglang_fl_gemma_fused_add_rmsnorm.py --collect-only -q
pytest benchmark/test_sglang_fl_gemma_fused_add_rmsnorm.py --list-cases --level core
pytest tests/test_sglang_fl_gemma_fused_add_rmsnorm.py --ref-as-candidate
pytest benchmark/test_sglang_fl_gemma_fused_add_rmsnorm.py --ref-as-candidate --level core
pytest tests/test_sglang_fl_gemma_fused_add_rmsnorm.py --dummy-candidate sglang_fl_gemma_fused_add_rmsnorm
pytest benchmark/test_sglang_fl_gemma_fused_add_rmsnorm.py --dummy-candidate sglang_fl_gemma_fused_add_rmsnorm --level core
```

## Reuse published performance measurements

Use the unified ops-baseline artifact `op_sglang_perf_baseline.json`, which
contains the 28 native cases and six mRoPE cases alongside the legacy operators.
The eight documented top-level
operator keys match the `sglang_fl_` Test / Definition names above. Benchmarks
look up their own operator name directly. Existing measurements, exact workload
keys and timing metadata are preserved. A missing exact key remains a miss.
Update this JSON file together with this checkout and the KGS baseline selector.

```sh
pytest benchmark/test_sglang_fl_gemma_fused_add_rmsnorm.py --ref-as-candidate \
  --level core --metrics latency \
  --base-data /path/to/ops-baseline/op_sglang_perf_baseline.json
```

Native collectors publish the same `sglang_fl_` keys while calling the original
SGLang entrypoints under their native names. The source workload helpers accept both native IDs (for the
collector) and the eight explicit plugin aliases. Latency/counter values, workload keys and timing specifications are preserved.

## Reuse fixed NVIDIA tensors

New problem discovery/export paths start with `sglang_fl_`, under
`verification/problems/operators` and `verification/problems/rope`. The
immutable historical KS3 manifests retain their existing problem IDs and
prefixes. `verification/sglang_fl_datasets.json` maps every plugin to its original
dataset and baseline group. `verification/legacy` provides isolated aliases
for both historical `qwen36/*` and model-neutral `operators/*`/`rope/*` IDs.
Their local adapters select the prefixed vendor reference, while native export
still selects the original SGLang function.

```sh
# Inspect the mapping without network/device initialization.
python verification/check_sglang_fl.py --list
# Use op-verify's existing credentials/config; read the original datasets.
python verification/check_sglang_fl.py --device cuda --output plugin-replay.json
# Select one plugin; use npu/musa/etc. for another backend.
python verification/check_sglang_fl.py --operator sglang_fl_gemma_fused_add_rmsnorm \
  --device npu --output gemma-plugin-replay.json
```

Add this checkout's `verification` directory to `PYTHONPATH` for op-verify
workers. `--cache-dir` supplies a parent containing `qwen36-cache` and
`mrope-cache`; these caches may be copied from previous runs to reuse tensor
files. The output uses documented names, retains the original dataset IDs/prefixes,
and verifies all 35 fixed cases. Missing results/cases, numerical failures or
skipped cases fail the replay. It never exports or uploads data. Historical receipts under
`docs/validation` retain their original names, source revisions and hashes.

## Reusable CLI regression

`pytest tests/core/test_sglang_fl_suite_cli.py -q` runs twelve real CLI checks on a
CUDA-compatible device: correctness collection, benchmark list-cases, both
reference-as-candidate modes, and one dummy run per plugin over correctness and
benchmark together. It requires all 45 correctness cases, 34 benchmark records
and 79 dummy invocations. Host-only environments skip this device regression;
the paired static-data/name/reset checks are in `test_suite_operator_names.py`.
The CI selector explicitly includes the device regression when its source or
the shared workload module changes. Default plugin tests can skip when no
plugin implementation is supplied; the explicit reference/dummy regressions
must cover every case successfully.
