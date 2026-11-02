# Documented plugin operator names

The eight interfaces from the workload documents use `_plugin` to avoid sharing
an API name with independent operators synchronized from master. This is a name
migration: signatures, input values/layouts, mathematical references, mutation
and alias checks, tolerances, timing and workload coverage are unchanged.

| Test / Definition / candidate name | Native implementation / stored baseline group |
| --- | --- |
| `gemma_rmsnorm_plugin` | `gemma_rmsnorm` |
| `gemma_fused_add_rmsnorm_plugin` | `gemma_fused_add_rmsnorm` |
| `silu_and_mul_out_plugin` | `silu_and_mul_out` |
| `select_experts_plugin` | `select_experts` |
| `_fused_moe_kernel_sequence_plugin` | `_fused_moe_kernel_sequence` |
| `chunk_gated_delta_rule_fwd_plugin` | `chunk_gated_delta_rule_fwd` |
| `fused_recurrent_gated_delta_rule_packed_decode_plugin` | `fused_recurrent_gated_delta_rule_packed_decode` |
| `mrotary_embedding_plugin` | `mrotary_embedding` |

Files are `tests/test_<plugin>.py`, `benchmark/test_<plugin>.py` and
`definitions/<plugin>.json`. The MoE marker is
`underscore_fused_moe_kernel_sequence_plugin`; other markers equal the plugin
name. Existing upstream operator implementations/exports keep their names.
A plugin candidate must be supplied under its plugin name; it never implicitly
falls back to a same-named native or upstream candidate with a different ABI.

```sh
pytest tests/test_gemma_fused_add_rmsnorm_plugin.py --collect-only -q
pytest benchmark/test_gemma_fused_add_rmsnorm_plugin.py --list-cases --level core
pytest tests/test_gemma_fused_add_rmsnorm_plugin.py --ref-as-candidate
pytest benchmark/test_gemma_fused_add_rmsnorm_plugin.py --ref-as-candidate --level core
pytest tests/test_gemma_fused_add_rmsnorm_plugin.py --dummy-candidate gemma_fused_add_rmsnorm_plugin
pytest benchmark/test_gemma_fused_add_rmsnorm_plugin.py --dummy-candidate gemma_fused_add_rmsnorm_plugin --level core
```

## Reuse published performance measurements

Pass the existing paired ops-baseline artifacts unchanged:
`op_sglang_native_h800_perf_baseline.json` (28 cases) and
`op_sglang_mrotary_h800_perf_baseline.json` (6 cases). Each plugin benchmark sets
`baseline_op_name` to the stored native group and emits `baseline_operator` in
its shape metadata. Its exact geometry/ABI/timing hash remains unchanged. A
missing key is still a miss; the alias does not enable shape-only fallback.
Output operator names and case IDs use the plugin namespace.

```sh
pytest benchmark/test_gemma_fused_add_rmsnorm_plugin.py --ref-as-candidate \
  --level core --metrics latency \
  --base-data /path/to/ops-baseline/op_sglang_native_h800_perf_baseline.json
```

Native collectors continue to call the original SGLang entrypoints under their
native names. The source workload helpers accept both native IDs (for the
collector) and the eight explicit plugin aliases. No latency/counter, baseline
artifact or timing specification is rewritten by this migration.

## Reuse fixed NVIDIA tensors

New problem discovery/export paths end in `_plugin`, under
`verification/problems/operators` and `verification/problems/rope`. The
immutable historical KS3 manifests retain their existing problem IDs and
prefixes. `verification/plugin_datasets.json` maps every plugin to its original
dataset and baseline group. `verification/legacy` provides isolated aliases
for both historical `qwen36/*` and model-neutral `operators/*`/`rope/*` IDs.
Their local adapters select the plugin vendor reference, while native export
still selects the original SGLang function.

```sh
# Inspect the mapping without network/device initialization.
python verification/check_plugins.py --list
# Use op-verify's existing credentials/config; read the original datasets.
python verification/check_plugins.py --device cuda --output plugin-replay.json
# Select one plugin; use npu/musa/etc. for another backend.
python verification/check_plugins.py --operator gemma_fused_add_rmsnorm_plugin \
  --device npu --output gemma-plugin-replay.json
```

Add this checkout's `verification` directory to `PYTHONPATH` for op-verify
workers. `--cache-dir` supplies a parent containing `qwen36-cache` and
`mrope-cache`; these caches may be copied from previous runs to reuse tensor
files. The output uses plugin names, retains the original dataset IDs/prefixes,
and verifies all 35 fixed cases. Missing results/cases, numerical failures or
skipped cases fail the replay. It never exports or uploads data. Historical receipts under
`docs/validation` retain their original names, source revisions and hashes.

## Reusable CLI regression

`pytest tests/core/test_plugin_suite_cli.py -q` runs twelve real CLI checks on a
CUDA-compatible device: correctness collection, benchmark list-cases, both
reference-as-candidate modes, and one dummy run per plugin over correctness and
benchmark together. It requires all 45 correctness cases, 34 benchmark records
and 79 dummy invocations. Host-only environments skip this device regression;
the paired static-data/name/reset checks are in `test_plugin_operator_names.py`.
The CI selector explicitly includes the device regression when its source or
the shared workload module changes. Default plugin tests can skip when no
plugin implementation is supplied; the explicit reference/dummy regressions
must cover every case successfully.
