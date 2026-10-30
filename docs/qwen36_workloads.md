# Qwen3.6 workload interfaces

This suite adds seven explicit SGLang interfaces, each with a Definition,
correctness test and benchmark. It follows the linked 27B/35B workload documents
(main revision 107, child revisions 21/54). The inputs are deterministic synthetic
values: the documents supply geometry and call configuration, not tensor dumps.

| Interface | Documented cases | Comprehensive cases |
| --- | ---: | ---: |
| `gemma_rmsnorm` | 12 | 13 |
| `gemma_fused_add_rmsnorm` | 4 | 5 |
| `silu_and_mul_out` | 4 | 5 |
| `select_experts` | 2 | 3 |
| `_fused_moe_kernel_sequence` | 2 | 3 |
| `chunk_gated_delta_rule_fwd` | 2 | 6 |
| `fused_recurrent_gated_delta_rule_packed_decode` | 2 | 3 |

The existing single-argument `silu_and_mul` interface is unchanged; the new
`*_out` name exposes the native optional-out ABI. MoE uses the
`underscore_fused_moe_kernel_sequence` pytest marker. Correctness includes the
comprehensive cases; benchmark `--level core` selects the 28 documented cases.
Use ordinary pytest `--collect-only` to list correctness cases and benchmark
`--list-cases` for its inner shape catalog.

The Gemma K views retain strides 3584/2560 and offsets 3072/2048. GDN retains
full FP32 state pools of 1381/3193 slots with a nonconstant active slot and exact
checks on unselected slots. The chunk oracle returns all six native slots,
including None entries and BF16 intermediates. Decode verifies both return aliases;
fused-add verifies both mutations and its None return. TopK compares paired IDs
and weights independently of ordering. MoE preserves the documented TP4 partial
output, routing/padding geometry and in-place return.

Correctness checks dtype/device, geometry, allocation, readonly values and effects
as well as numerical output. BF16/state tolerances are atol 0.002 / rtol 0.016;
FP32 intermediates use 2e-5 / 2e-5. These are engineering thresholds, not values
from the workload documents. No new default candidates are supplied: the normal
missing-candidate skip remains, and override, dummy and reference-as-candidate
use the existing shared mechanisms.

## Original NVIDIA reference environment

Native baselines and golden outputs use upstream SGLang v0.5.11, commit
`612785ffdcaf35552f1ed433a981d596ca9fe900`, with Torch 2.11.0+cu130,
sglang-kernel 0.4.2, FlashInfer 0.6.8.post1 and Triton 3.6.0. These are original
CUDA/FlashInfer/Triton entrypoints; the portable Torch oracle is not a native
performance substitute. The manifest in `verification/qwen36-native-sources.json`
checks entrypoint ASTs with Python 3.12, and `qwen36-native-wheels.json` records
official wheel URLs and SHA256 hashes. Third-party SGLang plugins are disabled
before import to preserve native dispatch. Gemma explicitly requires FlashInfer.

Create a dedicated Linux Python 3.12 environment on a Torch 2.11/CUDA 13 image.
Download the nine official wheels in the manifest to a wheelhouse, then run:

```sh
python tools/prepare_qwen36_native_env.py --venv /workspace/venv \
  --wheelhouse /workspace/wheels
```

The tool rejects existing environments and verifies wheels before offline
installation. `--verify-only` rechecks an existing task environment. It does not
modify the image's system packages. Other SGLang dependencies must already be
provided by the base image; this is not a complete general-purpose installer.

## Fixed tensors on KS3

The seven problems under `verification/problems/qwen36` use op-verify contract
version 2 (companion op-verify PR #9). Set `PYTHONPATH` to include this checkout's
`verification` directory and point `OP_VERIFY_PROBLEMS_DIR` at
`verification/problems`. Export each problem explicitly, for example:

```sh
op-verify export -p qwen36/chunk_gated_delta_rule_fwd \
  --reference-source native --compress --output-dir /workspace/golden-bundle
```

Upload the offline bundle from a KS3-connected machine with a fresh prefix.
Compression preserves the full tensor bytes, layout and original content hash.
Target replay uses the portable vendor Torch oracle on those exact inputs.
Keep this checkout available on the target because problem adapters import its
shared helpers. The chunk contract appends the mutated state after six return
slots; packed decode captures its returned state only once. Optional outputs,
readonly inputs and aliases retain their actual ABI.

## Native performance

Use the companion ops-baseline `ops/sglang_qwen36` recipes with
`FLAGGEMS_SGLANG_SOURCE` pointing at this checkout. `--repo sglang_qwen36` keeps
these pinned recipes separate from older same-named SGLang profiles. The 28
core records use median timing and geometry/configuration hashes shared with
these benchmarks. Target benchmarks consume the collected artifact through
`--base-data`; target candidates are timed locally. Reference-as-candidate is a
pipeline check, not a claim of optimized candidate performance.

Validation receipts belong to the PR description. H800 native-vs-Torch checks
passed all 38 cases, including boundary controls, before golden export.


## Multi-backend reference validation

Golden replay uses the same `get_reference` dispatch as correctness and
benchmark: `reference/vendors/_<vendor>/<operator>.py` takes precedence over the
shared Torch oracle. No vendor-specific mathematical oracle was needed on the
verified Ascend, MetaX, Iluvatar, Hygon, MThreads, T-Head and Kunlunxin devices.

For the handoff's Kunlunxin `tj_gems` environment, preserve its existing
`python310_torch25_cuda` and set `XPU_EVENT_KL3_ENABLE=1`. One unrelated installed
candidate uses a Triton decorator unsupported by that SDK. The explicit opt-in
`FLAGGEMS_SGLANG_REFERENCE_ONLY=1` skips built-in candidate registration while
retaining actual device detection and reference dispatch. Ordinary imports are
unchanged. Use this only for reference validation or explicitly injected
candidates; it does not validate built-in optimized candidates.

Kunlunxin routing fixture indices are generated with an isolated CPU permutation
and transferred to the real target device, as required by the machine handoff's
known device-randperm limitation. This is outside timed reference execution;
mathematical reference operations stay on the target, and fixed KS3 inputs are
never regenerated.

Cross-chip scaling accepts the historical `t-head` baseline label as `thead`.
When a vendor SDK reports only the generic device name `GPU`, set
`FLAGGEMS_SGLANG_CHIP` to the model independently observed with its device tool
(e.g. `P800` from `xpu-smi`). This chooses a scaling record, not a device; retain
the normal visibility mask to select a card. Without this opt-in, actual device
name detection and missing-factor behavior remain unchanged.

Suiyuan is skipped in this validation run at the user's request. Other machines
outside the handoff plus the explicitly added T-Head/MThreads targets are not
part of acceptance. See `docs/qwen36_multibackend_validation.json`.

The current P800 hardware table has no bandwidth scaling factor. These memory-
bottleneck cases therefore retain `factor=None` and report the raw H800/target
latency ratio. P900 factors are never substituted. This does not affect golden
correctness or exact baseline-case matching.
