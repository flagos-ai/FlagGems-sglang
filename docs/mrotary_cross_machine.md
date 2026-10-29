# mrotary_embedding across devices

The six model benchmark configurations and the seventh correctness-only
axis-routing case remain unchanged. NVIDIA `get_reference` dispatch uses the
original SGLang v0.5.11 Triton implementation. The default reference is now a
portable Torch implementation with the same in-place q/k, None-returning ABI.
Only the documented interleaved Neox/head256/rotary64/sections11,11,10 mode is
supported by that Torch reference; other modes raise explicitly.

## Fixed NVIDIA golden data

Use op-verify's native/layout contract feature, with this checkout available on
both export and replay machines. The verification adapters load standalone
sources, so they do not require installing the SGLang runtime or C++ extensions.

```sh
export OP_VERIFY_PROBLEMS_DIR="$PWD/verification/problems"
export OP_VERIFY_KS3_PREFIX=op-verify/sglang-mrotary/20261029-native-v1
# Set OP_VERIFY_KS3_AK / OP_VERIFY_KS3_SK privately.
# On NVIDIA:
op-verify export -p rope/mrotary_embedding --reference-source native
# On a target, after loading its existing Torch vendor runtime:
op-verify check -p rope/mrotary_embedding --device npu
```

The native adapter always calls the version-pinned Triton source; it never uses
the portable reference to generate golden outputs. Export serializes input
values before either q or k is changed. Replay restores positions stride and
backing capacity, invokes the portable/local vendor reference, then compares
mutated q/k with downloaded NVIDIA q/k. The additional axis-routing case keeps
positions stride `[512,1]` and uses distinct T/H/W rows. Numerical comparisons
use atol=0.002, rtol=0.016 and reject NaNs. This cross-device check establishes
Torch-reference agreement; the existing operator pytest separately checks
candidate behavior, exact trailing channels, immutable inputs and in-place ABI.
Do not treat reference-as-candidate alone as independent correctness evidence.

For a necessary vendor specialization, use the existing
`reference/vendors/_<vendor>/mrotary_embedding.py` dispatch and point op-verify's
local vendor adapter at the same implementation. Revalidate it on the same
NVIDIA dataset. No vendor specialization is introduced without evidence of a
failure in the common Torch implementation.

## H800 performance

Use ops-baseline's `ops/sglang/mrotary_embedding.py` with
`FLAGGEMS_SGLANG_SOURCE` pointing at this checkout:

```sh
python tools/collect_baseline_nvidia.py --repo sglang --ops mrotary_embedding \
  --device 0 --reference-chip H800 --report-dir reports/mrotary \
  --output op_sglang_mrotary_h800.json
```

Do not pass `--no-ncu` when collecting data for resource-based scaling. Preserve
NCU reports, actual environment metadata and source hashes. The six keys are
hashes of the full tensor geometry/dtype/offset and scalar ABI, not just q shape;
this distinguishes contiguous and graph-strided positions. Full descriptions
are retained in each record's config. An exact-key miss cannot use shape-only
fallback. Source and target must use matching versions of this case contract.

On the target, benchmark the injected candidate locally:

```sh
pytest benchmark/test_mrotary_embedding.py -m mrotary_embedding --level core \
  --override mrotary_embedding:/absolute/path/candidate.py:kernel \
  --metrics latency --base-data /absolute/path/op_sglang_mrotary_h800.json
```

The `vs Base` value is `(H800 latency / target latency) / factor`, where factor
is target/reference capability for the measured bottleneck resource. Record the
actual chip and factor used. Missing factors yield the framework's raw ratio;
that is not a hardware-normalized result. This is an estimate based on the
existing ops-baseline model, not a measured vendor-native target baseline.
The shared timer repeats the in-place operator, matching existing collection
semantics; input creation and cache construction are outside the timed callable.

If the NVIDIA host cannot reach KS3, use `export --output-dir` there, transfer the
bundle unchanged, then use `op-verify upload --input-dir` on a connected host.
This is the path used for the initial dataset; it does not regenerate golden
outputs outside NVIDIA.

## Initial validation (2026-10-29)

The fixed dataset is `op-verify/sglang-mrotary/20261029-native-v1`, with manifest
SHA256 `22d91f78dfc1f702732ebe57747c6a376fdd3a7c405dab3742fc4c384a74efee`.
It contains seven cases generated on actual NVIDIA H800 with Torch
`2.8.0a0+5228986c39.nv25.05`, CUDA 12.9 and Triton 3.6.0. The native compute
function ASTs are unchanged from PR #247. The implementation file SHA256 is
`3d925d9da864bdcdb93cf802e1566a01e773ab3598c3e967b144ec92a67847e7`.

Ascend 910B4, Torch `2.9.0+cpu` plus torch_npu `2.9.0.post2`, downloaded the real
KS3 tensors and passed all seven comparisons with zero skips. The largest
absolute difference was 0.0078125 and passed the stated atol/rtol combination;
this is numerical agreement, not bitwise identity. Downloaded payload digests
were checked against the same manifest. Other domestic backends have not been
validated by this run. The native SGLang performance snapshot is maintained
separately in ops-baseline.
