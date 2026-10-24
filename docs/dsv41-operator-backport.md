# DeepSeek V4.1 operator backport for SGLang v0.5.18

This patch supplies the native V4.1 kernel bindings, JIT headers and compatible
operator implementations missing from SGLang v0.5.18. It installs no scheduler
hook and has no dependency on the optimized FlagGems operator integration.
The native block-FP8 H100 configuration tables cover target and draft shapes,
preserving their tile choices on the release backport. Hopper swap-operand
matmul, Split-K and its FP32 reduction support those native configurations. FlagGems kernel dispatch
and operator replacement are supplied by the optimization integration.

Install this repository and TileLang 0.1.12. In a clean SGLang v0.5.18 checkout,
apply the SGLang-plugin-FL model/runtime backport and this kernel patch:

```bash
runtime_patch=/path/to/sglang-plugin-FL/sglang_fl/backports/sglang-v0.5.18-dsv41.patch
kernel_patch=/path/to/FlagGems-sglang/src/flaggems_sglang_backport/patches/sglang-v0.5.18-dsv41-kernels.patch
git apply --check "$runtime_patch" "$kernel_patch"
git apply "$runtime_patch" "$kernel_patch"
```

The model/runtime patch and kernel patch cover disjoint source files. Native
V4.1-capable SGLang revisions do not need either release backport.

Compatibility operators live under `flaggems_sglang_plugin.compat.dsv41`.
The SM90 indexer retains FP4 storage and the native BF16 score rounding points;
prefill supports dense and compressed-candidate layouts. The mHC exact entry
composes native statistics with TileLang Sinkhorn and preserves two-dimensional
coefficient inputs. Stable Top-K tie ordering and FlagTree 3.6 integer-metadata
compatibility are part of the source backport.

`tests/test_dsv41_backport_structure.py` checks that the backport has no optimized
dependency or installation hook. `tests/test_dsv41_compat_math.py` exercises its
numerical contracts on SM90. Applying the patch does not enable the separate
FP8 indexer, finite-shape dense GEMM, exact mHC fusion or W4A8 runner tuning.
