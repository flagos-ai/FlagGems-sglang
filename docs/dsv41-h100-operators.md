# DeepSeek V4.1 SM90 SGLang integration

This opt-in adapter replaces reviewed native SGLang entry points before
model construction and CUDA Graph capture. Implementation ownership follows
the interface and data layout, independently of operator name, model family,
source provenance or implementation language.

| Interface / layout | Owner | Submission |
|---|---|---|
| Block-FP8 GEMM with matrix/numeric-scale inputs, Split-K reduction and finite H100 configurations | FlagGems | [#6631](https://github.com/flagos-ai/FlagGems/pull/6631) |
| mHC statistics/combine/Sinkhorn/exact fusion, including TileLang | FlagGems | [#6684](https://github.com/flagos-ai/FlagGems/pull/6684), mHC review commit |
| Packed E2M1 decode and explicit-physical-slot index score math | FlagGems | [#6684](https://github.com/flagos-ai/FlagGems/pull/6684), FP4 review commit |
| req_to_token addressing, compressed-slot translation, candidate maxima/consumer layout and six-query/persistent fusion | FlagGems-sglang | This PR |
| Plugin lifecycle, scheduler GPU selection, native source checks and identity-based rebinding | FlagGems-sglang | This PR |

There are no private FP8 GEMM, mHC kernels or H100 configuration copies in
this repository. Their numerical tests and standalone benchmarks accompany
their owning FlagGems PRs.

## Exact dependency and opt-in installation

The `dsv41` extra pins **FlagGems at `6720f4f11c4f345d5866114ea402caa5e7faba83`** and TileLang
0.1.12. That immutable fork integration commit combines the two still-open
FlagGems PRs for reproducible testing; it is not an upstream release. The
pin must move to the corresponding upstream revision after both PRs merge.
The adapter checks the three common operator ABI values `(1,1,1)` at install.
Changing that contract requires explicit review in both repositories.

```bash
pip install '.[dsv41]'
FLAGGEMS_SGLANG_DSV41=1 \
SGLANG_PLUGINS=flaggems_dsv41 \
DNN_VENDOR=nvidia ARCH=sm_90 \
python -m sglang.launch_server ...
```

The extra is optional. Importing the default-off plugin does not import
Torch/CUDA/FlagGems. Importing the framework operator package does not
require the extra until a V4.1 fusion entry is invoked. Backend vendor and
architecture imports use this package's complete namespace, avoiding bare
`_nvidia`/`hopper` module-cache collisions when the two libraries coexist.
Both library import orders are regression-tested.

## Framework-specific FP4 fusion

`fp4_index_logits_req_to_token` and `fp4_index_logits_candidate_blocks`
remain framework-owned. They accept SGLang's device request IDs, request to
full-token-slot map, compression ratio 1/2 and candidate layout. They read
request membership and visible lengths on every Graph replay. Only an
explicit semantic `query_group_size=6` hint enables six-query reuse;
row count alone never implies grouping. Partial groups and unsupported
head/candidate combinations retain the native per-query path.

The fused kernels import the FlagGems `_e2m1_decode` and
`fp4_weighted_head_score` Triton helpers. The common math is inlined inside
the framework kernels, preserving direct address calculation and candidate
writes. There are no new intermediate tensors or additional GPU launches.
Candidate sentinels, maxima, visible lengths and padding remain owned here.

The independent explicit-slot API is imported directly from FlagGems by the
adapter; it is not re-registered as a framework-specific operator. Its
packed format and numerical contract live in FlagGems. The fused paths
preserve BF16 key/dot/product/head-sum rounding and FP32 logit storage.

### Optional packed FP8 schedule on FlagTree 3.6

The guarded FP8 path requires the common `FP4_FP8_SCORE_ABI=2` helpers and
the explicit six-query semantic hint. Query and scale eligibility are
recomputed on device on every replay; unsupported values use the BF16
fallback. Batches divisible by six do not by themselves establish grouping.

On FlagTree/Triton 3.6.0, the optional TLE schedule uses FlagGems' packed
word decoder and retains the framework's fused request/slot/candidate
layout. It selects 25 measured batch/ratio buckets from batches
6/12/18/24/30/36/42/48/60/72/84/96/120/192 and ratios 1/2. B6-ratio1,
B6-ratio2 and B12-ratio1 retain the Gluon FP8 schedule. Missing TLE or
helper support, other compiler versions, and
`FLAGGEMS_DSV41_INDEXER_TLE=0` also retain Gluon. Existing BF16 admission
checks remain ahead of either FP8 schedule.

The default TLE schedule is serial. Experimental warp-specialized kernels
and compiler patches are not enabled by its dispatch. Install before Graph
capture so optional helper resolution occurs during warmup. Qualification
and performance of this new schedule are recorded separately from the
historical serving results below.

## Source-pinned lifecycle

At discovery time the entry-point module is lightweight. In each scheduler
process it first runs native process configuration, selects that scheduler's
explicit GPU, then installs the adapter before model construction/Graph.

The adapter checks three complete native operator-file SHA256 identities
before replacing any function. Unknown SGLang revisions fail explicitly;
compatibility is not inferred from a version string. It rebinds already
imported SGLang aliases by original function identity. Uninstall restores
only bindings still owned by this installation.

Generic mHC calls route to FlagGems. FP8 routes to FlagGems' existing
`w8a8_block_fp8_matmul(A,B,As,Bs,block_size,output_dtype)` entry only for
reviewed finite E4M3FN/FP32-scale/BF16-or-FP32 shapes. Packed scales, FP16
output, Torch compilation and unlisted shapes retain the native entry.
No routing, placement, communication, model weights or KV allocator is
replaced. Extending source support requires reviewing native numerical
contracts and updating source pins.

## Verification and serving shapes

```bash
PYTHONPATH=src python -m pytest -q tests/test_dsv41_fp4_indexer.py \
  tests/test_dsv41_plugin.py tests/test_dsv41_dependency_imports.py
PYTHONPATH=src python benchmark/dsv41_h100_graph.py --full --rounds 2 \
  --rep-ms 100 --out fp4-graph.json
```

FP4 tests cover independent packed-page references, ratio/page boundaries,
compact candidates, partial query groups and changed-input Graph replay.
Cross-repository verification additionally compares all 212 serving shapes
against the exact pre-split implementations; FP8 covers BF16 and FP32 output.
Those comparison results are reported separately from code ownership.

The FP4 Graph benchmark preserves full-token capacity 655627, visible
length approximately 131080, page64, ratio1 candidate scoring
(`candidate_block_size=8, write_logits=False`) and ratio2 dense output
(width327813). Optional `--fixtures DIR` uses retained C32 B48 inputs;
other row counts select/repeat complete six-query groups and duplicate
complete KV pages with relocated physical slots. Only B48 is captured;
other sizes are labelled derived. Raw activations are not published.

Timing captures the complete public call, includes output allocation and
candidate writes, and excludes JIT/input generation. It uses 25 eager
warmups, median 100ms CUDA Graph timing, alternating implementations and
reversed case order in round2. Every shape passes byte-exact regression
before timing; two-round mean latencies form equal-shape geometric means.

For the tested TP2 deployment, target Graph captures include local
bs1/2/3/4/5/6/7/8/10/12/14/16/20. Draft has five tokens per request, so the
native TP2 alignment rule captures even buckets and pads odd local batches
to the next even bucket. These are padded Graph replays. Extra TP1 odd
rows remain in the generic mHC benchmark.

## Measurement provenance

The original pre-split source is commit
`b188ef6b9de41138331299651aa23f3e11fb8187` of this repository. Its two-round
H100 measurements are retained in [the historical report](dsv41-h100-benchmark.json):

| Operator owner and scope | Shapes | Geometric mean | Range |
|---|---:|---:|---:|
| FlagGems FP8 complete GEMM | 162 | 1.410x | 1.042–3.795x |
| FlagGems-sglang fused FP4 scoring/candidates | 26 | 2.751x | 1.678–3.069x |
| FlagGems mHC statistics + Sinkhorn chain | 24 | 1.084x | 1.000–1.285x |

These are historical, measured numerical bodies, not fresh measurements
of the repository split. FP8/mHC use synthetic values, FP4 retained/derived
inputs. FP4's ratio2 output width is the full serving Graph width; earlier
smaller-output diagnostics are excluded. New numerical parity, Graph and
performance regressions are checked separately before changing the pinned
dependency. Neither microbenchmarks nor this synthetic serving workload
establish whole-model ERQA/OCRBench quality.

## Post-split regression measurement

The [post-split report](dsv41-split-verification.json) retains all 212 rows.
The full two-round repeat measured FP8 1.4099x (1.0428–3.8070x), fused
framework FP4 2.7516x (1.6814–3.0501x), and mHC 1.0832x
(0.9997–1.2851x). Methods, input provenance and shape weighting match the
historical report. The 212-shape byte-parity check includes 324 FP8 dtype
comparisons and 78 changed-input FP4 Graph replays. Small differences
around 1.0x remain in the table. The common math and its configuration data
were also audited against the measured pre-split source.


## Prefill operators and numerical fixes

Common prefill score math, exact post/combine, per-call mHC weight preparation,
masked row quantization and SwiGLU/row quantization now live in FlagGems.
`dsv41_fp8_prefill_candidates` retains compressed-candidate addressing and
consumer column layout. `dsv41_w4a8_moe` qualifies the SGLang/Humming top-k-six
physical row layout and passes the original Humming quantization helpers as
explicit JIT callbacks to FlagGems. Its W2 schedule is qualified only for
98,304 route rows; other shapes retain their original GEMM configuration.
These prefill APIs do not automatically rewrite model methods.

The `stable-topk` offline profile gives equal scores a logical-index tie key.
It fixes candidate membership for the qualified radix tie pool of at most
2,048 entries; downstream canonical index sorting remains required. Pool
overflow and NaN scoring are not newly qualified. This is a precision fix:
the measured 75-row Graph microbenchmark increased from 7.18 to 9.09 us.

The `flagtree36-metadata` profile rewrites SGLang's ragged/padded prefill
metadata kernel for the qualified Triton 3.6 interface. It is an operator
compatibility patch, not a FlagTree compiler modification. Both profiles
verify complete source/output hashes and run offline before worker startup:

```bash
python -m flaggems_sglang_plugin.apply_operator_patches /path/to/site-packages stable-topk
python -m flaggems_sglang_plugin.apply_operator_patches /path/to/site-packages flagtree36-metadata
```

Original TopK qualification covered repeated real tie rows, 35 boundary,
ragged/paged and changed-input Graph cases, and Compute Sanitizer with no
errors. Repackaged patch application is checked separately on CPU. The new
prefill API boundaries and callback extraction still need a fresh GPU run
before readiness; earlier GPU results apply to the recorded original sources.


## Historical all-concurrency W4A8 adapter

`flaggems_sglang_plugin.w4a8_moe` preserves the qualified all-concurrency
masked-row launch and native W13/W2 tactic selection for the historical W4A8
runner. It calls the common `flag_gems.fused.dsv41.masked_row_quant` core and
keeps original allocations, numeric helpers and physical expert/capacity rows.
It installs only three owned Python bindings before Graph capture and restores
them by identity. It is default-off, is not registered by the main plugin, and
requires the exact historical source pins, four experiment flags, Torch
2.13/cu130, FlagTree 0.7.0+gite86aa7cf and Humming 0.1.12. Current-main runners
are intentionally rejected. Existing captured Graphs require recapture after
uninstalling the adapter.

This is native-W4A8 operator parity qualification; it does not establish W4A8
versus W4A16 model-quality parity or enable W4A8 as a serving default. Historical
30-case A8/A16 comparisons did not meet the fixed original tolerance. The
public draft retains that limitation independently of its A8 byte parity.
