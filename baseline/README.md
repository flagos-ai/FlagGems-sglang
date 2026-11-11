# NVIDIA performance baselines

FlagGems-sglang owns the workload, baseline artifact, and target test run.
Use a pinned FlagOS optest checkout as the H800 collector. The supported
workloads are listed in `optest-v1.json`. Their benchmark case plans reuse the
collection input builders and preserve the existing core and comprehensive grids.
The manifest lists the collection keys for each workload.

Each module exposes `OP_NAME`, `DTYPES`, `IS_INPLACE`, `native()`, `grid()`,
`build_inputs()`, and `key_shape()`. Use the actual SGLang CUDA callable;
`native()` returns `None` if that implementation cannot be loaded.
`config()` describes the exact workload and `MUTATED_INPUTS` identifies
arguments that optest must restore before timing.
`softcap_inplace_logits` uses cap 30.0 and restores the logits outside the event
interval in kernel mode (five warmups, median of 25 samples) when the supplied
baseline record declares that protocol. The out-of-place workloads use Triton
kernel timing with a 25 ms warmup and a 100 ms sampling budget when their
external records declare that protocol. If the requested mode, budget, or
timer cannot match a declared protocol, the benchmark uses the regular timer
and reports a local ratio.

`optest-v1.json` declares that the checkout supports these timing protocols
and lists the exact workload keys required for publication. Update it with
the workload grid when adding or changing collection cases.

In the H800 SGLang/NCU environment, invoke the collector from a pinned optest
checkout (`e160a8ae56dcc3594e8b34d6408f04647d5b7e16` was used for workload
discovery). Set `--output` to a fresh local artifact path:

```sh
WORKLOADS=/path/to/FlagGems-sglang/baseline/optest-v1.json
OPS=$(python -c 'import json, sys; print(",".join(op for op in json.load(open(sys.argv[1])) if op != "chunk_scaled_dot_kkt"))' "$WORKLOADS")
python /path/to/optest/src/nv_baseline/collect_baseline_multi_gpu.py \
    --ops-dir /path/to/FlagGems-sglang/baseline --devices 0 \
    --ops "$OPS" --output /tmp/sglang-nv.json
```

Run the command above with SGLang v0.5.21. For standalone KKT, use SGLang
v0.5.11 and run the same collector with `--ops chunk_scaled_dot_kkt` and
`--output /tmp/sglang-kkt-nv.json`. Later releases fuse KKT with the triangular
solve. Add the `chunk_scaled_dot_kkt` entry to a copy of the primary JSON
before replay and retain the two original files for provenance.

Before publishing a versioned artifact, check the collector's
`/tmp/sglang-nv.json.work/status.json` and confirm every selected key listed in
`optest-v1.json` has a latency and NCU record in the JSON. On the target machine,
install `flagos-optest` from the same pinned checkout, make that JSON available
locally, and pass it to the benchmark with `--base-data`:

```sh
WORKLOADS=/path/to/FlagGems-sglang/baseline/optest-v1.json
OPS=$(python -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))))' "$WORKLOADS")
flagos-optest /path/to/FlagGems-sglang --gpus 0 --ops "$OPS" \
    --base-data /tmp/sglang-nv.json --speedup-source nv_baseline
```

Check `speedup_source` per result: `origin` means the runner used a local
ratio instead. Workloads such as `gate_topk` with multiple configurations per shape use
explicit collection keys in their benchmark case plans. The benchmark looks
up those keys directly.
The existing `tools/op_sglang_perf_baseline.json` and repository workflow
remain available during the migration. Correctness tests still use their own
reference and inputs.
