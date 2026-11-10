# NVIDIA performance baselines

The operator library owns the workload; FlagOS optest owns H800 collection and
publishes the resulting JSON. This directory starts with `silu_and_mul` and
`softcap_inplace_logits`. Their benchmark case plans reuse these input builders
and preserve the existing core and comprehensive grids.

Each module exposes `OP_NAME`, `DTYPES`, `IS_INPLACE`, `native()`, `grid()`,
`build_inputs()`, and `key_shape()`. Use the actual SGLang CUDA callable;
`native()` returns `None` if that implementation cannot be loaded.
`config()` describes the exact workload and `MUTATED_INPUTS` identifies
positional arguments that optest must restore before timing. The softcap
workload uses cap 30.0 and restores the logits outside the event interval in
kernel mode (five warmups, median of 25 samples) when the supplied baseline
record declares that protocol. Other softcap runs use the regular timer,
including runs without base data and runs with explicit `--warmup` or `--iter`.
Those runs report a local ratio when an external record is incompatible.

`optest-v1.json` declares that the checkout supports these timing protocols
and lists the exact workload keys required for publication. Update it with
the workload grid when adding or changing collection cases.

In the optest repository, run the `nv-baseline` workflow with
`repo=FlagGems-sglang` and `ref` set to the desired FlagGems-sglang commit.
For verification set `auto_update=false` and use the uploaded JSON and its
collection status report. The equivalent collector invocation in the
configured H800 SGLang/NCU environment is:

```sh
python /path/to/optest/src/nv_baseline/collect_baseline_nvidia.py \
    --ops-dir /path/to/FlagGems-sglang/baseline --device 0 \
    --ops silu_and_mul,softcap_inplace_logits --output /tmp/sglang-nv.json
```

The workflow requires all declared shapes and dtypes to complete timing and
profiling before it can replace the published file. The target runner can
then use the static H800 baseline:

```sh
flagos-optest /path/to/FlagGems-sglang --gpus 0 \
    --ops silu_and_mul,softcap_inplace_logits \
    --base-data /tmp/sglang-nv.json --speedup-source nv_baseline
```

Check `speedup_source` per result: `origin` means the runner used a local
ratio instead. These two workloads have one configuration per shape and dtype;
future variants with the same shape need distinct keys and verified matching.
The existing `tools/op_sglang_perf_baseline.json` and repository workflow
remain available during the migration. Correctness tests still use their own
reference and inputs.
