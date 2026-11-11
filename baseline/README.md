# NVIDIA performance baselines

FlagGems-sglang owns the workload, baseline artifact, and target test run.
Use a pinned FlagOS optest checkout as the H800 collector. The supported
workloads are listed in `optest-v1.json`. Their benchmark case plans reuse the
collection input builders and preserve the existing core and comprehensive grids.

Each module exposes `OP_NAME`, `DTYPES`, `IS_INPLACE`, `native()`, `grid()`,
`build_inputs()`, and `key_shape()`. Use the actual SGLang CUDA callable;
`native()` returns `None` if that implementation cannot be loaded.
`config()` describes the exact workload and `MUTATED_INPUTS` identifies
positional arguments that optest must restore before timing. The softcap
workload uses cap 30.0 and restores the logits outside the event interval in
kernel mode (five warmups, median of 25 samples) when the supplied baseline
record declares that protocol. Other softcap runs use the regular timer,
including runs without base data and runs with explicit `--warmup` or `--iter`.
SiLU uses Triton kernel timing with a 25 ms warmup and a 100 ms sampling
budget when its external record declares that protocol. Both benchmarks
report a local ratio when the requested mode, budget, or timer cannot match
the external record.

`optest-v1.json` declares that the checkout supports these timing protocols
and lists the exact workload keys required for publication. Update it with
the workload grid when adding or changing collection cases.

In the H800 SGLang/NCU environment, invoke the collector from a pinned optest
checkout. Set `--output` to the desired local artifact path:

```sh
python /path/to/optest/src/nv_baseline/collect_baseline_nvidia.py \
    --ops-dir /path/to/FlagGems-sglang/baseline --device 0 \
    --ops silu_and_mul,softcap_inplace_logits --output /tmp/sglang-nv.json
```

Verify that all selected workload keys completed timing and profiling before
publishing a versioned artifact. On the target machine, make that JSON available
locally and pass it to the benchmark with `--base-data`. For example:

```sh
flagos-optest /path/to/FlagGems-sglang --gpus 0 \
    --ops silu_and_mul,softcap_inplace_logits \
    --base-data /tmp/sglang-nv.json --speedup-source nv_baseline
```

Check `speedup_source` per result: `origin` means the runner used a local
ratio instead. The current workloads have one configuration per shape and dtype;
future variants with the same shape need distinct keys and verified matching.
The existing `tools/op_sglang_perf_baseline.json` and repository workflow
remain available during the migration. Correctness tests still use their own
reference and inputs.
