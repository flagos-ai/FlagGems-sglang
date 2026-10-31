[English|[中文版](./README_cn.md)] [![basic ci](https://github.com/flagos-ai/FlagGems-sglang/actions/workflows/basic-ci.yml/badge.svg)](https://github.com/flagos-ai/FlagGems-sglang/actions/workflows/basic-ci.yml)

## Introduction

FlagGems-sglang is part of [FlagOS](https://flagos.io/).
FlagGems-sglang is a high-performance operator library designed for multiple hardware backends. It provides optimized implementations of common SGLang operators and supports high-performance inference and deployment for a variety of widely used models.

FlagGems-sglang is a high-performance deep learning operator library implemented using the [Triton programming language](https://github.com/openai/triton) launched by OpenAI.

## Features

- Operators have undergone deep performance tuning
- Triton kernel call optimization
- Flexible multi-backend support mechanism
- Support for common sglang operators (flashinfer-related operators, etc.)

## Quick Installation
### Install Dependencies
```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```
### Install FlagGems-sglang
```shell
git clone https://github.com/flagos-ai/FlagGems-sglang.git
cd FlagGems-sglang
pip install  .
```

## Usage Example

```python
import torch
import flaggems_sglang

# Create a tensor
x = torch.randn(1024, device='cuda')

# Apply ReLU activation
y = flaggems_sglang.relu(x)
```

## Tests and Benchmark Quick Start

The following commands can be used for quick validation after installation.

### Run tests

```shell
cd /workspace/FlagGems-sglang
pytest -q tests --collect-only
pytest -q tests/test_silu_and_mul.py --quick
```

### Run benchmark

```shell
cd /workspace/FlagGems-sglang
pytest -q benchmark --collect-only
pytest -q benchmark/test_silu_and_mul.py --level core --iter 1 --warmup 1
```

### Notes

- Most tests/benchmarks require a CUDA-capable GPU runtime.
- `--collect-only` is recommended first to quickly check import and discovery.
- All operator benchmarks support tensor-free case listing and exact replay;
  see [the case API guide](docs/benchmark_case_api.md).

This project is licensed under the [Apache (version 2.0) License](./LICENSE).

## MThreads BF16 MoE APIs

On MUSA/MP31, the public registrar exports:

- `fused_moe_bf16_gemm(a, b, out, topk_weights, sorted_token_ids, expert_ids,
  num_tokens_post_padded, config, *, top_k, mul_routed_weight=False,
  fuse_swiglu=False, tuning_profile=None)` consumes SGLang's padded routing.
  Inputs/weights/output are contiguous BF16, routing IDs are int32, and route
  weights are FP32. It writes and returns `out` without allocating framework
  workspaces. Fused W13 materializes BF16 gate/up before FP32 SwiGLU.
- `fused_shared_expert_gate_tail(hidden, gate_weight, shared_output, out=None,
  *, gate_logits_out=None, gate_prob_out=None, launch_options=None)` supports
  BF16 `[4, 2048]` activations and preserves BF16 logits/sigmoid boundaries.
- `deterministic_moe_combine(routed, gate_logits, shared, out, *,
  launch_options=None)` combines eight routes in
  order, then the shared expert, for BF16 H2048 and token counts
  2048/4096/6144/8192/16384. Input strides are supported; output is contiguous.

These APIs provide Triton computation only. The caller owns routing validity,
workspace lifetime, model/hardware selection and graph capture. Output aliasing
is rejected. Unsupported vendor implementations are not registered.

Vendor launch defaults live in
`src/flaggems_sglang/runtime/backend/_mthreads/tune_configs.yaml` and load once
per process. GEMM profiles (`m4_up`, `swiglu_up`, `swiglu_down`) apply only when
shape and baseline configuration match, and cannot change routing alignment.
Pass `tuning_profile=None` for the caller's explicit GEMM configuration;
shared/combine `launch_options` override their vendor defaults explicitly.

Tests use `reference.get_reference`; the MUSA GEMM reference calls upstream
SGLang's unfused BF16 GEMM and separately materializes SwiGLU. Benchmarks use
the same public APIs through `OpBenchmark`:

```shell
pytest -q tests/test_fused_moe_bf16_gemm.py tests/test_fused_shared_expert_gate_tail.py tests/test_deterministic_moe_combine.py
pytest -s benchmark/test_fused_moe_bf16_gemm.py benchmark/test_fused_shared_expert_gate_tail.py benchmark/test_deterministic_moe_combine.py --level core --warmup 25 --iter 100
```
