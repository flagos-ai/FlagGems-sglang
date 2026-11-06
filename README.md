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

## Active-head sparse MLA

`flaggems_sglang.sparse_mla_active_split` is an explicit inference operator for
SM90 BF16 query `[SQ, 64, 576]`, `SQ=1|4`, four active heads, BF16 KV
`[slots, 1, 576]`, INT32 physical indices `[SQ, 1, 2048]`, FP32 sink `[64]`,
and INT32 selected-position lengths `[SQ]`. Inputs require contiguous strides,
zero storage offset, a matching CUDA device, and no gradient or conjugate/negative
view state. Unsupported metadata raises `NotImplementedError` before launch;
callers select native fallback before invocation. Launch errors propagate.

The 2048 selected positions are partitioned without reselection; duplicate slots
retain their multiplicity. Invalid slots and positions beyond each length do not
contribute. The sink enters the denominator once. The result is `(output, maximum,
lse)` with shapes `[SQ, 4, 512]`, `[SQ, 4]`, `[SQ, 4]`. Statistics describe KV-only
logits. Empty rows return zero output, maximum `-inf`, and LSE `+inf`; a positive
infinite sink produces zero output. Qualified inputs are finite.

The module's `metadata` helper checks eligibility and `workspace(q, 4)` allocates
optional FP32 partial buffers. The caller owns each input, workspace, and output
through the last graph replay. Partial buffers require independent storage.
Registration does not select this operation for a serving model. Run the focused
checks with `pytest -q tests/test_sparse_mla_active_split.py` and the component
benchmark with `pytest -q benchmark/test_sparse_mla_active_split.py --level core`.
