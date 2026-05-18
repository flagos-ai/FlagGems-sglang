[中文版|[English](./README.md)]

## 介绍

FlagGems-sglang 是 [FlagOS](https://flagos.io/) 的一部分。
FlagGems-sglang是一个面向多种芯片后端的高性能算子库，它提供了常见sglang算子的高性能实现，支持多种常见模型的高性能推理及部署。

FlagGems-sglang 是一个使用 OpenAI 推出的[Triton 编程语言](https://github.com/openai/triton)实现的高性能深度学习算子库，

## 特性

- 算子已经过深度性能调优
- Triton kernel 调用优化
- 灵活的多后端支持机制
- 支持常见sglang算子（如 flashinfer 相关算子等）

## 快速安装

### 安装依赖

```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```
### 安装FlagGems-sglang
```shell
git clone https://github.com/flagos-ai/FlagGems-sglang.git
cd FlagGems-sglang
pip install  .
```

## 使用示例

```python
import torch
import flaggems_sglang

# 创建张量
x = torch.randn(1024, device='cuda')

# 应用 ReLU 激活函数
y = flaggems_sglang.ops.relu(x)
```

## Tests 与 Benchmark 快速使用

下面命令可用于安装后的快速检查。

### 运行 tests

```shell
cd /workspace/FlagGems-sglang
pytest -q tests --collect-only
pytest -q tests/test_outer.py --quick
```

### 运行 benchmark

```shell
cd /workspace/FlagGems-sglang
pytest -q benchmark --collect-only
pytest -q benchmark/test_outer.py::test_outer --level core --iter 1 --warmup 1
```

### 说明

- 大多数 tests/benchmark 需要 CUDA GPU 环境。
- 建议先执行 `--collect-only`，快速确认导入与用例发现是否正常。


本项目采用 [Apache (Version 2.0) License](./LICENSE) 授权许可。
