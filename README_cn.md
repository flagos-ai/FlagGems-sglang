[中文版|[English](./README.md)]

## 介绍

FlagDNN 是 [FlagOS](https://flagos.io/) 的一部分。
FlagDNN是一个面向多种芯片后端的深度神经网络计算库，它提供了常见深度学习算子的高性能实现，支持深度学习、计算机视觉、自然语言处理和人工智能等领域的高效计算。

FlagDNN 是一个使用 OpenAI 推出的[Triton 编程语言](https://github.com/openai/triton)实现的高性能深度学习算子库，

## 特性

- 算子已经过深度性能调优
- Triton kernel 调用优化
- 灵活的多后端支持机制
- 支持常见深度学习算子（如 ReLU 等）

## 快速安装

### 安装依赖

```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```
### 安装FlagDNN
```shell
git clone https://github.com/flagos-ai/FlagDNN.git
cd FlagDNN
pip install  .
```

## 使用示例

```python
import torch
import flag_dnn

# 创建张量
x = torch.randn(1024, device='cuda')

# 应用 ReLU 激活函数
y = flag_dnn.ops.relu(x)
```


本项目采用 [Apache (Version 2.0) License](./LICENSE) 授权许可。
