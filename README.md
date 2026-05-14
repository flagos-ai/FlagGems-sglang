[English|[中文版](./README_cn.md)]

## Introduction

FlagDNN is part of [FlagOS](https://flagos.io/).
FlagDNN is a deep neural network computing library oriented towards multiple chip backends. It provides high-performance implementations of common deep learning operators, supporting efficient computation in fields such as deep learning, computer vision, natural language processing, and artificial intelligence.

FlagDNN is a high-performance deep learning operator library implemented using the [Triton programming language](https://github.com/openai/triton) launched by OpenAI.

## Features

- Operators have undergone deep performance tuning
- Triton kernel call optimization
- Flexible multi-backend support mechanism
- Support for common deep learning operators (ReLU, etc.)

## Quick Installation
### Install Dependencies
```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```
### Install FlagDNN
```shell
git clone https://github.com/flagos-ai/FlagDNN.git
cd FlagDNN
pip install  .
```

## Usage Example

```python
import torch
import flag_dnn

# Create a tensor
x = torch.randn(1024, device='cuda')

# Apply ReLU activation
y = flag_dnn.ops.relu(x)
```

This project is licensed under the [Apache (version 2.0) License](./LICENSE).
