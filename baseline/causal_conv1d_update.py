# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""causal_conv1d_update baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.mamba.causal_conv1d_triton.causal_conv1d_update
    调用形态复刻 problems/mamba/causal_conv1d_update/baseline.py：
        causal_conv1d_update(x, conv_state, weight, bias, activation)
    kernel 原地更新 conv_state（baseline 先 clone 再传）。此处每次采样重建
    conv_state，故对外部无副作用；IS_INPLACE 标 True 以便采样前重建实参。

输入构造复刻 problems/mamba/causal_conv1d_update/cases.py 的 BENCH_CASES：
    seqlen=1 -> x = randn(batch, dim, 1).squeeze(-1).to(dtype)
    conv_state = randn(batch, dim, width-1).to(dtype)
    weight = randn(dim, width).to(dtype); bias = randn(dim).to(dtype)
dtype 取 benchmark/test_causal_conv1d_update.py 的 [bfloat16]。
"""

import torch

OP_NAME = "causal_conv1d_update"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 原地更新 conv_state，采样前需重建

# (batch, dim, width, seqlen)
_SHAPES = [
    (64, 4096, 4, 1),
    (256, 2048, 4, 1),
]


def native():
    """解析 sglang.kernels.ops.mamba.causal_conv1d_triton.causal_conv1d_update。"""
    try:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_update as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": b, "dim": dim, "width": width, "seqlen": seqlen}
        for (b, dim, width, seqlen) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    batch = binding["batch"]
    dim = binding["dim"]
    width = binding["width"]
    seqlen = binding["seqlen"]
    state_len = width - 1
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(batch, dim, seqlen, generator=g, device=device, dtype=torch.float32).to(
        dtype
    )
    if seqlen == 1:
        x = x.squeeze(-1)
    conv_state = torch.randn(
        batch, dim, state_len, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    weight = torch.randn(dim, width, generator=g, device=device, dtype=torch.float32).to(dtype)
    bias = torch.randn(dim, generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x, conv_state, weight, bias, "silu"), {}


def key_shape(binding):
    return (
        f"b{binding['batch']}_dim{binding['dim']}_w{binding['width']}"
        f"_sq{binding['seqlen']}"
    )


def config(binding, dtype):
    batch = binding["batch"]
    dim = binding["dim"]
    width = binding["width"]
    seqlen = binding["seqlen"]
    state_len = width - 1
    dt = str(dtype)
    # seqlen==1 时 x 被 squeeze 成 (batch, dim)，否则 (batch, dim, seqlen)。
    x_shape = [batch, dim] if seqlen == 1 else [batch, dim, seqlen]
    return {
        "inputs": {
            "x": {"shape": x_shape, "dtype": dt},
            "conv_state": {"shape": [batch, dim, state_len], "dtype": dt,
                           "note": "原地更新"},
            "weight": {"shape": [dim, width], "dtype": dt},
            "bias": {"shape": [dim], "dtype": dt},
            "activation": {"scalar": "silu"},
        },
        "outputs": {
            "out": {"shape": x_shape, "dtype": dt},
        },
        "dims": {
            "batch": batch, "dim": dim, "width": width, "seqlen": seqlen,
            "state_len": state_len,
        },
    }
