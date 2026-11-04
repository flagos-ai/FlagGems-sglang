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

"""causal_conv1d_fn baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.mamba.causal_conv1d_triton.causal_conv1d_fn
    调用形态复刻 problems/mamba/causal_conv1d_fn/baseline.py：
        在 build_inputs 内按 x/weight/seq_lens 构造 conv_states/cache_indices/
        has_initial_state，再以同名 kwargs 调 native。
    kernel 返回新张量；但其内部会把尾部窗口写回 conv_states（原地）。
    conv_states 为 build_inputs 现造的零张量、每次采样重建，故对外部输入无副作用；
    IS_INPLACE 标 True 以便采样前重建实参。

输入构造复刻 problems/mamba/causal_conv1d_fn/cases.py 的 BENCH_CASES：
    total = sum(seq_lens)
    x = randn(dim, total).to(dtype); weight = randn(dim, width).to(dtype)
    bias = randn(dim).to(dtype)
    query_start_loc = cumsum([0, *seq_lens])
dtype 取 benchmark/test_causal_conv1d_fn.py 的 [bfloat16]。
"""

import torch

OP_NAME = "causal_conv1d_fn"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 把尾窗写回 conv_states（现造缓冲），采样前需重建

# (seq_lens, dim, width)
_SHAPES = [
    ([2048] * 8, 4096, 4),
    ([512] * 32, 2048, 4),
]


def native():
    """解析 sglang.kernels.ops.mamba.causal_conv1d_triton.causal_conv1d_fn。"""
    try:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_fn as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"seq_lens": sl, "dim": dim, "width": width}
        for (sl, dim, width) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    seq_lens = list(binding["seq_lens"])
    dim = binding["dim"]
    width = binding["width"]
    total = sum(seq_lens)
    batch = len(seq_lens)
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(dim, total, generator=g, device=device, dtype=torch.float32).to(dtype)
    weight = torch.randn(dim, width, generator=g, device=device, dtype=torch.float32).to(dtype)
    bias = torch.randn(dim, generator=g, device=device, dtype=torch.float32).to(dtype)
    query_start_loc = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.cumsum(
        torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0
    )
    conv_states = torch.zeros(batch, dim, width - 1, dtype=x.dtype, device=x.device)
    cache_indices = torch.arange(batch, dtype=torch.int32, device=x.device)
    has_initial_state = torch.zeros(batch, dtype=torch.bool, device=x.device)
    return (x, weight, bias), {
        "conv_states": conv_states,
        "query_start_loc": query_start_loc,
        "seq_lens_cpu": seq_lens,
        "cache_indices": cache_indices,
        "has_initial_state": has_initial_state,
        "activation": "silu",
    }


def key_shape(binding):
    seq_lens = list(binding["seq_lens"])
    return (
        f"nseq{len(seq_lens)}_len{seq_lens[0]}_dim{binding['dim']}"
        f"_w{binding['width']}"
    )


def config(binding, dtype):
    seq_lens = list(binding["seq_lens"])
    dim = binding["dim"]
    width = binding["width"]
    total = sum(seq_lens)
    batch = len(seq_lens)
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [dim, total], "dtype": dt},
            "weight": {"shape": [dim, width], "dtype": dt},
            "bias": {"shape": [dim], "dtype": dt},
            "conv_states": {"shape": [batch, dim, width - 1], "dtype": dt,
                            "note": "原地更新（kernel 把尾窗写回）"},
            "query_start_loc": {"shape": [batch + 1], "dtype": "torch.int32"},
            "seq_lens_cpu": {"note": "seq_lens python list，长度 batch"},
            "cache_indices": {"shape": [batch], "dtype": "torch.int32"},
            "has_initial_state": {"shape": [batch], "dtype": "torch.bool"},
            "activation": {"scalar": "silu"},
        },
        "outputs": {
            "out": {"shape": [dim, total], "dtype": dt},
        },
        "dims": {
            "seq_lens": seq_lens, "batch": batch, "total": total,
            "dim": dim, "width": width,
        },
    }
