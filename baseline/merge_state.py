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

"""merge_state baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.attention.merge_state.merge_state_triton
    调用形态复刻 problems/attention/merge_state/baseline.py：
        merge_state_triton(prefix_output, prefix_lse, suffix_output, suffix_lse)
    output / output_lse 省略 -> kernel 内 empty_like 另分配，不原地写回。

输入构造复刻 problems/attention/merge_state/cases.py 的 BENCH_CASES：
    _case(n, 32, 128) for n in (1, 8, 64, 512, 4096)
    prefix/suffix_output = randn(n, H, Dh).to(dtype)；
    prefix/suffix_lse = randn(n, H) * 3（float32）。
dtype 取 benchmark 的 [bfloat16]。
"""

import torch

OP_NAME = "merge_state"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # output / output_lse 省略，kernel 内另分配

_NUM_HEADS = 32
_HEAD_SIZE = 128
_NS = [1, 8, 64, 512, 4096]


def native():
    """解析 sglang.kernels.ops.attention.merge_state.merge_state_triton。"""
    try:
        from sglang.kernels.ops.attention.merge_state import (
            merge_state_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"n_tokens": n, "num_heads": _NUM_HEADS, "head_size": _HEAD_SIZE}
        for n in _NS
    ]


def build_inputs(binding, dtype, device):
    n, num_heads, head_size = (
        binding["n_tokens"],
        binding["num_heads"],
        binding["head_size"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    prefix_output = torch.randn(
        n, num_heads, head_size, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    suffix_output = torch.randn(
        n, num_heads, head_size, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    prefix_lse = torch.randn(n, num_heads, generator=g, device=device) * 3
    suffix_lse = torch.randn(n, num_heads, generator=g, device=device) * 3
    return (prefix_output, prefix_lse, suffix_output, suffix_lse), {}


def key_shape(binding):
    return [binding["n_tokens"], binding["num_heads"], binding["head_size"]]


def config(binding, dtype):
    n, num_heads, head_size = (
        binding["n_tokens"],
        binding["num_heads"],
        binding["head_size"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "prefix_output": {"shape": [n, num_heads, head_size], "dtype": dt},
            "prefix_lse": {"shape": [n, num_heads], "dtype": "torch.float32"},
            "suffix_output": {"shape": [n, num_heads, head_size], "dtype": dt},
            "suffix_lse": {"shape": [n, num_heads], "dtype": "torch.float32"},
        },
        "outputs": {
            "output": {"shape": [n, num_heads, head_size], "dtype": dt},
            "output_lse": {"shape": [n, num_heads], "dtype": "torch.float32"},
        },
        "dims": {"n_tokens": n, "num_heads": num_heads, "head_size": head_size},
    }
