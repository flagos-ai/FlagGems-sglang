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

"""concat_and_cast_mha_k baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.kvcache.cache_ops.concat_and_cast_mha_k_triton
    调用形态复刻 problems/kvcache/concat_and_cast_mha_k/baseline.py：
        out = k.clone(); concat_and_cast_mha_k_triton(out, k_nope, k_rope)
    kernel 把 k_nope/k_rope 拼接写入 out（k）-> IS_INPLACE=True。

输入构造复刻 cases.py 的 BENCH_CASES：
    k = zeros(num_tokens, num_heads, nope_dim+rope_dim).bf16
    k_nope = randn(num_tokens, num_heads, nope_dim).bf16
    k_rope = randn(num_tokens, 1, rope_dim).bf16
    其中 num_heads=32, nope_dim=128, rope_dim=64，num_tokens in (1,8,64,512,4096)。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "concat_and_cast_mha_k"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 写 k（out）

_NUM_HEADS = 32
_NOPE = 128
_ROPE = 64
_TOKENS = [1, 8, 64, 512, 4096]


def native():
    """解析 concat_and_cast_mha_k_triton。"""
    try:
        from sglang.kernels.ops.kvcache.cache_ops import (
            concat_and_cast_mha_k_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"num_tokens": t, "num_heads": _NUM_HEADS, "nope_dim": _NOPE, "rope_dim": _ROPE}
        for t in _TOKENS
    ]


def build_inputs(binding, dtype, device):
    t, num_heads = binding["num_tokens"], binding["num_heads"]
    nope_dim, rope_dim = binding["nope_dim"], binding["rope_dim"]
    g = torch.Generator(device=device).manual_seed(0)
    k = torch.zeros(t, num_heads, nope_dim + rope_dim, dtype=dtype, device=device)
    k_nope = torch.randn(t, num_heads, nope_dim, device=device, generator=g).to(dtype)
    k_rope = torch.randn(t, 1, rope_dim, device=device, generator=g).to(dtype)
    return (k, k_nope, k_rope), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["num_heads"],
            binding["nope_dim"] + binding["rope_dim"]]


def config(binding, dtype):
    t, num_heads = binding["num_tokens"], binding["num_heads"]
    nope_dim, rope_dim = binding["nope_dim"], binding["rope_dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "k": {"shape": [t, num_heads, nope_dim + rope_dim], "dtype": dt,
                  "note": "原地写回"},
            "k_nope": {"shape": [t, num_heads, nope_dim], "dtype": dt},
            "k_rope": {"shape": [t, 1, rope_dim], "dtype": dt},
        },
        "outputs": {
            "k": {"shape": [t, num_heads, nope_dim + rope_dim], "dtype": dt},
        },
        "dims": {
            "num_tokens": t, "num_heads": num_heads,
            "nope_dim": nope_dim, "rope_dim": rope_dim,
        },
    }
