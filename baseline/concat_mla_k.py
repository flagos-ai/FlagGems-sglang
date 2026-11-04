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

"""concat_mla_k baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.concat_mla.concat_mla_k
    调用形态复刻 problems/attention/concat_mla_k/baseline.py：
        out = k.clone(); concat_mla_k(out, k_nope, k_rope)
    kernel 把 k_nope / k_rope 原地写入传入的 out（故 baseline 先 clone k）。
    采集侧每次 build_inputs 新建 out，被写回的是本地新张量 -> 对外无副作用，
    但 IS_INPLACE=True（第一个实参被原地写回，NCU 采样前应重建实参）。

输入构造复刻 problems/attention/concat_mla_k/cases.py 的 BENCH_CASES：
    k = randn(num_tokens, num_heads, nope_dim+rope_dim).bf16（seed=0，充当 out 基底）
    k_nope = randn(num_tokens, num_heads, nope_dim).bf16（seed=1）
    k_rope = randn(num_tokens, 1, rope_dim).bf16（seed=2）
    num_heads=128, nope_dim=128, rope_dim=64。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "concat_mla_k"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 原地写入 out（第一个实参）

_NUM_HEADS = 128
_NOPE_DIM = 128
_ROPE_DIM = 64
_CASES = [1, 8, 64, 512, 4096]


def native():
    """解析 sglang.kernels.ops.attention.concat_mla.concat_mla_k。"""
    try:
        from sglang.kernels.ops.attention.concat_mla import concat_mla_k as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"num_tokens": t, "num_heads": _NUM_HEADS,
         "nope_dim": _NOPE_DIM, "rope_dim": _ROPE_DIM}
        for t in _CASES
    ]


def build_inputs(binding, dtype, device):
    t = binding["num_tokens"]
    nh, nope, rope = binding["num_heads"], binding["nope_dim"], binding["rope_dim"]
    g0 = torch.Generator(device=device).manual_seed(0)
    g1 = torch.Generator(device=device).manual_seed(1)
    g2 = torch.Generator(device=device).manual_seed(2)
    k = torch.randn(t, nh, nope + rope, dtype=torch.float32, device=device, generator=g0).to(dtype)
    k_nope = torch.randn(t, nh, nope, dtype=torch.float32, device=device, generator=g1).to(dtype)
    k_rope = torch.randn(t, 1, rope, dtype=torch.float32, device=device, generator=g2).to(dtype)
    out = k.clone()
    return (out, k_nope, k_rope), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["num_heads"],
            binding["nope_dim"], binding["rope_dim"]]


def config(binding, dtype):
    t = binding["num_tokens"]
    nh, nope, rope = binding["num_heads"], binding["nope_dim"], binding["rope_dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "out": {"shape": [t, nh, nope + rope], "dtype": dt, "note": "原地写回"},
            "k_nope": {"shape": [t, nh, nope], "dtype": dt},
            "k_rope": {"shape": [t, 1, rope], "dtype": dt},
        },
        "outputs": {
            "out": {"shape": [t, nh, nope + rope], "dtype": dt},
        },
        "dims": {"num_tokens": t, "num_heads": nh, "nope_dim": nope, "rope_dim": rope},
    }
