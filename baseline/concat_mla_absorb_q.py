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

"""concat_mla_absorb_q baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.concat_mla.concat_mla_absorb_q
    调用形态复刻 problems/attention/concat_mla_absorb_q/baseline.py：
        concat_mla_absorb_q(a, b)
    返回新张量（a/b 沿最后一维拼接），不原地写回；IS_INPLACE=False。

输入构造复刻 problems/attention/concat_mla_absorb_q/cases.py 的 BENCH_CASES：
    a = randn(num_tokens, num_heads, a_last).bf16；b = randn(num_tokens, num_heads, b_last).bf16
    num_heads=128, a_last=512, b_last=64；a seed=0，b seed=1。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "concat_mla_absorb_q"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_NUM_HEADS = 128
_A_LAST = 512
_B_LAST = 64
_CASES = [1, 8, 64, 512, 4096]


def native():
    """解析 sglang.kernels.ops.attention.concat_mla.concat_mla_absorb_q。"""
    try:
        from sglang.kernels.ops.attention.concat_mla import concat_mla_absorb_q as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"num_tokens": t, "num_heads": _NUM_HEADS, "a_last": _A_LAST, "b_last": _B_LAST}
        for t in _CASES
    ]


def build_inputs(binding, dtype, device):
    t = binding["num_tokens"]
    nh, a_last, b_last = binding["num_heads"], binding["a_last"], binding["b_last"]
    ga = torch.Generator(device=device).manual_seed(0)
    gb = torch.Generator(device=device).manual_seed(1)
    a = torch.randn(t, nh, a_last, dtype=torch.float32, device=device, generator=ga).to(dtype)
    b = torch.randn(t, nh, b_last, dtype=torch.float32, device=device, generator=gb).to(dtype)
    return (a, b), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["num_heads"],
            binding["a_last"], binding["b_last"]]


def config(binding, dtype):
    t = binding["num_tokens"]
    nh, a_last, b_last = binding["num_heads"], binding["a_last"], binding["b_last"]
    dt = str(dtype)
    return {
        "inputs": {
            "a": {"shape": [t, nh, a_last], "dtype": dt},
            "b": {"shape": [t, nh, b_last], "dtype": dt},
        },
        "outputs": {
            "out": {"shape": [t, nh, a_last + b_last], "dtype": dt},
        },
        "dims": {"num_tokens": t, "num_heads": nh, "a_last": a_last, "b_last": b_last},
    }
