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

"""rmsnorm_hf baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.layernorm.rmsnorm_hf.rmsnorm_hf
    调用形态复刻 problems/activation_norm/rmsnorm_hf/baseline.py：
        rmsnorm_hf(input, weight, eps)
    返回新张量，不原地写回 -> IS_INPLACE=False。

输入构造复刻该 problem 的 cases.py 的 BENCH_CASES：
    input = randn(bs, hidden).to(dtype)（种子 seed）
    weight = randn(hidden).to(dtype)（种子 seed+1），eps=1e-6。
    dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "rmsnorm_hf"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-6
_SHAPES = [(bs, h) for bs in (1, 8, 64, 512, 4096) for h in (128, 4096, 7168)]


def native():
    """解析 sglang.kernels.ops.layernorm.rmsnorm_hf.rmsnorm_hf。"""
    try:
        from sglang.kernels.ops.layernorm.rmsnorm_hf import rmsnorm_hf as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs, "hidden": h} for (bs, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    bs, hidden = binding["bs"], binding["hidden"]
    g0 = torch.Generator(device=device).manual_seed(0)
    inp = torch.randn(bs, hidden, dtype=torch.float32, device=device, generator=g0).to(dtype)
    g1 = torch.Generator(device=device).manual_seed(1)
    weight = torch.randn(hidden, dtype=torch.float32, device=device, generator=g1).to(dtype)
    return (inp, weight, _EPS), {}


def key_shape(binding):
    return [binding["bs"], binding["hidden"]]


def config(binding, dtype):
    bs, hidden = binding["bs"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [bs, hidden], "dtype": dt},
            "weight": {"shape": [hidden], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [bs, hidden], "dtype": dt},
        },
        "dims": {"bs": bs, "hidden": hidden},
    }
