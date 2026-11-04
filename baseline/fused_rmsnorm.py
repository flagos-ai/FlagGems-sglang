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

"""fused_rmsnorm baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.elementwise.elementwise.fused_rmsnorm
    调用形态复刻 problems/activation_norm/fused_rmsnorm/baseline.py：
        fused_rmsnorm(x, weight, eps, autotune=False, inplace=False)
    inplace=False -> 另分配输出，不原地写回 x。

输入构造复刻 problems/activation_norm/fused_rmsnorm/cases.py 的 BENCH_CASES：
    x = randn(bs, hidden).to(dtype); weight = randn(hidden).to(dtype); eps=1e-6
dtype 取 benchmark/test_fused_rmsnorm.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_rmsnorm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # inplace=False，另分配输出

_EPS = 1e-6
_SHAPES = [(bs, h) for bs in (1, 8, 64, 512, 4096) for h in (1024, 4096, 8192)]


def native():
    """解析 sglang.kernels.ops.elementwise.elementwise.fused_rmsnorm。"""
    try:
        from sglang.kernels.ops.elementwise.elementwise import fused_rmsnorm as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs, "hidden": h} for (bs, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    bs, hidden = binding["bs"], binding["hidden"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(bs, hidden, dtype=torch.float32, device=device, generator=g).to(dtype)
    weight = torch.randn(hidden, dtype=torch.float32, device=device, generator=g).to(dtype)
    return (x, weight, _EPS), {"autotune": False, "inplace": False}


def key_shape(binding):
    return [binding["bs"], binding["hidden"]]


def config(binding, dtype):
    bs, hidden = binding["bs"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [bs, hidden], "dtype": dt},
            "weight": {"shape": [hidden], "dtype": dt},
            "eps": {"scalar": _EPS},
            "autotune": {"scalar": False},
            "inplace": {"scalar": False},
        },
        "outputs": {
            "out": {"shape": [bs, hidden], "dtype": dt},
        },
        "dims": {"bs": bs, "hidden": hidden},
    }
