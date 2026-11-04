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

"""l2norm baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.attention.fla.l2norm.l2norm_fwd
    调用形态复刻 problems/activation_norm/l2norm/baseline.py：
        l2norm_fwd(x, eps)   # eps 默认 1e-6
    另分配输出返回，不原地写回 -> IS_INPLACE=False。

输入构造复刻 problems/activation_norm/l2norm/cases.py BENCH_CASES：
    _case(t, d) for t in (1,32,512,4096) for d in (64,128,256)
    x = randn(t, d).to(dtype)
dtype 取 benchmark/test_l2norm.py 的 [bfloat16]。
"""

import torch

OP_NAME = "l2norm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-6
_SHAPES = [(t, d) for t in (1, 32, 512, 4096) for d in (64, 128, 256)]


def native():
    """解析 sglang.kernels.ops.attention.fla.l2norm.l2norm_fwd。"""
    try:
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"t": t, "d": d} for (t, d) in _SHAPES]


def build_inputs(binding, dtype, device):
    t, d = binding["t"], binding["d"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(t, d, generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x, _EPS), {}


def key_shape(binding):
    return [binding["t"], binding["d"]]


def config(binding, dtype):
    t, d = binding["t"], binding["d"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [t, d], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [t, d], "dtype": dt},
        },
        "dims": {"t": t, "d": d},
    }
