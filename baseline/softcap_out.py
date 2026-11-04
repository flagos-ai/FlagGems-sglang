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

"""softcap_out baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.activation.softcap.softcap_out
    调用形态复刻 problems/activation_norm/softcap_out/baseline.py：
        softcap_out(x, softcap_const)
    另分配输出返回（out 版本），不原地写回 -> IS_INPLACE=False。

输入构造复刻 problems/activation_norm/softcap_out/cases.py BENCH_CASES：
    _case(m, n) for m in (1,8,64,512) for n in (4096,32000,128256)
    x = (randn(m, n)*20).to(dtype)，softcap_const=30.0
dtype 取 benchmark/test_softcap_out.py 的 [bfloat16]。
"""

import torch

OP_NAME = "softcap_out"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SOFTCAP_CONST = 30.0
_SHAPES = [(m, n) for m in (1, 8, 64, 512) for n in (4096, 32000, 128256)]


def native():
    """解析 sglang.kernels.ops.activation.softcap.softcap_out。"""
    try:
        from sglang.kernels.ops.activation.softcap import softcap_out as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "n": n} for (m, n) in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(0)
    x = (torch.randn(m, n, generator=g, device=device, dtype=torch.float32) * 20).to(dtype)
    return (x, _SOFTCAP_CONST), {}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": dt},
            "softcap_const": {"scalar": _SOFTCAP_CONST},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": dt},
        },
        "dims": {"m": m, "n": n},
    }
