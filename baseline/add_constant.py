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

"""add_constant baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.elementwise.add_constant.add_constant
    调用形态复刻 problems/elementwise/add_constant/baseline.py：
        add_constant(src, constant)
    返回新张量，不原地写回 -> IS_INPLACE=False。

输入构造复刻该 problem 的 cases.py 的 BENCH_CASES：
    src = randint(-2**20, 2**20, (n,), int32)；constant = 42（python int）。
    这是整型算子，DTYPES 取 int32（与 cases.py 构造一致，dtype 参数不影响构造）。
"""

import torch

OP_NAME = "add_constant"
DTYPES = [torch.int32]
IS_INPLACE = False

_CONST = 42
_SIZES = [1 << 12, 1 << 16, 1 << 20, 1 << 24]


def native():
    """解析 sglang.kernels.ops.elementwise.add_constant.add_constant。"""
    try:
        from sglang.kernels.ops.elementwise.add_constant import add_constant as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"n": n} for n in _SIZES]


def build_inputs(binding, dtype, device):
    n = binding["n"]
    g = torch.Generator(device=device).manual_seed(0)
    src = torch.randint(
        -(2 ** 20), 2 ** 20, (n,), dtype=torch.int32, device=device, generator=g
    )
    return (src, _CONST), {}


def key_shape(binding):
    return [binding["n"]]


def config(binding, dtype):
    n = binding["n"]
    return {
        "inputs": {
            "src": {"shape": [n], "dtype": "torch.int32"},
            "constant": {"scalar": _CONST},
        },
        "outputs": {
            "out": {"shape": [n], "dtype": "torch.int32"},
        },
        "dims": {"n": n},
    }
