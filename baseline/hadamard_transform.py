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

"""hadamard_transform baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.hadamard.hadamard_transform
    调用形态复刻 problems/quantization/hadamard_transform/baseline.py：
        hadamard_transform(x, scale)
    返回新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    x=randn(m,n).to(bf16)；scale=1.0。
BENCH: _case(m,n,seed=9) for m in (1,8,64,512,4096) for n in (128,512,2048)。
"""

import torch

OP_NAME = "hadamard_transform"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SCALE = 1.0
_SEED = 9
_SHAPES = [
    {"m": m, "n": n}
    for m in (1, 8, 64, 512, 4096)
    for n in (128, 512, 2048)
]


def native():
    """解析 hadamard_transform。"""
    try:
        from sglang.kernels.ops.quantization.hadamard import hadamard_transform as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(m, n, device=device, generator=g).to(dtype)
    return (x, _SCALE), {}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": dt},
            "scale": {"scalar": _SCALE},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": dt},
        },
        "dims": {"m": m, "n": n},
    }
