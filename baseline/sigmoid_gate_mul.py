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

"""sigmoid_gate_mul baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.moe.triton_sigmoid_gate_mul.sigmoid_gate_mul
    调用形态复刻 problems/elementwise/sigmoid_gate_mul/baseline.py：
        sigmoid_gate_mul(x, gate)
    返回新张量，不原地写回 -> IS_INPLACE=False。
    （该算子 problem 归在 elementwise 组，但 native 符号在 moe 包下。）

输入构造复刻该 problem 的 cases.py 的 BENCH_CASES：
    x/gate = randn(bs, hidden).to(bf16)（种子 seed/seed+1）。
    dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "sigmoid_gate_mul"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SHAPES = [(bs, h) for bs in (1, 8, 64, 512, 4096) for h in (1024, 4096, 8192)]


def native():
    """解析 moe.triton_sigmoid_gate_mul.sigmoid_gate_mul。"""
    try:
        from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import (
            sigmoid_gate_mul as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs, "hidden": h} for (bs, h) in _SHAPES]


def _rand(shape, dtype, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32, device=device, generator=g).to(dtype)


def build_inputs(binding, dtype, device):
    bs, hidden = binding["bs"], binding["hidden"]
    x = _rand((bs, hidden), dtype, device, 0)
    gate = _rand((bs, hidden), dtype, device, 1)
    return (x, gate), {}


def key_shape(binding):
    return [binding["bs"], binding["hidden"]]


def config(binding, dtype):
    bs, hidden = binding["bs"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [bs, hidden], "dtype": dt},
            "gate": {"shape": [bs, hidden], "dtype": dt},
        },
        "outputs": {
            "out": {"shape": [bs, hidden], "dtype": dt},
        },
        "dims": {"bs": bs, "hidden": hidden},
    }
