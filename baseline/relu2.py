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

"""relu2 baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.activation.activation.relu2
    调用形态复刻 problems/activation_norm/relu2/baseline.py：
        relu2(input)
    返回新张量，不原地写回 -> IS_INPLACE=False。

输入构造复刻该 problem 的 cases.py 的 BENCH_CASES：
    input = randn(bs, hidden).to(dtype)。dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "relu2"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SHAPES = [(bs, h) for bs in (1, 8, 64, 512, 4096) for h in (2048, 4096, 8192)]


def native():
    """解析 sglang.kernels.ops.activation.activation.relu2。"""
    try:
        from sglang.kernels.ops.activation.activation import relu2 as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs, "hidden": h} for (bs, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    bs, hidden = binding["bs"], binding["hidden"]
    g = torch.Generator(device=device).manual_seed(0)
    inp = torch.randn(bs, hidden, dtype=torch.float32, device=device, generator=g).to(dtype)
    return (inp,), {}


def key_shape(binding):
    return [binding["bs"], binding["hidden"]]


def config(binding, dtype):
    bs, hidden = binding["bs"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [bs, hidden], "dtype": dt},
        },
        "outputs": {
            "out": {"shape": [bs, hidden], "dtype": dt},
        },
        "dims": {"bs": bs, "hidden": hidden},
    }
