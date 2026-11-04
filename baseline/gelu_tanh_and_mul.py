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

"""gelu_tanh_and_mul baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.activation.activation.gelu_tanh_and_mul
    调用形态复刻 problems/activation_norm/gelu_tanh_and_mul/baseline.py：
        gelu_tanh_and_mul(input)
    从单张 [bs, 2*d] 张量读 gate/up，另分配输出 [bs, d]，不原地写回 ->
    IS_INPLACE=False。

输入构造复刻 problems/activation_norm/gelu_tanh_and_mul/cases.py 的 BENCH_CASES：
    input = randn(bs, 2*d).to(dtype)。dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "gelu_tanh_and_mul"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SHAPES = [(bs, d) for bs in (1, 8, 64, 512, 4096) for d in (1024, 4096, 8192)]


def native():
    """解析 sglang.kernels.ops.activation.activation.gelu_tanh_and_mul。"""
    try:
        from sglang.kernels.ops.activation.activation import gelu_tanh_and_mul as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs, "d": d} for (bs, d) in _SHAPES]


def build_inputs(binding, dtype, device):
    bs, d = binding["bs"], binding["d"]
    g = torch.Generator(device=device).manual_seed(0)
    inp = torch.randn(bs, 2 * d, dtype=torch.float32, device=device, generator=g).to(dtype)
    return (inp,), {}


def key_shape(binding):
    return [binding["bs"], 2 * binding["d"]]


def config(binding, dtype):
    bs, d = binding["bs"], binding["d"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [bs, 2 * d], "dtype": dt,
                      "note": "前半 gate / 后半 up"},
        },
        "outputs": {
            "out": {"shape": [bs, d], "dtype": dt},
        },
        "dims": {"bs": bs, "d": d},
    }
