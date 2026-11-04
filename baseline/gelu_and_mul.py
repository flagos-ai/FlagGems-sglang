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

"""gelu_and_mul baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.elementwise.elementwise.gelu_and_mul_triton
    调用形态复刻 problems/activation_norm/gelu_and_mul/baseline.py：
        out, _ = gelu_and_mul_triton(hidden_states)
    从单张 [bs, 2*d] 张量读 x1/x3，另分配输出 [bs, d]，不原地写回 ->
    IS_INPLACE=False。

输入构造复刻 problems/activation_norm/gelu_and_mul/cases.py BENCH_CASES
（复用 silu_and_mul.cases._x）：hidden_states = randn(bs, 2*d).to(dtype)。
dtype 取 benchmark/test_gelu_and_mul.py 的 [bfloat16]。
"""

import torch

OP_NAME = "gelu_and_mul"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SHAPES = [(bs, d) for bs in (1, 8, 64, 512, 4096) for d in (1024, 4096, 8192)]


def native():
    """解析 sglang.kernels.ops.elementwise.elementwise.gelu_and_mul_triton。"""
    try:
        from sglang.kernels.ops.elementwise.elementwise import (
            gelu_and_mul_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs, "d": d} for (bs, d) in _SHAPES]


def build_inputs(binding, dtype, device):
    bs, d = binding["bs"], binding["d"]
    g = torch.Generator(device=device).manual_seed(0)
    hidden_states = torch.randn(
        bs, 2 * d, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    return (hidden_states,), {}


def key_shape(binding):
    return [binding["bs"], 2 * binding["d"]]


def config(binding, dtype):
    bs, d = binding["bs"], binding["d"]
    dt = str(dtype)
    return {
        "inputs": {
            "hidden_states": {"shape": [bs, 2 * d], "dtype": dt,
                              "note": "前半 gate / 后半 up"},
        },
        "outputs": {
            "out": {"shape": [bs, d], "dtype": dt},
        },
        "dims": {"bs": bs, "d": d},
    }
