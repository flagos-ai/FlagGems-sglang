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

"""silu_and_mul baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.elementwise.elementwise.silu_and_mul_triton
    调用形态复刻 problems/activation_norm/silu_and_mul/baseline.py：
        out, _ = silu_and_mul_triton(hidden_states)
    从单张 [bs, 2*d] 张量读 x1/x3，输出 [bs, d]；另分配输出，不原地写回。

输入构造复刻 problems/activation_norm/silu_and_mul/cases.py 的 BENCH_CASES：
    hidden_states = randn(bs, 2*d).to(dtype)
dtype 取 benchmark/test_silu_and_mul.py 的 [bfloat16]。
"""

import torch

OP_NAME = "silu_and_mul"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SHAPES = [(bs, d) for bs in (1, 8, 64, 512, 4096) for d in (1024, 4096, 8192)]


def native():
    """解析 sglang.kernels.ops.elementwise.elementwise.silu_and_mul_triton。"""
    try:
        from sglang.kernels.ops.elementwise.elementwise import (
            silu_and_mul_triton as op,
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
    # hidden_states 为单张 [bs, 2*d]，前半 x1 后半 x3；silu(x1)*x3 -> [bs, d]。
    return {
        "inputs": {
            "hidden_states": {"shape": [bs, 2 * d], "dtype": dt,
                              "note": "前半 x1 / 后半 x3"},
        },
        "outputs": {
            "out": {"shape": [bs, d], "dtype": dt},
        },
        "dims": {"bs": bs, "d": d, "hidden": 2 * d},
    }
