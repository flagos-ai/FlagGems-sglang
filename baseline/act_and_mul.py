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

"""act_and_mul baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.moe.fused_moe_triton_kernels.act_and_mul_triton
    调用形态复刻 problems/moe/act_and_mul/baseline.py：
        down_input = empty(m, hidden//2)
        act_and_mul_triton(gateup_output, down_input, config={},
                           activation=activation, swiglu_limit=swiglu_limit)
    从 [m, 2*half] 读 gate/up，写入新分配的 down_input [m, half]；
    不原地写回输入 gateup_output -> IS_INPLACE=False。

输入构造复刻 problems/moe/act_and_mul/cases.py 的 BENCH_CASES：
    gateup_output = randn(m, half*2).to(dtype)，activation="silu"，swiglu_limit=None
dtype 取 benchmark/test_act_and_mul.py 的 [bfloat16]。
"""

import torch

OP_NAME = "act_and_mul"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # 写入另分配的 down_input，不改输入

# BENCH_CASES：_case(m, 4096, "silu") for m in (1,8,64,512,4096)；half_hidden=4096。
_SHAPES = [(m, 4096) for m in (1, 8, 64, 512, 4096)]
_ACTIVATION = "silu"
_SWIGLU_LIMIT = None


def native():
    """解析 sglang.kernels.ops.moe.fused_moe_triton_kernels.act_and_mul_triton。"""
    try:
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            act_and_mul_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "half_hidden": h} for (m, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    m, half_hidden = binding["m"], binding["half_hidden"]
    g = torch.Generator(device=device).manual_seed(0)
    gateup_output = torch.randn(
        m, half_hidden * 2, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    down_input = torch.empty(m, half_hidden, dtype=dtype, device=device)
    args = (gateup_output, down_input)
    kwargs = {"config": {}, "activation": _ACTIVATION, "swiglu_limit": _SWIGLU_LIMIT}
    return args, kwargs


def key_shape(binding):
    return [binding["m"], 2 * binding["half_hidden"]]


def config(binding, dtype):
    m, half_hidden = binding["m"], binding["half_hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "gateup_output": {"shape": [m, 2 * half_hidden], "dtype": dt,
                              "note": "前半 gate / 后半 up"},
            "down_input": {"shape": [m, half_hidden], "dtype": dt,
                           "note": "原地写回（另分配的输出缓冲）"},
            "config": {"scalar": {}},
            "activation": {"scalar": _ACTIVATION},
            "swiglu_limit": {"scalar": _SWIGLU_LIMIT},
        },
        "outputs": {
            "down_input": {"shape": [m, half_hidden], "dtype": dt},
        },
        "dims": {"m": m, "half_hidden": half_hidden},
    }
