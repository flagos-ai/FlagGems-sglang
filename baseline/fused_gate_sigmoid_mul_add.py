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

"""fused_gate_sigmoid_mul_add baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.elementwise.elementwise.fused_gate_sigmoid_mul_add
    调用形态复刻 problems/activation_norm/fused_gate_sigmoid_mul_add/baseline.py：
        out = final_hidden_states.clone()
        fused_gate_sigmoid_mul_add(hidden_states, gate_weight, shared_output, out)
    kernel 把结果原地写入第 4 个实参（out）-> IS_INPLACE=True。采集侧每次
    build_inputs 新建 out，等价于那次 clone；NCU 采样前重建实参避免 warmup 污染。

输入构造复刻 problems/activation_norm/fused_gate_sigmoid_mul_add/cases.py 的 BENCH_CASES：
    hidden_states/shared_output/final_hidden_states = randn(bs, hidden).to(dtype)
    gate_weight = randn(hidden).to(dtype)。dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_gate_sigmoid_mul_add"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 原地写入 out 实参

_SHAPES = [(bs, h) for bs in (1, 8, 64, 512, 4096) for h in (2048, 4096, 7168)]


def native():
    """解析 sglang.kernels.ops.elementwise.elementwise.fused_gate_sigmoid_mul_add。"""
    try:
        from sglang.kernels.ops.elementwise.elementwise import (
            fused_gate_sigmoid_mul_add as op,
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
    hidden_states = _rand((bs, hidden), dtype, device, 0)
    gate_weight = _rand((hidden,), dtype, device, 1)
    shared_output = _rand((bs, hidden), dtype, device, 2)
    out = _rand((bs, hidden), dtype, device, 3)  # = final_hidden_states.clone()
    return (hidden_states, gate_weight, shared_output, out), {}


def key_shape(binding):
    return [binding["bs"], binding["hidden"]]


def config(binding, dtype):
    bs, hidden = binding["bs"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "hidden_states": {"shape": [bs, hidden], "dtype": dt},
            "gate_weight": {"shape": [hidden], "dtype": dt},
            "shared_output": {"shape": [bs, hidden], "dtype": dt},
            "out": {"shape": [bs, hidden], "dtype": dt, "note": "原地写回"},
        },
        "outputs": {
            "out": {"shape": [bs, hidden], "dtype": dt},
        },
        "dims": {"bs": bs, "hidden": hidden},
    }
