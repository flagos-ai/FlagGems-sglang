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

"""fused_gdn_gating baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.attention.fla.fused_gdn_gating.fused_gdn_gating
    调用形态复刻 problems/fla/fused_gdn_gating/baseline.py：
        fused_gdn_gating(A_log, a, b, dt_bias, beta=beta, threshold=threshold)
    返回 (g, beta_output)，另分配输出，不原地写回。

输入构造复刻 problems/fla/fused_gdn_gating/cases.py 的 BENCH_CASES：
    _case(m, 64) for m in (1, 8, 64, 512, 4096)
    A_log = randn(num_heads) * 0.5; a/b = randn(batch, num_heads);
    dt_bias = randn(num_heads)。beta/threshold 用 baseline 默认 (1.0 / 20.0)。
dtype 取 benchmark 的 [float32]。
"""

import torch

OP_NAME = "fused_gdn_gating"
DTYPES = [torch.float32]
IS_INPLACE = False

_MS = [1, 8, 64, 512, 4096]
_NUM_HEADS = 64


def native():
    """解析 sglang.kernels.ops.attention.fla.fused_gdn_gating.fused_gdn_gating。"""
    try:
        from sglang.kernels.ops.attention.fla.fused_gdn_gating import (
            fused_gdn_gating as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"batch": m, "num_heads": _NUM_HEADS} for m in _MS]


def build_inputs(binding, dtype, device):
    batch, num_heads = binding["batch"], binding["num_heads"]
    g = torch.Generator(device=device).manual_seed(0)
    a_log = torch.randn(num_heads, generator=g, device=device, dtype=torch.float32) * 0.5
    a = torch.randn(batch, num_heads, generator=g, device=device, dtype=torch.float32)
    b = torch.randn(batch, num_heads, generator=g, device=device, dtype=torch.float32)
    dt_bias = torch.randn(num_heads, generator=g, device=device, dtype=torch.float32)
    return (a_log, a, b, dt_bias), {"beta": 1.0, "threshold": 20.0}


def key_shape(binding):
    return [binding["batch"], binding["num_heads"]]


def config(binding, dtype):
    batch, num_heads = binding["batch"], binding["num_heads"]
    dt = str(dtype)
    return {
        "inputs": {
            "A_log": {"shape": [num_heads], "dtype": "torch.float32"},
            "a": {"shape": [batch, num_heads], "dtype": "torch.float32"},
            "b": {"shape": [batch, num_heads], "dtype": "torch.float32"},
            "dt_bias": {"shape": [num_heads], "dtype": "torch.float32"},
            "beta": {"scalar": 1.0},
            "threshold": {"scalar": 20.0},
        },
        "outputs": {
            "g": {"shape": [batch, num_heads], "dtype": "torch.float32"},
            "beta_output": {"shape": [batch, num_heads], "dtype": "torch.float32"},
        },
        "dims": {"batch": batch, "num_heads": num_heads},
    }
