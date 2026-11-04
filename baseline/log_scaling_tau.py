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

"""log_scaling_tau baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.attention.log_scaling_tau.apply_log_scaling_tau
    调用形态复刻 problems/attention/log_scaling_tau/baseline.py：
        apply_log_scaling_tau(x, tau)
    kernel 另分配 out 返回（out = empty(x.shape)），不原地写回 x。

输入构造复刻 problems/attention/log_scaling_tau/cases.py 的 BENCH_CASES：
    _case((t, 16, 128)) for t in (1, 128, 2048, 8192)
    x = randn(t, 16, 128).to(dtype); tau = 0.5 + rand(t)（per-row）。
dtype 取 benchmark 的 [float16]。
"""

import torch

OP_NAME = "log_scaling_tau"
DTYPES = [torch.float16]
IS_INPLACE = False  # kernel 另分配 out 返回

_HEADS = 16
_D = 128
_TS = [1, 128, 2048, 8192]


def native():
    """解析 sglang.kernels.ops.attention.log_scaling_tau.apply_log_scaling_tau。"""
    try:
        from sglang.kernels.ops.attention.log_scaling_tau import (
            apply_log_scaling_tau as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"t": t, "heads": _HEADS, "d": _D} for t in _TS]


def build_inputs(binding, dtype, device):
    t, heads, d = binding["t"], binding["heads"], binding["d"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(t, heads, d, generator=g, device=device, dtype=torch.float32).to(dtype)
    tau = 0.5 + torch.rand(t, generator=g, device=device, dtype=torch.float32)
    return (x, tau), {}


def key_shape(binding):
    return [binding["t"], binding["heads"], binding["d"]]


def config(binding, dtype):
    t, heads, d = binding["t"], binding["heads"], binding["d"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [t, heads, d], "dtype": dt},
            "tau": {"shape": [t], "dtype": "torch.float32", "note": "per-row"},
        },
        "outputs": {
            "out": {"shape": [t, heads, d], "dtype": dt},
        },
        "dims": {"t": t, "heads": heads, "d": d},
    }
