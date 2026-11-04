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

"""selective_state_update baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.mamba.triton_ops.mamba_ssm.selective_state_update
    调用形态复刻 problems/mamba/selective_state_update/baseline.py：
        out = empty_like(x)
        selective_state_update(state, x, dt, A, B, C, D=D, z=z,
                               dt_bias=dt_bias, dt_softplus=dt_softplus, out=out)
    kernel 原地更新 state 并写 out（baseline 先 clone state、现造 out）。此处每次
    采样重建 state/out，故对外部无副作用；IS_INPLACE 标 True 以便采样前重建实参。

输入构造复刻 problems/mamba/selective_state_update/cases.py 的 BENCH_CASES
（D、dt_bias 恒为实张量；两条 BENCH 都 has_z=True、dt_softplus=True、ngroups=8）：
    state = randn(batch, nheads, dim, dstate).to(dtype)
    x/dt  = randn(batch, nheads, dim).to(dtype)
    A = -(rand(nheads, dim, dstate)+0.1)          # fp32
    B/C = randn(batch, ngroups, dstate).to(dtype)
    D = randn(nheads, dim)（fp32）; z = randn(batch, nheads, dim).to(dtype)
    dt_bias = randn(nheads, dim)（fp32）
dtype 取 benchmark/test_selective_state_update.py 的 [bfloat16]。
"""

import torch

OP_NAME = "selective_state_update"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 原地更新 state、写 out，采样前需重建

# (batch, nheads, dim, dstate, ngroups)
_SHAPES = [
    (64, 32, 128, 128, 8),
    (256, 64, 64, 128, 8),
]


def native():
    """解析 sglang.kernels.ops.mamba.triton_ops.mamba_ssm.selective_state_update。"""
    try:
        from sglang.kernels.ops.mamba.triton_ops.mamba_ssm import (
            selective_state_update as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "batch": b,
            "nheads": nh,
            "dim": dim,
            "dstate": ds,
            "ngroups": ng,
        }
        for (b, nh, dim, ds, ng) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    batch = binding["batch"]
    nheads = binding["nheads"]
    dim = binding["dim"]
    dstate = binding["dstate"]
    ngroups = binding["ngroups"]
    g = torch.Generator(device=device).manual_seed(0)
    state = torch.randn(
        batch, nheads, dim, dstate, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    x = torch.randn(batch, nheads, dim, generator=g, device=device, dtype=torch.float32).to(
        dtype
    )
    dt = torch.randn(batch, nheads, dim, generator=g, device=device, dtype=torch.float32).to(
        dtype
    )
    a = (
        -torch.rand(nheads, dim, dstate, generator=g, device=device, dtype=torch.float32)
        - 0.1
    )
    b = torch.randn(
        batch, ngroups, dstate, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    c = torch.randn(
        batch, ngroups, dstate, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    d = torch.randn(nheads, dim, generator=g, device=device, dtype=torch.float32)
    z = torch.randn(batch, nheads, dim, generator=g, device=device, dtype=torch.float32).to(
        dtype
    )
    bias = torch.randn(nheads, dim, generator=g, device=device, dtype=torch.float32)

    out = torch.empty_like(x)
    return (state, x, dt, a, b, c), {
        "D": d,
        "z": z,
        "dt_bias": bias,
        "dt_softplus": True,
        "out": out,
    }


def key_shape(binding):
    return (
        f"b{binding['batch']}_nh{binding['nheads']}_dim{binding['dim']}"
        f"_ds{binding['dstate']}_ng{binding['ngroups']}"
    )


def config(binding, dtype):
    batch = binding["batch"]
    nheads = binding["nheads"]
    dim = binding["dim"]
    dstate = binding["dstate"]
    ngroups = binding["ngroups"]
    dt = str(dtype)
    return {
        "inputs": {
            "state": {"shape": [batch, nheads, dim, dstate], "dtype": dt,
                      "note": "原地更新"},
            "x": {"shape": [batch, nheads, dim], "dtype": dt},
            "dt": {"shape": [batch, nheads, dim], "dtype": dt},
            "A": {"shape": [nheads, dim, dstate], "dtype": "torch.float32"},
            "B": {"shape": [batch, ngroups, dstate], "dtype": dt},
            "C": {"shape": [batch, ngroups, dstate], "dtype": dt},
            "D": {"shape": [nheads, dim], "dtype": "torch.float32"},
            "z": {"shape": [batch, nheads, dim], "dtype": dt},
            "dt_bias": {"shape": [nheads, dim], "dtype": "torch.float32"},
            "dt_softplus": {"scalar": True},
            "out": {"shape": [batch, nheads, dim], "dtype": dt, "note": "原地写回"},
        },
        "outputs": {
            "out": {"shape": [batch, nheads, dim], "dtype": dt},
            "state": {"shape": [batch, nheads, dim, dstate], "dtype": dt,
                      "note": "原地更新后的 state"},
        },
        "dims": {
            "batch": batch, "nheads": nheads, "dim": dim,
            "dstate": dstate, "ngroups": ngroups,
        },
    }
