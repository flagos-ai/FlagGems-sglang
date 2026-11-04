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

"""fused_recurrent_gdn baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.fla.fused_recurrent.fused_recurrent_gated_delta_rule_fwd
    调用形态复刻 problems/fla/fused_recurrent_gdn/baseline.py：
        fused_recurrent_gated_delta_rule_fwd(
            q, k, v, g, beta, scale, initial_state, output_final_state,
            use_qk_l2norm_in_kernel=...)
    返回 (o, final_state)，不原地写回输入；IS_INPLACE=False。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    q/k = randn(batch, t, h, k_dim).bf16；v = randn(batch, t, hv, v_dim).bf16
    g = -rand(batch, t, hv)*0.1 fp32；beta = sigmoid(randn(batch, t, hv)) fp32
    initial_state=None（has_init=False），output_final_state=True，l2norm=False
    scale = k_dim**-0.5；BENCH: (8,128,8,8,64,64) 与 (32,32,8,8,64,64)。seed=0。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_recurrent_gdn"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# (batch, t, h, hv, k_dim, v_dim)
_CASES = [
    (8, 128, 8, 8, 64, 64),
    (32, 32, 8, 8, 64, 64),
]


def native():
    """解析 fla.fused_recurrent.fused_recurrent_gated_delta_rule_fwd。"""
    try:
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_fwd as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": b, "t": t, "h": h, "hv": hv, "k_dim": k, "v_dim": v}
        for (b, t, h, hv, k, v) in _CASES
    ]


def build_inputs(binding, dtype, device):
    batch, t, h, hv = binding["batch"], binding["t"], binding["h"], binding["hv"]
    k_dim, v_dim = binding["k_dim"], binding["v_dim"]
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(batch, t, h, k_dim, generator=g, device=device, dtype=torch.float32).to(dtype)
    k = torch.randn(batch, t, h, k_dim, generator=g, device=device, dtype=torch.float32).to(dtype)
    v = torch.randn(batch, t, hv, v_dim, generator=g, device=device, dtype=torch.float32).to(dtype)
    gate = -torch.rand(batch, t, hv, generator=g, device=device, dtype=torch.float32) * 0.1
    beta = torch.sigmoid(
        torch.randn(batch, t, hv, generator=g, device=device, dtype=torch.float32)
    )
    scale = k_dim ** -0.5
    return (q, k, v, gate, beta, scale, None, True), {"use_qk_l2norm_in_kernel": False}


def key_shape(binding):
    return "b%d_t%d_h%d_hv%d_k%d_v%d" % (
        binding["batch"], binding["t"], binding["h"],
        binding["hv"], binding["k_dim"], binding["v_dim"],
    )


def config(binding, dtype):
    batch, t, h, hv = binding["batch"], binding["t"], binding["h"], binding["hv"]
    k_dim, v_dim = binding["k_dim"], binding["v_dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [batch, t, h, k_dim], "dtype": dt},
            "k": {"shape": [batch, t, h, k_dim], "dtype": dt},
            "v": {"shape": [batch, t, hv, v_dim], "dtype": dt},
            "g": {"shape": [batch, t, hv], "dtype": "torch.float32"},
            "beta": {"shape": [batch, t, hv], "dtype": "torch.float32"},
            "scale": {"scalar": k_dim ** -0.5},
            "initial_state": {"scalar": None},
            "output_final_state": {"scalar": True},
            "use_qk_l2norm_in_kernel": {"scalar": False},
        },
        "outputs": {
            "o": {"shape": [batch, t, hv, v_dim], "dtype": dt},
            "final_state": {"shape": [batch, hv, v_dim, k_dim], "dtype": "torch.float32"},
        },
        "dims": {
            "batch": batch, "t": t, "h": h, "hv": hv, "k_dim": k_dim, "v_dim": v_dim,
        },
    }
