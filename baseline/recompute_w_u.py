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

"""recompute_w_u baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.fla.wy_fast.recompute_w_u_fwd
    调用形态复刻 problems/fla/recompute_w_u/baseline.py：
        recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens)
    返回 (w, u)，不原地写回；IS_INPLACE=False。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    k = randn(b, t, hg, k_dim).bf16；v = randn(b, t, h, v_dim).bf16
    beta = randn(b, t, h).sigmoid() fp32；g_cumsum = -randn(b, t, h).abs() fp32
    A = randn(b, t, h, bt).bf16；cu_seqlens=None；k_dim=v_dim=128, bt=64。
    BENCH: (1, t, 16, 4) for t in (64,256,1024,4096)（h=16, hg=4）。seed=9。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "recompute_w_u"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_K_DIM = 128
_V_DIM = 128
_BT = 64
# (b, t, h, hg)
_CASES = [(1, t, 16, 4) for t in (64, 256, 1024, 4096)]


def native():
    """解析 sglang.kernels.ops.attention.fla.wy_fast.recompute_w_u_fwd。"""
    try:
        from sglang.kernels.ops.attention.fla.wy_fast import recompute_w_u_fwd as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"b": b, "t": t, "h": h, "hg": hg, "k_dim": _K_DIM, "v_dim": _V_DIM, "bt": _BT}
        for (b, t, h, hg) in _CASES
    ]


def build_inputs(binding, dtype, device):
    b, t, h, hg = binding["b"], binding["t"], binding["h"], binding["hg"]
    k_dim, v_dim, bt = binding["k_dim"], binding["v_dim"], binding["bt"]
    g = torch.Generator(device=device).manual_seed(9)

    def r(*shape, dt=torch.bfloat16):
        return torch.randn(*shape, device=device, generator=g).to(dt)

    k = r(b, t, hg, k_dim)
    v = r(b, t, h, v_dim)
    beta = r(b, t, h, dt=torch.float32).sigmoid()
    g_cumsum = -r(b, t, h, dt=torch.float32).abs()
    A = r(b, t, h, bt)
    return (k, v, beta, g_cumsum, A, None), {}


def key_shape(binding):
    return "b%d_t%d_h%d_hg%d_k%d_v%d_bt%d" % (
        binding["b"], binding["t"], binding["h"], binding["hg"],
        binding["k_dim"], binding["v_dim"], binding["bt"],
    )


def config(binding, dtype):
    b, t, h, hg = binding["b"], binding["t"], binding["h"], binding["hg"]
    k_dim, v_dim, bt = binding["k_dim"], binding["v_dim"], binding["bt"]
    dt = str(dtype)
    return {
        "inputs": {
            "k": {"shape": [b, t, hg, k_dim], "dtype": dt},
            "v": {"shape": [b, t, h, v_dim], "dtype": dt},
            "beta": {"shape": [b, t, h], "dtype": "torch.float32"},
            "g_cumsum": {"shape": [b, t, h], "dtype": "torch.float32"},
            "A": {"shape": [b, t, h, bt], "dtype": dt},
            "cu_seqlens": {"scalar": None},
        },
        "outputs": {
            "w": {"shape": [b, t, h, k_dim], "dtype": dt},
            "u": {"shape": [b, t, h, v_dim], "dtype": dt},
        },
        "dims": {
            "b": b, "t": t, "h": h, "hg": hg,
            "k_dim": k_dim, "v_dim": v_dim, "bt": bt,
        },
    }
