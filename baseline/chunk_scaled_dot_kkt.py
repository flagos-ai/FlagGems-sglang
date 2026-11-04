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

"""chunk_scaled_dot_kkt baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.attention.fla.kda.chunk_kda_scaled_dot_kkt_fwd
    调用形态复刻 problems/fla/chunk_kda_scaled_dot_kkt/baseline.py：
        chunk_kda_scaled_dot_kkt_fwd(
            q, k, gk, beta, scale=scale, gk_scale=gk_scale, chunk_size=chunk_size)
    返回新张量，不原地写回。

输入构造复刻 problems/fla/chunk_kda_scaled_dot_kkt/cases.py 的 BENCH_CASES：
    q/k = randn(b, t, h, kdim).to(dtype).contiguous()
    gk = within-chunk cumsum of (-0.05 * rand) 后 reshape 回 (b, t, h, kdim)
    beta = rand(b, t, h)
    scale = kdim**-0.5, gk_scale=1.0, chunk_size=64
dtype 取 benchmark 的 [bfloat16]。注意：这里采用 cases.py 的真实 native 签名
(q, k, gk, beta, ...)，与 task .md 的 reference 签名不同。
"""

import torch

OP_NAME = "chunk_scaled_dot_kkt"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_CHUNK_SIZE = 64
# BENCH_CASES: _case(1, t, 8, 128) for t in (64, 256, 1024, 4096)
_TS = [64, 256, 1024, 4096]


def native():
    """解析 sglang.kernels.ops.attention.fla.kda.chunk_kda_scaled_dot_kkt_fwd。"""
    try:
        from sglang.kernels.ops.attention.fla.kda import (
            chunk_kda_scaled_dot_kkt_fwd as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"b": 1, "t": t, "h": 8, "kdim": 128} for t in _TS]


def build_inputs(binding, dtype, device):
    b, t, h, kdim = binding["b"], binding["t"], binding["h"], binding["kdim"]
    cs = _CHUNK_SIZE
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(b, t, h, kdim, dtype=torch.float32, device=device, generator=g).to(dtype)
    k = torch.randn(b, t, h, kdim, dtype=torch.float32, device=device, generator=g).to(dtype)
    gate = -0.05 * torch.rand(
        b, t, h, kdim, dtype=torch.float32, device=device, generator=g
    )
    gk = gate.reshape(b, -1, cs, h, kdim).cumsum(dim=2).reshape(b, t, h, kdim)
    beta = torch.rand(b, t, h, dtype=torch.float32, device=device, generator=g)
    return (
        q.contiguous(),
        k.contiguous(),
        gk.contiguous(),
        beta.contiguous(),
    ), {"scale": kdim ** -0.5, "gk_scale": 1.0, "chunk_size": cs}


def key_shape(binding):
    return [binding["b"], binding["t"], binding["h"], binding["kdim"]]


def config(binding, dtype):
    b, t, h, kdim = binding["b"], binding["t"], binding["h"], binding["kdim"]
    cs = _CHUNK_SIZE
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [b, t, h, kdim], "dtype": dt},
            "k": {"shape": [b, t, h, kdim], "dtype": dt},
            "gk": {"shape": [b, t, h, kdim], "dtype": dt},
            "beta": {"shape": [b, t, h], "dtype": "torch.float32"},
            "scale": {"scalar": kdim ** -0.5},
            "gk_scale": {"scalar": 1.0},
            "chunk_size": {"scalar": cs},
        },
        "outputs": {
            "A": {"shape": [b, t, h, cs], "dtype": "torch.float32",
                  "note": "逐 chunk 的 [BT, BT] 严格下三角修正矩阵"},
        },
        "dims": {"b": b, "t": t, "h": h, "kdim": kdim, "chunk_size": cs},
    }
